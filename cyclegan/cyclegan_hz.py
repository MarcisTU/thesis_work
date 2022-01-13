import glob
import argparse # pip3 install argparse
import itertools
from copy import copy

from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import hashlib
import os
import pickle
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import torchvision
import random
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from PIL import Image

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)
plt.style.use('dark_background')

import torch.utils.data

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='output_hz', type=str)
parser.add_argument('-data_dir', type=str, default='../data/', help='Directory for dataset.')
parser.add_argument('-num_epochs', default=2000, type=int)
parser.add_argument('-batch_size', default=4, type=int)
parser.add_argument('-learning_rate_g', default=2e-4, type=float)
parser.add_argument('-learning_rate_d', default=2e-4, type=float)
parser.add_argument('-z_size', default=512, type=int)
parser.add_argument('-lambda_cyc', default=10, type=float)
parser.add_argument('-lambda_iden', default=5, type=float)
parser.add_argument("--local_rank", default=0, type=int)

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = 4
EPOCHS = args.num_epochs
Z_SIZE = args.z_size
# DEVICE = 'cuda'
DATA_DIR = args.data_dir
INPUT_SIZE = 256

if args.local_rank == 0:
    if len(RUN_PATH):
        RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
        if os.path.exists(RUN_PATH):
            shutil.rmtree(RUN_PATH)
        os.makedirs(RUN_PATH)

# if not torch.cuda.is_available():
#     DEVICE = 'cpu'

class ImageDataset(Dataset):
    def __init__(self, root_dir, type, transform=None, unaligned=False, mode='train'):
        self.transform = torchvision.transforms.Compose(transform)
        self.unaligned = unaligned
        self.type = type
        self.train = (mode == 'train')

        self.files = sorted(glob.glob(os.path.join(root_dir, f'{mode}{self.type}') + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = img.convert('RGB')
        item = self.transform(img)
        return item

    def __len__(self):
        return len(self.files)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(num_features=channels)
        )

    def forward(self, x):
        return x + self.block.forward(x)


class Generator(nn.Module):
    def __init__(self, channels, num_block=9):
        super().__init__()
        self.channels = channels

        model = [nn.ReflectionPad2d(padding=3)]
        model += self._create_layer(in_channels=self.channels, out_channels=64, kernel_size=7, stride=1, padding=0)
        # downsample
        model += self._create_layer(in_channels=64, out_channels=128, kernel_size=3)
        model += self._create_layer(in_channels=128, out_channels=256, kernel_size=3)
        # residual blocks
        model += [ResidualBlock(channels=256) for _ in range(num_block)]
        # upsample
        model += self._create_layer(in_channels=256, out_channels=128, kernel_size=3, transposed=True)
        model += self._create_layer(in_channels=128, out_channels=64, kernel_size=3, transposed=True)
        # output
        model += [nn.ReflectionPad2d(padding=3),
                  nn.Conv2d(in_channels=64, out_channels=self.channels, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def _create_layer(self, in_channels, out_channels, kernel_size, stride=2, padding=1, transposed=False):
        layers = []
        if transposed:
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                   stride, padding, output_padding=(1, 1)))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.ReLU(inplace=True))
        return layers

    def forward(self, x):
        return self.model.forward(x)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.model = nn.Sequential(
            *self._create_layer(in_channels=self.channels, out_channels=64, stride=2, normalize=False),
            *self._create_layer(in_channels=64, out_channels=128, stride=2),
            *self._create_layer(in_channels=128, out_channels=256, stride=2),
            *self._create_layer(in_channels=256, out_channels=512, stride=1),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)
        )

    def _create_layer(self, in_channels, out_channels, stride, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return layers

    def forward(self, x):
        return self.model.forward(x)


class ImageBuffer(object):
    def __init__(self, depth=50):
        self.depth = depth
        self.buffer = []

    def update(self, image):
        if len(self.buffer) == self.depth:
            i = random.randint(0, self.depth-1)
            self.buffer[i] = image
        else:
            self.buffer.append(image)
        if random.uniform(0,1) > 0.5:
            i = random.randint(0, len(self.buffer)-1)
            return self.buffer[i]
        else:
            return image


torch.distributed.init_process_group(backend='nccl')
DEVICE = torch.device('cuda', args.local_rank)
model_G_s_t = Generator(channels=3).to(DEVICE)
model_G_t_s = Generator(channels=3).to(DEVICE)
model_G_s_t = torch.nn.parallel.DistributedDataParallel(model_G_s_t,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
model_G_t_s = torch.nn.parallel.DistributedDataParallel(model_G_t_s,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
model_D_s = Discriminator(channels=3).to(DEVICE)
model_D_t = Discriminator(channels=3).to(DEVICE)
model_D_s = torch.nn.parallel.DistributedDataParallel(model_D_s,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
model_D_t = torch.nn.parallel.DistributedDataParallel(model_D_t,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

transform = [transforms.Resize(int(INPUT_SIZE*1.12), Image.BICUBIC),
                     transforms.RandomCrop((INPUT_SIZE, INPUT_SIZE)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5), (0.5))]
dataset_source = ImageDataset(os.path.join(DATA_DIR, "horse2zebra"), transform=transform, unaligned=True, mode='train', type='A')
dataset_target = ImageDataset(os.path.join(DATA_DIR, "horse2zebra"), transform=transform, unaligned=True, mode='train', type='B')
test_dataset_source = ImageDataset(os.path.join(DATA_DIR, "horse2zebra"), transform=transform, unaligned=True, mode='test', type='A')
test_dataset_target = ImageDataset(os.path.join(DATA_DIR, "horse2zebra"), transform=transform, unaligned=True, mode='test', type='B')

sampler_source_train = DistributedSampler(dataset_source)
sampler_target_train = DistributedSampler(dataset_target)
sampler_source_test = DistributedSampler(test_dataset_source)
sampler_target_test = DistributedSampler(test_dataset_target)

dataloader_source = torch.utils.data.DataLoader(dataset_source, batch_size=BATCH_SIZE, sampler=sampler_source_train, num_workers=8)
dataloader_target = torch.utils.data.DataLoader(dataset_target, batch_size=BATCH_SIZE, sampler=sampler_target_train, num_workers=8)
test_dataloader_source = DataLoader(test_dataset_source, batch_size=TEST_BATCH_SIZE, sampler=sampler_source_test, num_workers=8)
test_dataloader_target = DataLoader(test_dataset_target, batch_size=TEST_BATCH_SIZE, sampler=sampler_target_test, num_workers=8)

params_G = itertools.chain(model_G_s_t.parameters(), model_G_t_s.parameters())
optimizer_G = torch.optim.Adam(params_G, lr=args.learning_rate_g)
optimizer_D_s = torch.optim.Adam(model_D_s.parameters(), lr=args.learning_rate_d, betas=(0.5, 0.999))
optimizer_D_t = torch.optim.Adam(model_D_t.parameters(), lr=args.learning_rate_d, betas=(0.5, 0.999))

image_buffer_s = ImageBuffer()
image_buffer_t = ImageBuffer()

metrics = {}
for stage in ['train']:
    for metric in [
        'loss_g',
        'loss_d',
        'loss_c',
        'loss_i'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS):
    sampler_target_train.set_epoch(epoch)
    sampler_source_train.set_epoch(epoch)
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    iter_data_loader_target = iter(dataloader_target)

    for x_s in tqdm(dataloader_source, desc=stage):
        x_t = next(iter_data_loader_target)

        if x_s.size() != x_t.size():
            continue
        x_s = x_s.to(DEVICE) # source
        x_t = x_t.to(DEVICE) # target

        optimizer_G.zero_grad()
        ### Train G
        g_t = model_G_s_t.forward(x_s)
        g_s = model_G_t_s.forward(x_t)

        # generator loss
        y_g_t = model_D_t.forward(g_t)  # 1 = real, 0 = fake
        y_g_s = model_D_s.forward(g_s)
        loss_g_t = torch.mean((y_g_t - 1.0) ** 2)
        loss_g_s = torch.mean((y_g_s - 1.0) ** 2)
        loss_g = (loss_g_t + loss_g_s) / 2

        # cycle loss
        recov_s = model_G_t_s.forward(g_t)
        loss_c_s = torch.mean(torch.abs(recov_s - x_s))
        recov_t = model_G_s_t.forward(g_s)
        loss_c_t = torch.mean(torch.abs(recov_t - x_t))
        loss_c = (loss_c_s + loss_c_t) / 2

        # identity loss
        i_t = model_G_s_t.forward(x_t)
        i_s = model_G_t_s.forward(x_s)
        loss_i_t = torch.mean(torch.abs(i_t - x_t))
        loss_i_s = torch.mean(torch.abs(i_s - x_s))
        loss_i = (loss_i_s + loss_i_t) / 2

        loss = loss_g + loss_c * args.lambda_cyc + loss_i * args.lambda_iden
        loss.backward()
        optimizer_G.step()

        metrics_epoch[f'{stage}_loss_i'].append(loss_i.cpu().item())
        metrics_epoch[f'{stage}_loss_g'].append(loss_g.cpu().item())
        metrics_epoch[f'{stage}_loss_c'].append(loss_c.cpu().item())

        ### Train D_s
        optimizer_D_s.zero_grad()

        y_x_s = model_D_s.forward(x_s)
        loss_x_s = torch.mean((y_x_s - 1.0) ** 2)
        # g_s = model_G_t_s.forward(x_t)
        g_s = image_buffer_s.update(g_s)
        y_g_s = model_D_s.forward(g_s.detach())
        loss_g_s = torch.mean(y_g_s ** 2)

        d_loss_s = (loss_x_s + loss_g_s) / 2
        d_loss_s.backward()
        optimizer_D_s.step()

        ### Train D_s
        optimizer_D_t.zero_grad()

        y_x_t = model_D_t.forward(x_t)
        loss_x_t = torch.mean((y_x_t - 1.0) ** 2)
        # g_t = model_G_s_t.forward(x_s)
        g_t = image_buffer_t.update(g_t)
        y_g_t = model_D_t.forward(g_t.detach())
        loss_g_t = torch.mean(y_g_t ** 2)

        loss_d = (loss_g_t + loss_x_t) / 2
        loss_d.backward()
        optimizer_D_t.step()

        metrics_epoch[f'{stage}_loss_d'].append(loss_d.cpu().item())

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 2 == 0 and args.local_rank == 0:
        sampler_target_test.set_epoch(epoch)
        sampler_source_test.set_epoch(epoch)
        with torch.no_grad():
            imgs_A = next(iter(test_dataloader_source))
            imgs_B = next(iter(test_dataloader_target))
            _real_A = imgs_A.to(DEVICE)
            _fake_B = model_G_s_t.forward(_real_A)
            _real_B = imgs_B.to(DEVICE)
            _fake_A = model_G_t_s.forward(_real_B)
            viz_sample = torch.cat((_real_A, _fake_B, _real_B, _fake_A), 0)
            vutils.save_image(viz_sample,
                              os.path.join(RUN_PATH, 'samples_{}.png'.format(epoch)),
                              nrow=TEST_BATCH_SIZE,
                              normalize=True)
        plt.clf()
        plt.cla()

        plts = []
        c = 0
        for key, value in metrics.items():
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1
        plt.legend(plts, [it.get_label() for it in plts])
        plt.tight_layout(pad=0.5)
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')
