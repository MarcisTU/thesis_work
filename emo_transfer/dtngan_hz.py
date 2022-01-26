import glob
import argparse # pip3 install argparse
import itertools
from copy import copy

from torch.hub import download_url_to_file
from torchvision.transforms import InterpolationMode
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
import numpy as np
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import random
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from PIL import Image

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data

# aequuiPhiar7779
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='output_hz', type=str)
parser.add_argument('-data_dir', type=str, default='../data/', help='Directory for dataset.')
parser.add_argument('-num_epochs', default=5000, type=int)
parser.add_argument('-batch_size', default=4, type=int)
parser.add_argument('-learning_rate_g', default=5e-5, type=float)
parser.add_argument('-learning_rate_d', default=5e-5, type=float)
parser.add_argument('-z_size', default=512, type=int)
parser.add_argument('-coef_alpha', default=10, type=float)
parser.add_argument('-coef_beta', default=2, type=float)
parser.add_argument("--local_rank", default=0, type=int)

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = 4
EPOCHS = args.num_epochs
LEARNING_RATE_D = args.learning_rate_d
LEARNING_RATE_G = args.learning_rate_g
Z_SIZE = args.z_size
DATA_DIR = args.data_dir
INPUT_SIZE = 256
DISCRIMINATOR_N = 4

COEF_ALPHA = args.coef_alpha
COEF_BETA = args.coef_beta

if args.local_rank == 0:
    if len(RUN_PATH):
        RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
        if os.path.exists(RUN_PATH):
            shutil.rmtree(RUN_PATH)
        os.makedirs(RUN_PATH)


class ImageDataset(Dataset):
    def __init__(self, root_dir, type, transform=None, mode='train'):
        self.transform = torchvision.transforms.Compose(transform)
        self.type = type
        self.train = (mode == 'train')

        self.files = sorted(glob.glob(os.path.join(root_dir, f'{mode}{self.type}') + '/*.*'))

    def __getitem__(self, index):
        idx = index
        img = Image.open(self.files[idx % len(self.files)]).convert('RGB')
        item = self.transform(img)
        return item

    def __len__(self):
        return len(self.files)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, decay_rate=0.0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        block = [nn.ReflectionPad2d(padding=1)]
        block.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)))
        block.append(nn.InstanceNorm2d(num_features=channels))
        block.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # if dp_rate > 0.0:
        #     block.append(nn.Dropout2d(p=dp_rate))
        block.append(nn.ReflectionPad2d(padding=1))
        block.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)))
        block.append(nn.InstanceNorm2d(num_features=channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block.forward(x)


class Encoder(nn.Module):
    def __init__(self, channels, num_block=3):
        super().__init__()
        self.channels = channels

        # downsample
        model = [nn.ReflectionPad2d(padding=3)]
        model += self._create_layer(in_channels=self.channels, out_channels=64,
                                    kernel_size=7, stride=1, padding=0)
        model += self._create_layer(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        model += self._create_layer(in_channels=64, out_channels=128)
        model += self._create_layer(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        model += self._create_layer(in_channels=128, out_channels=256)
        model += [ResidualBlock(channels=256) for _ in range(num_block)]

        self.model = nn.Sequential(*model)

    def _create_layer(self, in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, normalize=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding))
        if normalize:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # if dp_rate > 0.0:
        #     layers.append(nn.Dropout2d(p=dp_rate))
        return layers

    def forward(self, x):
        return self.model.forward(x)


class Discriminator(nn.Module):
    def __init__(self, channels, std=0.1, std_decay_rate=0.0):
        super().__init__()
        self.std = std
        self.std_decay_rate = std_decay_rate
        self.channels = channels

        self.model = nn.Sequential(
            # out = (in + 2p - k)/s + 1
            *self._create_layer(in_channels=self.channels, out_channels=8, normalize=False), # out 128
            *self._create_layer(in_channels=8, out_channels=16),  # out 64
            *self._create_layer(in_channels=16, out_channels=32),  # out 32
            *self._create_layer(in_channels=32, out_channels=64),  # out 16
            *self._create_layer(in_channels=64, out_channels=128),  # out 8
            *self._create_layer(in_channels=128, out_channels=128),  # out 4
            *self._create_layer(in_channels=128, out_channels=256)  # out 2
        )
        self.mlp = torch.nn.Sequential(
            GaussianNoise(std=self.std, decay_rate=self.std_decay_rate),
            torch.nn.Linear(in_features=1024, out_features=3),
            torch.nn.Softmax(dim=-1)
        )

    def _create_layer(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [GaussianNoise(std=self.std, decay_rate=self.std_decay_rate)]
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding))
        if normalize:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return layers

    def forward(self, x):
        x_enc = self.model.forward(x)
        x_enc_flat = x_enc.view(x.size(0), -1)
        y_prim = self.mlp.forward(x_enc_flat)
        return y_prim


class Generator(nn.Module):
    def __init__(self, channels, num_block=3):
        super().__init__()
        self.channels = channels

        # upsample
        model = [ResidualBlock(channels=256) for _ in range(num_block)]
        model += self._create_layer(in_channels=256, out_channels=128)
        model += self._create_layer(in_channels=128, out_channels=128, upsample=False)
        model += self._create_layer(in_channels=128, out_channels=64)
        model += self._create_layer(in_channels=64, out_channels=64, upsample=False)
        # output
        model += [nn.ReflectionPad2d(padding=3),
                  nn.Conv2d(in_channels=64, out_channels=self.channels, kernel_size=7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def _create_layer(self, in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, upsample=True):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # if dp_rate > 0.0:
        #     layers.append(nn.Dropout2d(p=dp_rate))
        return layers

    def forward(self, x):
        return self.model.forward(x)


# class ImageBuffer(object):
#     def __init__(self, depth=20):
#         self.depth = depth
#         self.buffer = []
#
#     def update(self, image):
#         if len(self.buffer) == self.depth:
#             i = random.randint(0, self.depth-1)
#             self.buffer[i] = image
#         else:
#             self.buffer.append(image)
#         if random.uniform(0,1) > 0.5:
#             i = random.randint(0, len(self.buffer)-1)
#             return self.buffer[i]
#         else:
#             return image


torch.distributed.init_process_group(backend='nccl')
DEVICE = torch.device('cuda', args.local_rank)
model_G = Generator(channels=3).to(DEVICE)
model_G = torch.nn.parallel.DistributedDataParallel(model_G,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
model_D = Discriminator(channels=3, std=0.1, std_decay_rate=0.01).to(DEVICE)
model_D = torch.nn.parallel.DistributedDataParallel(model_D,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
model_E = Encoder(channels=3).to(DEVICE)
model_E = torch.nn.parallel.DistributedDataParallel(model_E,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

transform = [transforms.Resize(INPUT_SIZE, InterpolationMode.BICUBIC),
                     transforms.RandomCrop((INPUT_SIZE, INPUT_SIZE)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
dataset_source = ImageDataset(os.path.join(DATA_DIR, "horse2zebra"), transform=transform, mode='train', type='A')
dataset_target = ImageDataset(os.path.join(DATA_DIR, "horse2zebra"), transform=transform, mode='train', type='B')

sampler_source_train = DistributedSampler(dataset_source)
sampler_target_train = DistributedSampler(dataset_target)

dataloader_source = DataLoader(dataset_source, batch_size=BATCH_SIZE, sampler=sampler_source_train, num_workers=8)
dataloader_target = DataLoader(dataset_target, batch_size=BATCH_SIZE, sampler=sampler_target_train, num_workers=8)
if args.local_rank == 0: # only create on one gpu
    test_dataset_source = ImageDataset(os.path.join(DATA_DIR, "horse2zebra"), transform=transform, mode='test', type='A')
    sampler_source_test = DistributedSampler(test_dataset_source)
    test_dataloader_source = DataLoader(test_dataset_source, batch_size=TEST_BATCH_SIZE, sampler=sampler_source_test, num_workers=8)

optimizer_D = torch.optim.Adam(model_D.parameters(), lr=LEARNING_RATE_D)
optimizer_G = torch.optim.Adam(list(model_G.parameters()) + list(model_E.parameters()), lr=LEARNING_RATE_G)

# image_buffer_s = ImageBuffer()
# image_buffer_t = ImageBuffer()

def decay_gauss_std(net):
    for m in net.modules():
        if isinstance(m, GaussianNoise):
            m.decay_step()

metrics = {}
for stage in ['train']:
    for metric in ['loss_g', 'loss_d', 'loss_gang', 'loss_const', 'loss_tid']:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS):
    sampler_target_train.set_epoch(epoch)
    sampler_source_train.set_epoch(epoch)
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    iter_data_loader_target = iter(dataloader_target)
    n = 0
    for x_s in tqdm(dataloader_source, desc=stage):
        x_t = next(iter_data_loader_target)
        if x_s.size() != x_t.size():
            continue

        x_s = x_s.to(DEVICE)
        x_t = x_t.to(DEVICE)

        if n % DISCRIMINATOR_N == 0:
            optimizer_G.zero_grad()
            z_s = model_E.forward(x_s)
            z_t = model_E.forward(x_t)
            g_s = model_G.forward(z_s)
            g_t = model_G.forward(z_t)

            z_z_s = model_E.forward(g_s)

            for param in model_D.parameters():
                param.requires_grad = False

            y_g_s = model_D.forward(g_s)
            y_g_t = model_D.forward(g_t)

            loss_gang = -torch.mean(torch.log(y_g_s[:, 2] + 1e-8)) - torch.mean(torch.log(y_g_t[:, 2] + 1e-8))
            loss_const = torch.mean((z_s - z_z_s) ** 2) * COEF_ALPHA  # can be cosine distance, l1 ..
            loss_tid = torch.mean((x_t - g_t) ** 2) * COEF_BETA  # reconstruction
            loss_g = loss_gang + loss_const + loss_tid

            loss_g.backward()
            optimizer_G.step()

            metrics_epoch[f'{stage}_loss_gang'].append(loss_gang.cpu().item())
            metrics_epoch[f'{stage}_loss_const'].append(loss_const.cpu().item())
            metrics_epoch[f'{stage}_loss_tid'].append(loss_tid.cpu().item())
            metrics_epoch[f'{stage}_loss_g'].append(loss_g.cpu().item())
        else:
            optimizer_D.zero_grad()
            z_s = model_E.forward(x_s)
            z_t = model_E.forward(x_t)
            g_s = model_G.forward(z_s)
            g_t = model_G.forward(z_t)
            # g_s = image_buffer_s.update(g_s)
            # g_t = image_buffer_t.update(g_t)

            for param in model_D.parameters():
                param.requires_grad = True

            y_g_s = model_D.forward(g_s.detach())
            y_g_t = model_D.forward(g_t.detach())
            y_t = model_D.forward(x_t)

            # model_D = [0 = fake_source, 1 = fake_target, 2 = real_target]
            loss_d = -torch.mean(torch.log(y_g_s[:, 0] + 1e-8)) \
                     - torch.mean(torch.log(y_g_t[:, 1] + 1e-8)) \
                     - torch.mean(torch.log(y_t[:, 2] + 1e-8))

            loss_d.backward()
            optimizer_D.step()

            metrics_epoch[f'{stage}_loss_d'].append(loss_d.cpu().item())
        n += 1

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    # print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 4 == 0 and args.local_rank == 0:
        # sampler_target_test.set_epoch(epoch)
        sampler_source_test.set_epoch(epoch)
        with torch.no_grad():
            imgs_s = next(iter(test_dataloader_source))
            real_s = imgs_s.to(DEVICE)
            enc_s = model_E.forward(real_s)
            gen_s = model_G.forward(enc_s)
            viz_sample = torch.cat((real_s, gen_s), 0)
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

    # Decay gaussian noise
    # decay_gauss_std(model_D)
