import argparse
import os
import time
import random

import itertools
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import shutil

from loader import Dataset


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='output_emo', type=str)
parser.add_argument('-num_epochs', default=4000, type=int)
parser.add_argument('-batch_size', default=8, type=int)
parser.add_argument('-learning_rate_g', default=5e-5, type=float)
parser.add_argument('-learning_rate_d', default=1e-5, type=float)
parser.add_argument('-z_size', default=512, type=int)
parser.add_argument('-lambda_cyc', default=10, type=float)
parser.add_argument('-lambda_iden', default=5, type=float)

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = 8
EPOCHS = args.num_epochs
Z_SIZE = args.z_size
DEVICE = 'cuda'
INPUT_SIZE = 64
D_ITER = 3

if not torch.cuda.is_available():
    DEVICE = 'cpu'
    BATCH_SIZE = 5

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(num_features=channels),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(num_features=channels)
        )

    def forward(self, x):
        return x + self.block.forward(x)


class Generator(nn.Module):
    def __init__(self, channels, num_block=3):
        super().__init__()
        self.channels = channels

        model = [nn.ReflectionPad2d(padding=3)]
        model += self._create_layer(in_channels=self.channels, out_channels=64, kernel_size=7, stride=1, padding=0)
        # downsample
        model += self._create_layer(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        model += self._create_layer(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        # residual blocks
        model += [ResidualBlock(channels=256) for _ in range(num_block)]
        # upsample
        model += self._create_layer(in_channels=256, out_channels=128, upsample=True)
        model += self._create_layer(in_channels=128, out_channels=64, upsample=True)
        # output
        model += [nn.ReflectionPad2d(padding=3),
                  nn.Conv2d(in_channels=64, out_channels=self.channels, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def _create_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, upsample=False):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))
        return layers

    def forward(self, x):
        return self.model.forward(x)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.model = nn.Sequential(
            # out = (in + 2p - k)/s + 1
            *self._create_layer(in_channels=self.channels, out_channels=32, stride=2, normalize=False), # out 32
            *self._create_layer(in_channels=32, out_channels=64, stride=2), # out 16
            *self._create_layer(in_channels=64, out_channels=128, stride=2), # out 8
            *self._create_layer(in_channels=128, out_channels=128, stride=1), # out 7
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=1) # out 6
        )

    def _create_layer(self, in_channels, out_channels, stride, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))
        return layers

    def forward(self, x):
        return self.model.forward(x)


class ImageBuffer(object):
    def __init__(self, depth=25):
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


model_G_s_t = Generator(channels=1).to(DEVICE)
model_G_t_s = Generator(channels=1).to(DEVICE)
model_D_s = Discriminator(channels=1).to(DEVICE)
model_D_t = Discriminator(channels=1).to(DEVICE)

# def get_param_count(model):
#     params = list(model.parameters())
#     result = 0
#     for param in params:
#         count_param = np.prod(param.size()) # size[0] * size[1] ...
#         result += count_param
#     return result
#
# print(f'model_D params: {get_param_count(model_D_s)}')
# print(f'model_G params: {get_param_count(model_G_s_t)}')
# exit()

transform = torchvision.transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop((64, 64)),
    transforms.RandomHorizontalFlip()
    ])

# source
loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
source_dataset_train = Dataset('../data/fer_48_s.hdf5', mode='train', img_size=48, transform=transform)
dataloader_train_source = data.DataLoader(source_dataset_train, **loader_params)
loader_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
source_dataset_test = Dataset('../data/fer_48_s.hdf5', mode='test', img_size=48, transform=transform)
dataloader_test_source = data.DataLoader(source_dataset_train, **loader_params)

# target
loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
target_dataset_train = Dataset('../data/fer_48_t.hdf5', mode='train', img_size=48, transform=transform)
dataloader_train_target = data.DataLoader(target_dataset_train, **loader_params)
loader_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
target_dataset_test = Dataset('../data/fer_48_t.hdf5', mode='test', img_size=48, transform=transform)
dataloader_test_target = data.DataLoader(target_dataset_test, **loader_params)

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
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    iter_data_loader_target = iter(dataloader_train_target)

    n = 0
    for x_s, label_s in tqdm(dataloader_train_source, desc=stage):
        x_t, label_t = next(iter_data_loader_target)

        if x_s.size() != x_t.size():
            continue
        x_s = x_s.to(DEVICE) # source
        x_t = x_t.to(DEVICE) # target

        if n % D_ITER == 0:
            optimizer_G.zero_grad()
            for param in model_D_s.parameters():
                param.requires_grad = False
            for param in model_D_t.parameters():
                param.requires_grad = False
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
            loss_c_s = torch.mean(torch.abs(recov_s - x_s)) * args.lambda_cyc
            recov_t = model_G_s_t.forward(g_s)
            loss_c_t = torch.mean(torch.abs(recov_t - x_t)) * args.lambda_cyc
            loss_c = (loss_c_s + loss_c_t) / 2

            # identity loss
            i_t = model_G_s_t.forward(x_t)
            i_s = model_G_t_s.forward(x_s)
            loss_i_t = torch.mean(torch.abs(i_t - x_t)) * args.lambda_iden
            loss_i_s = torch.mean(torch.abs(i_s - x_s)) * args.lambda_iden
            loss_i = (loss_i_s + loss_i_t) / 2

            loss = loss_g + loss_c + loss_i
            loss.backward()
            optimizer_G.step()

            metrics_epoch[f'{stage}_loss_i'].append(loss_i.cpu().item())
            metrics_epoch[f'{stage}_loss_g'].append(loss_g.cpu().item())
            metrics_epoch[f'{stage}_loss_c'].append(loss_c.cpu().item())
        else:
            ### Train D_s
            for param in model_D_s.parameters():
                param.requires_grad = True
            optimizer_D_s.zero_grad()

            y_x_s = model_D_s.forward(x_s)
            loss_x_s = torch.mean((y_x_s - 1.0) ** 2)
            g_s = model_G_t_s.forward(x_t)
            g_s = image_buffer_s.update(g_s)
            y_g_s = model_D_s.forward(g_s.detach())
            loss_g_s = torch.mean(y_g_s ** 2)

            loss_ds = (loss_x_s + loss_g_s) / 2
            loss_ds.backward()
            optimizer_D_s.step()

            ### Train D_t
            for param in model_D_t.parameters():
                param.requires_grad = True
            optimizer_D_t.zero_grad()

            y_x_t = model_D_t.forward(x_t)
            loss_x_t = torch.mean((y_x_t - 1.0) ** 2)
            g_t = model_G_s_t.forward(x_s)
            g_t = image_buffer_t.update(g_t)
            y_g_t = model_D_t.forward(g_t.detach())
            loss_g_t = torch.mean(y_g_t ** 2)

            loss_dt = (loss_g_t + loss_x_t) / 2
            loss_dt.backward()
            optimizer_D_t.step()

            loss_d = loss_ds + loss_dt

            metrics_epoch[f'{stage}_loss_d'].append(loss_d.cpu().item())
        n += 1

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 2 == 0:
        with torch.no_grad():
            x_s, labels_s = next(iter(dataloader_test_source))
            # x_t, labels_t = next(iter(dataloader_test_target))
            x_s = x_s.to(DEVICE)
            # x_t = x_t.to(DEVICE)
            fake_t = model_G_s_t.forward(x_s)
            recovered_s = model_G_t_s.forward(fake_t)
            viz_sample = torch.cat((x_s, fake_t, recovered_s), 0)
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
