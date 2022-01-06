import argparse # pip3 install argparse
from copy import copy

from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import hashlib
import os
import pickle
import time
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import torchvision
import random
import torch.distributed
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data

# aequuiPhiar7779
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)
parser.add_argument('-num_epochs', default=2000, type=int)
parser.add_argument('-batch_size', default=8, type=int)
parser.add_argument('-chars_include', default='AET', type=str)
parser.add_argument('-samples_per_class', default=1000, type=int)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-z_size', default=128, type=int)
parser.add_argument('-style_weight', type=float, default=8.0)
parser.add_argument('-content_weight', type=float, default=12.0)
parser.add_argument('-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
Z_SIZE = args.z_size
DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
CHARS_INCLUDE = args.chars_include # '' = include all
IS_DEBUG = args.is_debug
INPUT_SIZE = 32

STYLE_W = args.style_weight
CONT_W = args.content_weight

if not torch.cuda.is_available() or IS_DEBUG:
    IS_DEBUG = True
    MAX_LEN = 400 # per class for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 8

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__() # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            split='byclass',
            train=(MAX_LEN == 0),
            download=True
        )
        self.labels = self.data.classes
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        if self.transform:
            pil_x = self.transform(pil_x)

        np_x = np.transpose(np.array(pil_x)).astype(np.float32)
        np_x = np.expand_dims(np_x, axis=0) / 255.0 # (1, W, H) => (1, 28, 28)

        return np_x, y_idx


class DatasetAriel(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        path_dataset = '../data/ariel_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1636095494-dml-course-2021-q4/ariel_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)
        self.labels = list(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.expand_dims(np.transpose(self.X[idx]), axis=0).astype(np.float32)
        if self.transform:
            x = np.transpose(x, (1, 2, 0))
            pil_x = self.transform(x)
            x = np.expand_dims(np.array(pil_x), axis=0)

        y = self.Y[idx]
        return x, y


def filter_dataset(dataset, name):
    str_args_for_hasing = [str(it) for it in [MAX_LEN, CHARS_INCLUDE] + dataset.labels]
    hash_args = hashlib.md5((''.join(str_args_for_hasing)).encode()).hexdigest()
    path_cache = f'../data/{hash_args}_dtngan_{name}.pkl'
    if os.path.exists(path_cache):
        print('loading from cache')
        with open(path_cache, 'rb') as fp:
            idxes_include = pickle.load(fp)
    else:
        idxes_include = []
        char_counter = {}
        for char in CHARS_INCLUDE:
            char_counter[char] = 0
        idx = 0
        for x, y in tqdm(dataset, f'filter_dataset {name}'):
            char = dataset.labels[int(y)]
            if char in CHARS_INCLUDE:
                char_counter[char] +=1
                if char_counter[char] < MAX_LEN:
                    idxes_include.append(idx)
            if all(it >= MAX_LEN for it in char_counter.values()):
                break
            idx += 1
        with open(path_cache, 'wb') as fp:
            pickle.dump(idxes_include, fp)

    dataset_filtered = torch.utils.data.Subset(dataset, idxes_include)
    return dataset_filtered


def mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content, style):
    assert (content.size()[:2] == style.size()[:2])
    size = content.size()
    style_mean, style_std = mean_std(style)
    content_mean, content_std = mean_std(content)

    normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),

            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),

            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),

            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.encoder.forward(x)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),

            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),

            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU(),

            torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1)),  # padding=(left, right, top, bottom)
            torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.LeakyReLU()
        )

    def forward(self, z):
        return self.decoder.forward(z)


class StyleTransferNet(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = torch.nn.Sequential(*enc_layers[:3])
        self.enc_2 = torch.nn.Sequential(*enc_layers[3:10])
        self.enc_3 = torch.nn.Sequential(*enc_layers[10:17])
        self.enc_4 = torch.nn.Sequential(*enc_layers[17:])
        self.mse_loss = torch.nn.MSELoss()
        self.decoder = decoder

        # Disable gradients for encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            enc = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(enc(results[-1]))
        return results[1:] # take only appended results not input

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = mean_std(input)
        target_mean, target_std = mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.content_loss(g_t_feats[-1], t.detach())
        loss_s = self.style_loss(g_t_feats[0], style_feats[0].detach())
        for i in range(1, 4):
            loss_s += self.style_loss(g_t_feats[i], style_feats[i].detach())
        return loss_c, loss_s, g_t


encoder = Encoder()
decoder = Decoder()
network = StyleTransferNet(encoder, decoder).to(DEVICE)

# def get_param_count(model):
#     params = list(model.parameters())
#     result = 0
#     for param in params:
#         count_param = np.prod(param.size()) # size[0] * size[1] ...
#         result += count_param
#     return result
#
# print(f'decoder params: {get_param_count(decoder)}')
# exit()

dataset_source = DatasetEMNIST(transform=transforms.Compose([
    transforms.Resize(size=(INPUT_SIZE, INPUT_SIZE))
]))
dataset_target = DatasetAriel(transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(INPUT_SIZE, INPUT_SIZE))
]))

dataset_source = filter_dataset(dataset_source, 'dataset_source')
dataset_target = filter_dataset(dataset_target, 'dataset_target')

dropped_samples_s = len(dataset_source) - BATCH_SIZE * int(len(dataset_source) // BATCH_SIZE)
subset_dataset_source = torch.utils.data.Subset(dataset_source, range(dropped_samples_s, len(dataset_source)))
dropped_samples_t = len(dataset_target) - BATCH_SIZE * int(len(dataset_target) // BATCH_SIZE)
subset_dataset_target = torch.utils.data.Subset(dataset_target, range(dropped_samples_t, len(dataset_target)))

print(f'dataset_source: {len(dataset_source)} dataset_target: {len(dataset_target)}')

data_loader_source = torch.utils.data.DataLoader(
    dataset=subset_dataset_source,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=(8 if not IS_DEBUG else 0)
)

data_loader_target = torch.utils.data.DataLoader(
    dataset=subset_dataset_target,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=(8 if not IS_DEBUG else 0)
)

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train']:
    for metric in ['loss', 'loss_c', 'loss_s']:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS):
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    iter_data_loader_target = iter(data_loader_target)
    for x_content, label_content in tqdm(data_loader_source, desc=stage):
        x_content = x_content.to(DEVICE)
        x_style, label_style = next(iter_data_loader_target)
        x_style = x_style.to(DEVICE)

        loss_c, loss_s, g_t = network(x_content, x_style)
        loss_c = CONT_W*loss_c
        loss_s = STYLE_W*loss_s
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
        metrics_epoch[f'{stage}_loss_c'].append(loss_c.cpu().item())
        metrics_epoch[f'{stage}_loss_s'].append(loss_s.cpu().item())

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()

    plt.subplot(222)  # row col idx
    grid_img = torchvision.utils.make_grid(
        x_content[:64].detach().cpu(),
        padding=10,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0))

    plt.subplot(224)  # row col idx
    grid_img = torchvision.utils.make_grid(
        g_t[:64].detach().cpu(),
        padding=10,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0))

    plt.subplot(221)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])

    plt.tight_layout(pad=0.5)

    if len(RUN_PATH) == 0:
        plt.show()
    else:
        if np.isnan(metrics[f'train_loss'][-1]):
            exit()
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')
