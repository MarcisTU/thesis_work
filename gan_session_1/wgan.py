import argparse
from copy import copy

from tqdm import tqdm
import hashlib
import os
import pickle
import time
import random
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.distributed
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=500, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-classes_count', default=20, type=int)
parser.add_argument('-samples_per_class', default=10000, type=int)

parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-z_size', default=128, type=int)

parser.add_argument('-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
Z_SIZE = args.z_size
DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
MAX_CLASSES = args.classes_count  # 0 = include all
IS_DEBUG = args.is_debug
INPUT_SIZE = 28

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 300 # per class for debugging
    MAX_CLASSES = 6 # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 66

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)


class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()  # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            split='balanced',
            train=(MAX_LEN == 0),
            download=True
        )
        class_to_idx = self.data.class_to_idx
        idx_to_class = dict((value, key) for key, value in class_to_idx.items())
        self.labels = [idx_to_class[idx] for idx in range(len(idx_to_class))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.transpose(np.array(pil_x).astype(np.float32))
        np_x = np.expand_dims(np_x, axis=0) / 255.0  # (1, W, H) => (1, 28, 28)
        return np_x, y_idx


class ModelD(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # B, 8, 14, 14

            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # B, 32, 14, 14

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # B, 64, 14, 14
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))  # B, 64, 1, 1
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x):
        x_enc = self.encoder.forward(x)
        x_enc_flat = x_enc.squeeze()  # B, 64, 1, 1 => B, 64
        y_prim = self.mlp.forward(x_enc_flat)
        return y_prim


class ModelG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_size = INPUT_SIZE//4  # upsampel twice * 2dim (W, H)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=Z_SIZE, out_features=self.decoder_size**2 * 128)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),

            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),

            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, z):
        z_flat = self.mlp.forward(z)
        z_2d = z_flat.view(z.size(0), 128, self.decoder_size, self.decoder_size)
        y_prim = self.decoder.forward(z_2d)
        return y_prim


dataset_full = DatasetEMNIST()

np.random.seed(2)
labels_train = copy(dataset_full.labels)
random.shuffle(labels_train, random=np.random.random)
labels_train = labels_train[:MAX_CLASSES]
np.random.seed(int(time.time()))

idx_train = []
str_args_for_hashing = [str(it) for it in [MAX_LEN, MAX_CLASSES] + labels_train]
hash_args = hashlib.md5((''.join(str_args_for_hashing)).encode()).hexdigest()
path_cache = f'../data/{hash_args}_gan.pkl'
if os.path.exists(path_cache):
    print('loading from cache')
    with open(path_cache, 'rb') as fp:
        idx_train = pickle.load(fp)

else:
    labels_count = dict((key, 0) for key in dataset_full.labels)
    for idx, (x, y_idx) in tqdm(
        enumerate(dataset_full),
        'splitting dataset',
        total=len(dataset_full)
    ):
        label = dataset_full.labels[y_idx]
        if MAX_LEN > 0:
            if labels_count[label] >= MAX_LEN:
                if all(it >= MAX_LEN for it in labels_count.values()):
                    break
                continue
        labels_count[label] += 1
        if label in labels_train:
            idx_train.append(idx)

    with open(path_cache, 'wb') as fp:
        pickle.dump(idx_train, fp)

dataset_train = torch.utils.data.Subset(dataset_full, idx_train)
data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE < 12),
    num_workers=(8 if not IS_DEBUG else 0)
)

model_D = ModelD().to(DEVICE)
model_G = ModelG().to(DEVICE)

def get_param_count(model):
    params = list(model.parameters())
    result = 0
    for param in params:
        count_param = np.prod(param.size())  # size[0] * size[1] ...
        result += count_param
    return result
print(f'model_D params: {get_param_count(model_D)}')
print(f'model_G params: {get_param_count(model_G)}')
exit()
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=LEARNING_RATE)
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train']:
    for metric in ['loss', 'loss_g', 'loss_d']:
        metrics[f'{stage}_{metric}'] = []

dist_z = torch.distributions.Normal(
    loc=0.0,
    scale=1.0
)

for epoch in range(1, 500):
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    for x, x_idx in tqdm(data_loader_train, desc=stage):
        x = x.to(DEVICE)

        z = dist_z.sample((x.size(0), Z_SIZE)).to(DEVICE)
        x_gen = model_G.forward(z) # fake image
        for param in model_D.parameters():
            param.requires_grad = False
        y_gen = model_D.forward(x_gen)

        loss_G = -torch.mean(y_gen)
        loss_G.backward()
        optimizer_G.step()
        optimizer_G.zero_grad()

        for n in range(5):
            z = dist_z.sample((x.size(0), Z_SIZE)).to(DEVICE)
            x_fake = model_G.forward(z)
            for param in model_D.parameters():
                param.requires_grad = True
            y_fake = model_D.forward(x_fake.detach()) # x_fake.detach() makes optimizer not learn G
            y_real = model_D.forward(x)

            loss_D = torch.mean(y_fake) - torch.mean(y_real)
            loss_D.backward()

            torch.nn.utils.clip_grad_norm(model_D.parameters(), max_norm=1e-2, norm_type=1)
            optimizer_D.step()
            optimizer_D.zero_grad()

        loss = loss_D + loss_G
        metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
        metrics_epoch[f'{stage}_loss_g'].append(loss_G.cpu().item())
        metrics_epoch[f'{stage}_loss_d'].append(loss_D.cpu().item())

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.subplot(111)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        ax = plt.twinx()
        plts += ax.plot(value, f'C{c}', label=key)
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])
    plt.tight_layout(pad=0.5)
    plt.show()

    plt.clf()
    for i in range(16):
        plt.subplot(4, 4, i + 1)  # row col idx
        plt.imshow(x_fake.detach().squeeze(dim=1).numpy()[i])
    plt.tight_layout(pad=0.5)
    plt.show()

    plt.pause(5)
