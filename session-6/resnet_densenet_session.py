import os
import pickle

import torch.nn.functional as F
import torch
from torchvision.transforms import transforms
import numpy as np
from torch.hub import download_url_to_file
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
plt.style.use('dark_background')


import torch.utils.data
import scipy.ndimage


USE_CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 64
MAX_LEN = 2240 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if USE_CUDA:
    MAX_LEN = None


class DatasetApples(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        path_dataset = '../data/apples_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/apples_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        self.X = np.array(X)
        Y = torch.LongTensor(Y)
        self.Y = Y

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        else:
            x = torch.FloatTensor(x)
            x /= 255

        return x, y


transformer = transforms.Compose([
    transforms.ToTensor(),  # Converts numpy.ndarray to torch.Tensor Values from 0:255 to 0:1
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomRotation(degrees=90),
    transforms.CenterCrop(size=100),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4)
])

dataset_full = DatasetApples(transform=transformer)
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.is_bottleneck = False
        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=(stride, stride), bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1.forward(x)
        out = F.relu(out)
        out = self.bn1.forward(out)
        out = self.conv2.forward(out)

        if self.is_bottleneck:
            residual = self.shortcut.forward(x)

        out += residual

        out = F.relu(out)
        out = self.bn2.forward(out)
        return out


class DenseBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.bn3 = torch.nn.BatchNorm2d(num_features=96)
        self.bn4 = torch.nn.BatchNorm2d(num_features=128)

        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1))

    def forward(self, x):
        conv1 = self.conv1.forward((self.bn1.forward(F.relu(x))))
        conv2_in = torch.cat([x, conv1], dim=1)

        conv2 = self.conv2.forward((self.bn2.forward(F.relu(conv2_in))))
        conv3_in = torch.cat([x, conv1, conv2], dim=1)

        conv3 = self.conv3.forward((self.bn3.forward(F.relu(conv3_in))))
        conv4_in = torch.cat([x, conv1, conv2, conv3], dim=1)

        conv4 = self.conv4.forward((self.bn4.forward(F.relu(conv4_in))))
        out = torch.cat([x, conv1, conv2, conv3, conv4], dim=1)

        return out


class TransitionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.bn.forward(F.relu(self.conv.forward(x)))
        out = self.avg_pool.forward(out)
        return out


class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Make DenseBlock's
        self.dense_block_1 = DenseBlock()
        self.dense_block_2 = DenseBlock()
        self.dense_block_3 = DenseBlock()

        # Make Transition Layer's
        self.transition_layer_1 = TransitionLayer(in_channels=160, out_channels=32)
        self.transition_layer_2 = TransitionLayer(in_channels=160, out_channels=32)
        self.transition_layer_3 = TransitionLayer(in_channels=160, out_channels=32)

        # Classifier 288 = 32x3x3
        self.linear = torch.nn.Linear(288, 5)

    def forward(self, x):
        out = self.conv.forward(x)
        out = self.max_pool.forward(out)

        out = self.dense_block_1.forward(out)
        out = self.transition_layer_1.forward(out)

        out = self.dense_block_2.forward(out)
        out = self.transition_layer_2.forward(out)

        out = self.dense_block_3.forward(out)
        out = self.transition_layer_3.forward(out)

        out = out.view(-1, 288)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)

        return out

class ModelResnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=4,
                                     kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=4)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Make ResBlock's
        self.identity_block_1 = ResBlock(in_channels=4, out_channels=4)
        self.identity_block_2 = ResBlock(in_channels=4, out_channels=4)

        self.bottleneck_block_1 = ResBlock(in_channels=4, out_channels=8, stride=2)
        self.identity_block_3 = ResBlock(in_channels=8, out_channels=8)

        self.bottleneck_block_2 = ResBlock(in_channels=8, out_channels=16, stride=2)
        self.identity_block_4 = ResBlock(in_channels=16, out_channels=16)

        self.bottleneck_block_3 = ResBlock(in_channels=16, out_channels=32, stride=2)
        self.identity_block_5 = ResBlock(in_channels=32, out_channels=32)

        self.linear = torch.nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        out = self.bn1.forward(F.relu(self.conv1.forward(x)))
        out = self.max_pool.forward(out)

        out = self.identity_block_1.forward(out)
        out = self.identity_block_2.forward(out)

        out = self.bottleneck_block_1.forward(out)
        out = self.identity_block_3.forward(out)

        out = self.bottleneck_block_2.forward(out)
        out = self.identity_block_4.forward(out)

        out = self.bottleneck_block_3.forward(out)
        out = self.identity_block_5.forward(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)

        return out

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=5)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = F.adaptive_avg_pool2d(out, output_size=(1,1))
        out = out.view(x.size(0), -1)
        out = F.softmax(out, dim=1)
        return out

model = DenseNet()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

if USE_CUDA:
    model = model.to('cuda:0')
    # model = model.cuda()
    # loss_func = loss_func.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):
    sample_x = []
    sample_y_idx = []
    sample_y_prim_idx = []

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y_idx in tqdm(data_loader):
            if USE_CUDA:
                x = x.cuda()
                y = y_idx.cuda()

            y_prim = model.forward(x)
            idx = range(len(y_idx))
            loss = torch.mean(-torch.log(y_prim + 1e-20)[idx, y_idx])

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.cpu()
            x = x.cpu()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y_idx = y_idx.cpu().data.numpy()
            idx_y_prim = np.argmax(np_y_prim, axis=1)
            acc = np.average((np_y_idx == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f
            metrics_epoch[f'{stage}_acc'].append(acc)
            if len(x) == BATCH_SIZE:
                sample_x = x
                sample_y_idx = np_y_idx
                sample_y_prim_idx = idx_y_prim

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.subplot(121) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 10, 11, 12, 16, 17, 18]):
        plt.subplot(3, 6, j)
        color = 'green' if sample_y_prim_idx[i]==sample_y_idx[i] else 'red'
        plt.title(f"pred: {sample_y_prim_idx[i]}\n real: {sample_y_idx[i]}", c=color)
        plt.imshow(sample_x[i].permute(1, 2, 0))

    plt.tight_layout(pad=0.5)
    plt.draw()
    plt.pause(0.1)

input('quit?')