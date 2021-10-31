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
MAX_LEN = 2240
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


class InceptionBlockA(torch.nn.Module):
    def __init__(self, in_channels, out_1x1, red_5x5, out_5x5, red_3x3,
                       out_3x3_1, out_3x3, out_1x1pool, stride=1):
        super().__init__()
        self.out_total_channels = sum([out_1x1, out_5x5, out_3x3, out_1x1pool])
        self.batch_norm = torch.nn.BatchNorm2d(num_features=self.out_total_channels)

        self.branch1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_1x1, kernel_size=(1, 1))
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=red_5x5, kernel_size=(1, 1)),
            torch.nn.Conv2d(in_channels=red_5x5, out_channels=out_5x5, kernel_size=(5, 5), padding=2)
        )
        self.branch3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=red_3x3, kernel_size=(1, 1)),
            torch.nn.Conv2d(in_channels=red_3x3, out_channels=out_3x3_1, kernel_size=(3, 3), padding=1),
            torch.nn.Conv2d(in_channels=out_3x3_1, out_channels=out_3x3, kernel_size=(3, 3), padding=1)
        )
        self.branch4 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_1x1pool, kernel_size=(1, 1))
        )

        self.is_bottleneck = False
        if stride != 1:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                            kernel_size=(1, 1), stride=(stride, stride), bias=False)

    def forward(self, x):
        if self.is_bottleneck:
            x = self.shortcut.forward(x)

        out = torch.cat([
            self.branch1.forward(x),
            self.branch2.forward(x),
            self.branch3.forward(x),
            self.branch4.forward(x)],
            dim=1)
        out = self.batch_norm.forward(F.relu(out))

        return out


class InceptionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(in_channels=88, out_channels=64, kernel_size=(5, 5))
        """
            In this order: in_channels, out_1x1, red_5x5, out_5x5, red_3x3, out_3x3_1, out_3x3, out_1x1pool, stride?
        """
        self.inception_block1 = InceptionBlockA(16, 16, 8, 24, 8, 16, 32, 16, stride=2)
        self.inception_block2 = InceptionBlockA(64, 32, 32, 48, 32, 32, 64, 64, stride=2)

        self.linear = torch.nn.Linear(in_features=100672, out_features=5)  # in_features = out_channels*img_size*img_size

    def forward(self, x):
        out = self.conv1.forward(x)
        # out = self.max_pool.forward(out)
        out = F.relu(out)
        out = self.inception_block1.forward(out)

        out = self.conv2.forward(out)
        # out = self.max_pool.forward(out)
        out = F.relu(out)
        out = self.inception_block2.forward(out)

        # (64, channels*conv_image_size*conv_image_size) -> (64, channels*22*22)
        out = out.view(out.size(0), -1)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)

        return out


model = InceptionNet()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-6)

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
                y_idx = y_idx.cuda()

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