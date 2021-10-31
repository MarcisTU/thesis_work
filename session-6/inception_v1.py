import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
plt.style.use('dark_background')

import torch.utils.data
import scipy.ndimage

USE_CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 64
IMAGE_SIZE = 224
MAX_LEN = 2240
if USE_CUDA:
    MAX_LEN = None


transformer = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),  # Converts numpy.ndarray to torch.Tensor Values from 0:255 to 0:1
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    # transforms.RandomRotation(degrees=90),
    transforms.CenterCrop(size=IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4)
])

dataset_full = datasets.ImageFolder("../data/archive/Fish_Dataset/Fish_Dataset", transform=transformer)

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu.forward(self.batchnorm.forward(self.conv.forward(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super().__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1.forward(x),
            self.branch2.forward(x),
            self.branch3.forward(x),
            self.branch4.forward(x)
        ], dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool.forward(x)
        x = self.conv.forward(x)
        x = self.relu.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu.forward(self.fc1.forward(x))
        x = self.dropout.forward(x)
        x = self.fc2.forward(x)
        x = F.softmax(x, dim=1)

        return x


class GoogleNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3,
                                out_channels=64,
                                kernel_size=(7,7),
                                stride=(2,2),
                                padding=(3,3))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, padding=1)

        """
            In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        """
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)  # 1024 = out_1x1+out_3x3+out_5x5+out_1x1pool


        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)


    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.maxpool.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        x = self.maxpool.forward(x)

        x = self.inception3a.forward(x)
        x = self.inception3b.forward(x)
        x = self.maxpool.forward(x)

        x = self.inception4a.forward(x)

        aux1 = self.aux1.forward(x)

        x = self.inception4b.forward(x)
        x = self.inception4c.forward(x)
        x = self.inception4d.forward(x)

        aux2 = self.aux2.forward(x)

        x = self.inception4e.forward(x)
        x = self.maxpool.forward(x)

        x = self.inception5a.forward(x)
        x = self.inception5b.forward(x)
        x = self.avgpool.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout.forward(x)
        x = self.fc1.forward(x)
        x = F.softmax(x, dim=1)

        return aux1, aux2, x


model = GoogleNet()
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

            aux1, aux2, y_prim = model.forward(x)

            idx = range(len(y_idx))
            if data_loader == data_loader_train:
                # The total loss used by the inception net during training.
                # total_loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2
                aux1_loss = torch.mean(-torch.log(aux1 + 1e-20)[idx, y_idx])
                aux2_loss = torch.mean(-torch.log(aux2 + 1e-20)[idx, y_idx])
                real_loss = torch.mean(-torch.log(y_prim + 1e-20)[idx, y_idx])
                loss = real_loss + 0.3*aux1_loss + 0.3*aux2_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss = torch.mean(-torch.log(y_prim + 1e-20)[idx, y_idx])

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

    plt.tight_layout(pad=1)
    plt.draw()
    plt.pause(0.1)

input('quit?')