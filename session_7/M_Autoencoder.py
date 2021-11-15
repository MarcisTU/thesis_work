import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision
from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import random

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import scipy.misc
import scipy.ndimage
import sklearn.decomposition
from skimage import io, transform

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TRAIN_TEST_SPLIT = 0.8
DEVICE = 'cuda'
MAX_LEN = 0

if not torch.cuda.is_available():
    MAX_LEN = 0 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
    DEVICE = 'cpu'
    BATCH_SIZE = 32


class DatasetApples(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
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

        X = torch.from_numpy(np.array(X))
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = Y.unsqueeze(dim=-1)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y_label = self.Y[idx]

        noisy_sample = torch.randint(0, 2, (1,))
        if noisy_sample:
            idxs_w = torch.randint(0, 100, (100,))
            idxs_h = torch.randint(0, 100, (100,))
            x[:, idxs_h, idxs_w] = 0

        y_target = x
        return x, y_target, y_label


dataset_full = DatasetApples()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1)
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_test) % BATCH_SIZE == 1)
)


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, size=None, lastblock=False):
        super(EncoderBlock, self).__init__()
        self.size = size
        self.activation = torch.nn.Mish() if not lastblock else torch.nn.Tanh()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1, 1)),
            torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        )

    def forward(self, x):
        out = self.block.forward(x)
        out = self.activation.forward(out)
        if self.size:
            out = torch.nn.Upsample(size=self.size).forward(out)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, size=None, lastblock=False):
        super(DecoderBlock, self).__init__()
        self.size = size
        self.activation = torch.nn.Mish() if not lastblock else torch.nn.Sigmoid()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                                 stride=(1, 1), padding=(1, 1)),
            torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        )

    def forward(self, x):
        out = self.block.forward(x)
        out = self.activation.forward(out)
        if self.size:
            out = torch.nn.Upsample(size=self.size).forward(out)
        return out


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            EncoderBlock(in_channels=3, out_channels=8, num_groups=2, size=64),
            EncoderBlock(in_channels=8, out_channels=16, num_groups=4, size=32),
            EncoderBlock(in_channels=16, out_channels=32, num_groups=8, size=16),
            EncoderBlock(in_channels=32, out_channels=64, num_groups=16, size=4),
            EncoderBlock(in_channels=64, out_channels=128, num_groups=32, size=2),
            EncoderBlock(in_channels=128, out_channels=256, num_groups=64, size=1, lastblock=True)
        )

        self.decoder = torch.nn.Sequential(
            DecoderBlock(in_channels=256, out_channels=256, num_groups=64, size=2),
            DecoderBlock(in_channels=256, out_channels=128, num_groups=64),
            DecoderBlock(in_channels=128, out_channels=128, num_groups=16),
            DecoderBlock(in_channels=128, out_channels=96, num_groups=32, size=4),
            DecoderBlock(in_channels=96, out_channels=96, num_groups=32),
            DecoderBlock(in_channels=96, out_channels=96, num_groups=16),
            DecoderBlock(in_channels=96, out_channels=64, num_groups=16, size=16),
            DecoderBlock(in_channels=64, out_channels=64, num_groups=16),
            DecoderBlock(in_channels=64, out_channels=64, num_groups=8),
            DecoderBlock(in_channels=64, out_channels=32, num_groups=8, size=32),
            DecoderBlock(in_channels=32, out_channels=32, num_groups=8),
            DecoderBlock(in_channels=32, out_channels=16, num_groups=4),
            DecoderBlock(in_channels=16, out_channels=16, num_groups=4, size=64),
            DecoderBlock(in_channels=16, out_channels=8, num_groups=4),
            DecoderBlock(in_channels=8, out_channels=8, num_groups=2),
            DecoderBlock(in_channels=8, out_channels=3, num_groups=1, size=100),
            DecoderBlock(in_channels=3, out_channels=3, num_groups=1, lastblock=True)
        )

    def forward(self, x):
        z = self.encoder.forward(x)
        z = z.view(-1, 256)
        y_prim = self.decoder.forward(z.view(-1, 256, 1, 1))
        return y_prim, z


model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model = model.to(DEVICE)

# Add anomaly label
dataset_full.labels += ['Anomaly']
img_paths = ['anom1_wrench.jpg', 'anom2_screwdriver.jpg', 'anom3_saw.jpg', 'anom4_shovel.jpg']
anomaly_images = torch.zeros((len(img_paths), 3, 100, 100))
for i, path in enumerate(img_paths):
    img = io.imread(path)
    img = transform.resize(img, (100,100))
    anomaly_images[i] = torch.FloatTensor(img).permute(2, 0, 1)

anomaly_label = torch.ones((len(img_paths),)) * 5

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'z',
        'labels'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 150):
    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [data_loader_train, data_loader_test]:
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y_target, y_label in tqdm(data_loader, desc=stage):
            x = x.to(DEVICE)
            y_target = y_target.to(DEVICE)
            y_label = y_label.squeeze().to(DEVICE)

            y_prim, z = model.forward(x)
            loss = torch.mean(torch.pow((x - y_prim), 2))
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_z = z.cpu().data.numpy()
            np_x = x.cpu().data.numpy()
            np_y_target = y_target.cpu().data.numpy()
            np_y_label = y_label.cpu().data.numpy()

            metrics_epoch[f'{stage}_z'] += np_z.tolist()
            metrics_epoch[f'{stage}_labels'] += np_y_label.tolist()

        # Add anomaly
        if stage == 'test':
            anomaly_images = anomaly_images.to(DEVICE)
            anomaly_label = anomaly_label.to(DEVICE)
            anomaly_y_prim, anomaly_z = model.forward(anomaly_images)
            metrics_epoch[f'{stage}_z'] += anomaly_z.cpu().data.numpy().tolist()
            metrics_epoch[f'{stage}_labels'] += anomaly_label.cpu().data.numpy().tolist()

    metrics_strs = []
    for key in metrics_epoch.keys():
        if '_z' not in key and '_labels' not in key:
            value = np.mean(metrics_epoch[key])
            metrics[key].append(value)
            metrics_strs.append(f'{key}: {round(value, 2)}')
    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.subplot(221) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18]):
        plt.subplot(8, 6, j) # row col idx
        plt.title(f"class: {dataset_full.labels[np_y_label[i]]}")
        plt.imshow(np.transpose(np_x[i], (1, 2, 0)))

        plt.subplot(8, 6, j+6) # row col idx
        plt.imshow(np.transpose(np_y_prim[i], (1, 2, 0)))

    plt.subplot(223) # row col idx

    pca = sklearn.decomposition.KernelPCA(n_components=2, gamma=0.1)

    plt.title('train_z')
    np_z = np.array(metrics_epoch[f'train_z'])
    np_z = pca.fit_transform(np_z)
    np_z_label = np.array(metrics_epoch[f'train_labels'])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=dataset_full.labels[:-1])

    plt.subplot(224) # row col idx

    plt.title('test_z')
    np_z = np.array(metrics_epoch[f'test_z'])
    np_z = pca.fit_transform(np_z)
    np_z_label = np.array(metrics_epoch[f'test_labels'])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=dataset_full.labels)

    plt.tight_layout(pad=0.5)
    plt.show()
