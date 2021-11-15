import copy
import json
import os
import pickle

import torch
import torch.nn.functional
import numpy as np
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from torch.hub import download_url_to_file
from tqdm import tqdm
plt.rcParams["figure.figsize"] = (15, 15)
plt.style.use('dark_background')
import torch.utils.data
import scipy.misc
import scipy.ndimage
import sklearn.decomposition
import argparse # pip install argparse
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=400, type=int)
parser.add_argument('-batch_size', default=16, type=int)

parser.add_argument('-learning_rate', default=1e-3, type=float)

parser.add_argument('-noise_prob', default=0.0, type=float)
parser.add_argument('-vae_beta', default=1e-3, type=float)
parser.add_argument('-z_size', default=64, type=int)
parser.add_argument('-is_skip', default=True, type=lambda x: (str(x).lower() == 'true'))


args, _ = parser.parse_known_args()

VAE_BETA = args.vae_beta
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
NOISE_PERCENTAGE = args.noise_prob
RUN_PATH = args.run_path
TRAIN_TEST_SPLIT = 0.8
Z_SIZE = args.z_size

if len(RUN_PATH):
    os.makedirs(RUN_PATH)

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if DEVICE == 'cuda':
    MAX_LEN = None

# empty to include all
INCLUDE_CLASSES = [
    'Pomegranate',
    'Orange',
    'PepperRed',
    'Tomato3',
    'GrapeWhite',
    'Banana',
    'AppleGrannySmith',
    'GrapePink',
    'Strawberry'
]

class DatasetFruits(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        dataset_name = 'fruits_32.pkl'
        path_dataset = f'../data/{dataset_name}'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                f'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/{dataset_name}',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X_tmp, Y_tmp, self.labels = pickle.load(fp)

        X = []
        Y = []
        if len(INCLUDE_CLASSES) == 0:
            X = X_tmp
            Y = Y_tmp
        else:
            for x, y in zip(X_tmp, Y_tmp):
                y_label = self.labels[y]
                if y_label in INCLUDE_CLASSES:
                    X.append(x)
                    Y.append(y)

        X = torch.from_numpy(np.array(X)).float()
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = Y.unsqueeze(dim=-1)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255.0
        y_label = self.Y[idx]

        y_target = x.clone()
        if NOISE_PERCENTAGE > 0:
            noise = torch.randn(x.size())
            x[noise < NOISE_PERCENTAGE] = 0

        return x, y_target, y_label


dataset_full = DatasetFruits()
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
    drop_last=(len(dataset_train) % BATCH_SIZE < 6)
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_test) % BATCH_SIZE < 6)
)


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, out_size):
        super().__init__()

        self.out_size = out_size
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.GroupNorm(num_groups, out_channels),
            torch.nn.Mish(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.GroupNorm(num_groups, out_channels),
            torch.nn.Mish()
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)

        if x.size(-1) != self.out_size:
            y_prim = torch.nn.functional.interpolate(y_prim, size=self.out_size)

            if args.is_skip:
                y_skip = torch.nn.functional.adaptive_avg_pool3d(
                    x, output_size=(y_prim.size(1), self.out_size, self.out_size))
                y_prim += y_skip

        return y_prim


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torchvision.models.resnet18( # 3, W, H
            pretrained=True
        )
        self.encoder_mu = torch.nn.Linear(
            in_features=self.encoder.fc.in_features,
            out_features=Z_SIZE
        )
        self.encoder_sigma = torch.nn.Linear(
            in_features=self.encoder.fc.in_features,
            out_features=Z_SIZE
        )
        self.encoder.fc = torch.nn.Identity()

        self.decoder = torch.nn.Sequential(
            DecoderBlock(in_channels=Z_SIZE, out_channels=64, num_groups=16, out_size=1),
            DecoderBlock(in_channels=64, out_channels=32, num_groups=16, out_size=2),
            DecoderBlock(in_channels=32, out_channels=16, num_groups=8, out_size=4),
            DecoderBlock(in_channels=16, out_channels=16, num_groups=8, out_size=8),
            DecoderBlock(in_channels=16, out_channels=8, num_groups=4, out_size=16),
            DecoderBlock(in_channels=8, out_channels=3, num_groups=1, out_size=32),

            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder.forward(x)
        out_flat = out.view(x.size(0), -1)

        z_sigma = torch.sigmoid(self.encoder_sigma.forward(out_flat))
        z_mu = torch.tanh(self.encoder_mu.forward(out_flat))

        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size()).to(DEVICE)
        # re-parameterization trick
        z = z_mu + z_sigma * eps

        z_2d = z.view(-1, Z_SIZE, 1, 1)
        y_prim = self.decoder.forward(z_2d)
        return y_prim, z, z_sigma, z_mu


model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(DEVICE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss_rec',
        'loss_kl',
        'loss',
        'z',
        'labels'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS+1):
    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [data_loader_train, data_loader_test]:
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'
            torch.set_grad_enabled(False)
            model = model.eval()
        else:
            torch.set_grad_enabled(True)
            model = model.train()

        for x, y, label_idx in tqdm(data_loader, desc=stage):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            model = model.train()
            y_prim, z, z_sigma, z_mu = model.forward(x)

            loss_rec = torch.mean(torch.abs(y_prim - y))
            loss_kl = torch.mean(VAE_BETA * torch.mean(
                -0.5 * (2.0 * torch.log(z_sigma + 1e-8) - z_sigma**2 - z_mu**2 + 1),
                dim=0
            ))
            loss = loss_rec + loss_kl

            metrics_epoch[f'{stage}_loss_rec'].append(loss_rec.item())
            metrics_epoch[f'{stage}_loss_kl'].append(loss_kl.item())
            metrics_epoch[f'{stage}_loss'].append(loss.item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_x = x.cpu().data.numpy()
            np_y = y.cpu().data.numpy()
            np_y_label = label_idx.squeeze().cpu().data.numpy()
            np_z = z.cpu().data.numpy()

            metrics_epoch[f'{stage}_z'] += np_z.tolist()
            metrics_epoch[f'{stage}_labels'] += np_y_label.tolist()

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
    decor = '-'
    for key, value in metrics.items():

        ax = plt.twinx()
        plts += ax.plot(value, f'C{c}{decor}', label=key)

        c += 1
        if c > 8:
            c = 0
            decor = '--'
    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18]):
        plt.subplot(8, 6, j) # row col idx
        plt.title(f"class: {dataset_full.labels[int(np_y_label[i])]}")
        plt.imshow(np.transpose(np_x[i], (1, 2, 0)))

        plt.subplot(8, 6, j+6) # row col idx
        plt.imshow(np.transpose(np_y_prim[i], (1, 2, 0)))

    plt.subplot(223) # row col idx

    pca = sklearn.decomposition.KernelPCA(n_components=2, gamma=0.1)

    plt.title('train_z')
    np_z = np.array(metrics_epoch[f'train_z'])
    np_z = pca.fit_transform(np_z)
    labels_raw = metrics_epoch[f'train_labels']
    labels_idx_set = list(set(labels_raw))
    labels_text_set = [dataset_full.labels[it] for it in labels_idx_set]
    np_z_label = np.array([labels_idx_set.index(it) for it in labels_raw])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=labels_text_set)

    plt.subplot(224) # row col idx

    plt.title('test_z')
    np_z = np.array(metrics_epoch[f'test_z'])
    np_z = pca.fit_transform(np_z)
    labels_raw = metrics_epoch[f'test_labels']
    labels_idx_set = list(set(labels_raw))
    labels_text_set = [dataset_full.labels[it] for it in labels_idx_set]
    np_z_label = np.array([labels_idx_set.index(it) for it in labels_raw])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=labels_text_set)

    plt.tight_layout(pad=0.5)

    if len(RUN_PATH) == 0:
        plt.show()
    else:
        if np.isnan(metrics[f'train_loss'][-1]) or np.isinf(metrics[f'test_loss'][-1]):
            exit()

        # save model weights
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')
        torch.save(model.state_dict(), f'{RUN_PATH}/model-{epoch}.pt')

