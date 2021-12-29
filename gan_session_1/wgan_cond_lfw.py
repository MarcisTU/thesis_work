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
from torchvision.utils import save_image
import torch.distributed
import torch.multiprocessing as mp
from sklearn.datasets import fetch_lfw_people

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='output', type=str)

parser.add_argument('-num_epochs', default=1000, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-min_faces_count', default=100, type=int)
parser.add_argument('-learning_rate', default=5e-5, type=float)
parser.add_argument('-z_size', default=256, type=int)
parser.add_argument('-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
Z_SIZE = args.z_size
DEVICE = 'cuda'
MIN_FACES_PER_PERSON = args.min_faces_count
IS_DEBUG = args.is_debug
INPUT_SIZE = 56
CONTINUE_TRAINING = False  # True if checkpoint is available

if not torch.cuda.is_available() or IS_DEBUG:
    MIN_FACES_PER_PERSON = 100
    DEVICE = 'cpu'
    BATCH_SIZE = 128

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

exist = os.path.exists(f'{RUN_PATH}_checkpoints')
if not exist:
    os.makedirs(f'{RUN_PATH}_checkpoints')


class DatasetLFW(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = []
        self.labels = []
        self.label_weights = []
        self.load_lfw()
        self.class_count()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        np_x, y_idx = self.data[idx]
        mid = 255.0 / 2.0
        np_x = (np_x - mid) / mid
        np_x = np.transpose(np_x, (2, 0, 1))

        return np_x, y_idx

    def class_count(self):
        counter = 0
        label_count = []
        for i in range(len(self.labels)):
            for sample in self.data:
                if sample[1] == i:
                    counter += 1
            label_count.append(counter)
            print(f"Class {i}:{self.labels[i]} => {counter}")
            counter = 0

        self.label_weights = 1 - (np.array(label_count) / sum(label_count))

    def load_lfw(self):
        dataset = fetch_lfw_people(
            data_home='../data',
            slice_= (slice(75, 187), slice(75, 187)),
            color=True,
            min_faces_per_person=MIN_FACES_PER_PERSON)
        images = dataset.images
        self.labels = dataset.target_names

        self.data = [(images[idx], target) for idx, target in enumerate(dataset.target)]

        # image_count_per_label = np.zeros(len(dataset.target_names))
        # for idx, target in enumerate(dataset.target):  # Balance the dataset to have equal amount of class images (under-sampling)
        #     if image_count_per_label[target] < MIN_FACES_PER_PERSON:
        #         self.data.append((images[idx], target))
        #         image_count_per_label[target] += 1


class ModelD(torch.nn.Module):
    def __init__(self, label_count):
        super().__init__()

        self.label_embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=label_count, embedding_dim=label_count),
            torch.nn.Linear(in_features=label_count, out_features=INPUT_SIZE ** 2)
        )

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[8, 56, 56]),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[16, 56, 56]),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=1),

            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[16, 28, 28]),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[32, 28, 28]),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=1),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[64, 14, 14]),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[64, 14, 14]),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=1),

            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[128, 7, 7]),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.LayerNorm(normalized_shape=[128, 7, 7]),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, x, labels):
        label_enc = self.label_embedding.forward(labels)
        label_enc = label_enc.view(labels.size(0), 1, INPUT_SIZE, INPUT_SIZE)

        x_labels_cat = torch.cat((x, label_enc), dim=1)
        x_enc = self.encoder.forward(x_labels_cat)

        x_enc_flat = x_enc.squeeze()
        y_prim = self.mlp.forward(x_enc_flat)

        return y_prim


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upsample=False, dropout=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        # Add regularization in form of dropout (25%)
        self.reg = False
        if dropout:
            self.reg = True
            self.dropout = torch.nn.Dropout(p=0.25)

        self.upsample = False
        if upsample:
            self.upsample = True

        # Check if input and output channels are changing
        self.bottle_neck = False
        if in_channels != out_channels:
            self.bottle_neck = True
            self.shortcut = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        residual = x

        out = x
        if self.upsample:
            out = F.upsample(out, scale_factor=2)
            residual = F.upsample(residual, scale_factor=2)
        out = self.conv1.forward(out)
        out = self.bn1.forward(out)
        out = F.leaky_relu_(out)

        out = self.conv2.forward(out)
        if self.bottle_neck:
            residual = self.shortcut.forward(residual)
        out += residual
        out = self.bn2.forward(out)
        out = F.leaky_relu_(out)
        if self.reg:
            out = self.dropout.forward(out)

        return out


class ModelG(torch.nn.Module):
    def __init__(self, label_count):
        super().__init__()

        self.decoder_size = INPUT_SIZE // 8
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=Z_SIZE, out_features=self.decoder_size ** 2 * 511),
        )
        self.label_embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=label_count, embedding_dim=label_count),
            torch.nn.Linear(in_features=label_count, out_features=self.decoder_size ** 2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=512),

            ResBlock(in_channels=512, out_channels=256, upsample=True),
            ResBlock(in_channels=256, out_channels=256, dropout=True),
            ResBlock(in_channels=256, out_channels=128, upsample=True),
            ResBlock(in_channels=128, out_channels=64, upsample=True),
            ResBlock(in_channels=64, out_channels=32, dropout=True),
            ResBlock(in_channels=32, out_channels=16),
            ResBlock(in_channels=16, out_channels=8),

            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.Tanh()
        )

    def forward(self, z, labels):
        label_enc = self.label_embedding.forward(labels)
        label_2d = label_enc.view(labels.size(0), 1, self.decoder_size, self.decoder_size)
        z_flat = self.mlp.forward(z)
        z_2d = z_flat.view(z.size(0), 511, self.decoder_size, self.decoder_size)
        z_label_enc = torch.cat((label_2d, z_2d), dim=1)
        y_prim = self.decoder.forward(z_label_enc)

        return y_prim


dataset_full = DatasetLFW()
data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_full,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_full) % BATCH_SIZE < 12),
    num_workers=(8 if not IS_DEBUG else 0)
)

model_D = ModelD(label_count=len(dataset_full.labels)).to(DEVICE)
model_G = ModelG(label_count=len(dataset_full.labels)).to(DEVICE)

# def get_param_count(model):
#     params = list(model.parameters())
#     result = 0
#     for param in params:
#         count_param = np.prod(param.size())  # size[0] * size[1] ...
#         result += count_param
#     return result
# print(f'model_D params: {get_param_count(model_D)}')
# print(f'model_G params: {get_param_count(model_G)}')
# exit()
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=LEARNING_RATE)
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train']:
    for metric in ['loss', 'loss_g', 'loss_d']:
        metrics[f'{stage}_{metric}'] = []

# Load checkpoint
if CONTINUE_TRAINING:
    checkpoint = torch.load('./lfw-model-750.pt')
    model_G.load_state_dict(checkpoint['model_G_state_dict'])
    model_G = model_G.train()
    model_D.load_state_dict(checkpoint['model_D_state_dict'])
    model_D = model_D.train()
    optimizer_G.load_state_dict(checkpoint['G_optimizer_state_dict'])
    optimizer_D.load_state_dict(checkpoint['D_optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    metrics = checkpoint['metrics']
else:
    epoch_start = 1

def sample_image(labels_interval, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    start_label = labels_interval[0]
    end_label = labels_interval[1]
    n_row = end_label - start_label
    # Sample noise
    z = torch.FloatTensor(np.random.normal(0, 1, ((n_row) ** 2, Z_SIZE))).to(DEVICE)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(start_label, end_label) for num in range(start_label, end_label)])
    labels = torch.LongTensor(labels).to(DEVICE)
    gen_imgs = model_G(z, labels)

    save_image(gen_imgs.data, f"{RUN_PATH}/{epoch}.png", nrow=n_row, normalize=True)


def gradient_penalty(critic, real_data, fake_data, labels, penalty, device):
    n_elements = real_data.nelement()
    batch_size = real_data.size()[0]
    colors = real_data.size()[1]
    image_width = real_data.size()[2]
    image_height = real_data.size()[3]
    alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
    alpha = alpha.view(batch_size, colors, image_width, image_height).to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates, labels)

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty
    return gradient_penalty


dist_z = torch.distributions.Normal(
    loc=0.0,
    scale=1.0
)

label_weights = dataset_full.label_weights
label_weights = torch.FloatTensor(label_weights).unsqueeze(dim=1).to(DEVICE)

for epoch in range(epoch_start, EPOCHS):
    metrics_epoch = {key: [] for key in metrics.keys()}
    stage = 'train'

    for x, labels in tqdm(data_loader_train, desc=stage):
        x = x.to(DEVICE)
        labels = labels.to(DEVICE)

        z = dist_z.sample((x.size(0), Z_SIZE)).to(DEVICE)
        x_gen_labels = torch.randint(0, len(dataset_full.labels), (x.size(0),), device=DEVICE)
        x_gen = model_G.forward(z, x_gen_labels)
        for param in model_D.parameters():
            param.requires_grad = False
        y_gen = model_D.forward(x_gen, x_gen_labels)

        y_gen_label_weights = label_weights[x_gen_labels]
        loss_G = -torch.mean(y_gen * y_gen_label_weights) # Weight the y_gen because of significant class imbalance
        loss_G.backward()
        optimizer_G.step()
        optimizer_G.zero_grad()

        for n in range(2):
            z = dist_z.sample((x.size(0), Z_SIZE)).to(DEVICE)
            x_fake = model_G.forward(z, labels)
            for param in model_D.parameters():
                param.requires_grad = True
            y_fake = model_D.forward(x_fake.detach(), labels)
            y_real = model_D.forward(x, labels)

            y_label_weights = label_weights[labels]

            # penalty = gradient_penalty(critic=model_D,
            #                            real_data=x,
            #                            fake_data=x_fake,
            #                            labels=labels,
            #                            penalty=10,
            #                            device=DEVICE)

            loss_D = torch.mean(y_fake * y_label_weights) - torch.mean(y_real * y_label_weights)
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

    # Save model checkpoint
    if epoch % 20 == 0:
        PATH = f'{RUN_PATH}_checkpoints/lfw-model-{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_G_state_dict': model_G.cpu().state_dict(),
            'model_D_state_dict': model_D.cpu().state_dict(),
            'G_optimizer_state_dict': optimizer_G.state_dict(),
            'D_optimizer_state_dict': optimizer_D.state_dict(),
            'metrics': metrics
        }, PATH)
        model_G = model_G.to(DEVICE)
        model_D = model_D.to(DEVICE)

        plt.clf()
        plt.subplot(121)  # row col idx
        plts = []
        c = 0
        for key, value in metrics.items():
            ax = plt.twinx()
            plts += ax.plot(value, f'C{c}', label=key)
            c += 1
        plt.legend(plts, [it.get_label() for it in plts])
        plt.tight_layout()
        plt.savefig(f"{RUN_PATH}/plot_epoch_{epoch}.jpg")

        # Save grid of len(dataset_full.labels)//2 generated images from random label interval
        # s = np.random.randint(0, len(dataset_full.labels) - (len(dataset_full.labels)//2))
        # sample_image(labels_interval=[s, s + (len(dataset_full.labels)//2)], epoch=epoch)
        sample_image(labels_interval=[0, 5], epoch=epoch)
