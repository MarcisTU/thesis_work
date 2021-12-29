import argparse
from copy import copy

from tqdm import tqdm
import hashlib
import os
import pickle
import time
import random
import subprocess as sp
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.utils import save_image
import torch.distributed
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data
import cv2


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=1000, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-classes_count', default=20, type=int)
parser.add_argument('-samples_per_class', default=400, type=int) # 400 is max per label

parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-z_size', default=128, type=int)

parser.add_argument('-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
Z_SIZE = args.z_size
DEVICE = 'cpu'
MAX_LEN = args.samples_per_class
MAX_CLASSES = args.classes_count
IS_DEBUG = args.is_debug
INPUT_SIZE = 28

# if not torch.cuda.is_available() or IS_DEBUG:
#     MAX_LEN = 300 # per class for debugging
#     MAX_CLASSES = 10 # reduce number of classes for debugging
#     DEVICE = 'cpu'
#     BATCH_SIZE = 64

# if len(RUN_PATH):
#     RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
#     if os.path.exists(RUN_PATH):
#         shutil.rmtree(RUN_PATH)
#     os.makedirs(RUN_PATH)


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
        mid = 255.0 / 2.0
        np_x = (np.expand_dims(np_x, axis=0) - mid) / mid  # Improvement 1. (Normalize images between -1 and 1). Use tanh in generator last layer
        return np_x, y_idx


class ModelG(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_size = INPUT_SIZE // 4
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=Z_SIZE, out_features=self.decoder_size ** 2 * 127),
        )
        self.label_embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=MAX_CLASSES, embedding_dim=MAX_CLASSES),
            torch.nn.Linear(in_features=MAX_CLASSES, out_features=self.decoder_size ** 2)
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
            torch.nn.Tanh()
        )

    def forward(self, z, labels):
        label_enc = self.label_embedding.forward(labels)
        label_2d = label_enc.view(labels.size(0), 1, self.decoder_size, self.decoder_size)
        z_flat = self.mlp.forward(z)
        z_2d = z_flat.view(z.size(0), 127, self.decoder_size, self.decoder_size)
        z_label_enc = torch.cat((label_2d, z_2d), dim=1)
        y_prim = self.decoder.forward(z_label_enc)

        return y_prim

    def create_latent_var(self, batch_size, label, seed=None):
        """Create latent variable z with label info"""
        if seed:
            torch.manual_seed(seed)
        z = torch.randn(batch_size, Z_SIZE).to(DEVICE)
        labels = torch.LongTensor([label])
        label_enc = self.label_embedding.forward(labels)
        label_2d = label_enc.view(labels.size(0), 1, self.decoder_size, self.decoder_size)

        z_flat = self.mlp.forward(z)
        z_2d = z_flat.view(z.size(0), 127, self.decoder_size, self.decoder_size)

        z_label_enc = torch.cat((label_2d, z_2d), dim=1)
        return z_label_enc


dataset_full = DatasetEMNIST()

model_G = ModelG()
checkpoint = torch.load('./wgan_emnist/emnist-model-540.pt', map_location='cpu')
model_G.load_state_dict(checkpoint['model_G_state_dict'])
model_G.train()
torch.set_grad_enabled(False)


def image_grid(class_idx, n_row=5):
    # Sample noise
    z = torch.FloatTensor(np.random.normal(0, 1, ((n_row) ** 2, Z_SIZE)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([class_idx for _ in range(n_row) for num in range(n_row)])
    labels = torch.LongTensor(labels)
    gen_imgs = model_G(z, labels)
    exist = os.path.exists("./images")
    if not exist:
        os.makedirs("./images")
    save_image(gen_imgs.data, f"images/images_generated_{dataset_full.labels[class_idx]}.png", nrow=n_row, normalize=True)

def latent_lerp(gan, z0, z1, n_frames): # from z0 -> z1
    """Interpolate between two images in latent space"""
    imgs = []
    for i in range(n_frames):
        alpha = i / n_frames
        z = (1 - alpha) * z0 + alpha * z1
        imgs.append(gan.decoder.forward(z).squeeze(dim=0))
    return imgs

def latent_slerp(gan, z0, z1, n_frames): # from z0 -> z1
    """Interpolate between two images in spherical latent space"""
    # Compute angle between vectors
    unit_vector_1 = z0 / np.linalg.norm(z0, axis=1)
    unit_vector_2 = z1 / np.linalg.norm(z1, axis=1)
    product = np.matmul(unit_vector_1, unit_vector_2)
    angle = np.arccos(product)

    imgs = []
    for i in range(n_frames):
        alpha = i / n_frames
        z = (torch.sin((1 - alpha) * angle) / torch.sin(angle)) * z0 + \
            (torch.sin(alpha * angle) / torch.sin(angle)) * z1
        imgs.append(gan.decoder.forward(z).squeeze(dim=0))
    return imgs

n_frames = 100
label_1 = 7
label_2 = 2

z0 = model_G.create_latent_var(batch_size=1, label=label_1, seed=55)
z1 = model_G.create_latent_var(batch_size=1, label=label_2, seed=55)

imgs_l = latent_lerp(gan=model_G, z0=z0, z1=z1, n_frames=n_frames)
imgs_s = latent_slerp(gan=model_G, z0=z0, z1=z1, n_frames=n_frames)

save_image(imgs_l, f"images/linear_interpolation_{dataset_full.labels[label_1]}_{dataset_full.labels[label_2]}.png", nrow=10, normalize=True)
save_image(imgs_s, f"images/spherical_interpolation_{dataset_full.labels[label_1]}_{dataset_full.labels[label_2]}.png", nrow=10, normalize=True)

# exist = os.path.exists("./video")
# if not exist:
#     os.makedirs("./video")
#
# height = 96
# width = 96
# # initialize video writer
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fps = 30
#
# for imgs in [imgs_l, imgs_s]:
#     video_filename = f'./video/{interp}_{dataset_full.labels[label_1]}_to_{dataset_full.labels[label_2]}.avi'
#     out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
#
#     for i, img in enumerate(imgs):
#         img = img.squeeze(dim=0).detach().numpy()
#         img = np.transpose(img, (1, 2, 0))
#         img = cv2.normalize(img, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#         img_3c = cv2.merge([img, img, img])
#         resized = cv2.resize(img_3c, (height, width), interpolation=cv2.INTER_AREA)
#         out.write(resized)
#
# # close out the video writer
# out.release()

# generated image grid
image_grid(label_1)
image_grid(label_2)
