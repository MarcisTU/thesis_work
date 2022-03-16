import h5py
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, file_path, mode, img_size, transform):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        self.file_path = file_path
        self.transform = transform

        file = h5py.File(self.file_path)
        self.labels = file[f"{self.mode}_labels"]
        self.imgs = file[f"{self.mode}_imgs"]

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]
        y = np.expand_dims(y, axis=0)

        x = np.reshape(x, (1, self.img_size, self.img_size))
        x = torch.FloatTensor(x)
        x = self.transform(x)
        # x = x / 255.0
        mid = 255.0 / 2
        x = (x - mid) / mid  # -1,1

        y = torch.from_numpy(y)

        return x, y

    def __len__(self):
        return len(self.labels)
