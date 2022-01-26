import h5py
import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, file_path, mode, img_size, transform):
        super().__init__()
        self.data = []
        self.mode = mode
        self.img_size = img_size
        self.file_path = file_path
        self.transform = transform

        file = h5py.File(self.file_path)

        labels = file[f"{self.mode}_labels"]
        imgs = file[f"{self.mode}_imgs"]
        self.data = list(zip(imgs, labels))

    def __getitem__(self, index):
        x, y = self.data[index]
        y = np.expand_dims(y, axis=0)

        x = np.reshape(x, (1, self.img_size, self.img_size))
        x = torch.FloatTensor(x)
        x = self.transform(x)
        mid = 255.0 / 2
        x = (x - mid) / mid

        y = torch.from_numpy(y)

        return x, y

    def __len__(self):
        return len(self.data)
