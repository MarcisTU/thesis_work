import sklearn.datasets
import numpy as np
import time
import torch
import torch.utils.data
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-4
BATCH_SIZE = 7


class DatasetDiabetes(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = sklearn.datasets.load_diabetes(return_X_y=True)
        self.is_train = is_train
        self.standardize()

    def __len__(self):
        X, Y = self.data
        return len(X)

    def __getitem__(self, idx):
        x = self.data[0][idx]
        y = self.data[1][idx]

        return x, y

    def standardize(self):
        np_x, np_y = self.data
        np_y = np.expand_dims(np_y, axis=1)

        x = torch.FloatTensor(np_x)
        y = torch.FloatTensor(np_y)

        # Separate categorical
        X_cat = torch.unsqueeze(x[:, 1], dim=1)
        # encode sex feature as a number 1 or 0 instead of float value
        X_cat[:, 0] = X_cat[:, 0] >= 0

        # Get all of the continuous data
        X_contin = torch.cat((x[:, :1], x[:, 2:]), 1)
        # pytorch standardization
        x_means = torch.mean(X_contin, dim=0, keepdim=True)
        x_stds = torch.std(X_contin, dim=0, keepdim=True)
        X_norm = (X_contin - x_means) / x_stds

        y_mean = torch.mean(y, dim=0, keepdim=True)
        y_std = torch.std(y, dim=0, keepdim=True)
        Y_norm = (y - y_mean) / y_std

        # Add everything back together after normalising continuous variables
        X_norm = torch.cat((X_norm[:, :1], X_cat, X_norm[:, 1:]), 1)

        torch.manual_seed(0)
        # split training/test data
        idxes_rand = torch.randperm(len(X_norm))
        X = X_norm[idxes_rand]
        Y = Y_norm[idxes_rand]

        idx_split = int(len(X) * 0.8)  # 80% for training and 20% for testing
        dataset_train = (X[:idx_split], Y[:idx_split])
        dataset_test = (X[idx_split:], Y[idx_split:])

        # torch.manual_seed(int(time.time()))

        if self.is_train:
            self.data = dataset_train
        else:
            self.data = dataset_test


dataloader_train = torch.utils.data.DataLoader(
    dataset=DatasetDiabetes(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=False
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=DatasetDiabetes(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Separate categorical
        X_cat = torch.unsqueeze(x[:, 1], dim=1)
        X_cat = X_cat.type(torch.IntTensor)

        cat_emb = torch.squeeze(self.embedding(X_cat))  # (BATCH_SIZE, 3)
        X_emb = torch.cat((x[:, :1], cat_emb, x[:, 2:]), 1)

        return X_emb  # (BATCH_SIZE, 12)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Embedding(num_embeddings=2, embedding_dim=3),
            torch.nn.Linear(in_features=12, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=5),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


model = Model()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
      'loss',
      'r2_score'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 2001):
    for data_loader in [dataloader_train, dataloader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataloader_test:
            stage = 'test'

        for x, y in data_loader:
            y_prim = model.forward(x)
            loss = torch.mean(torch.abs(y - y_prim)) # l1
            # loss = torch.mean(torch.pow((y - y_prim), 2)) # l2

            y = y.detach()
            y_prim = y_prim.detach()
            # r-2 metric
            r_square = 1 - (torch.sum(torch.pow((y - y_prim), 2)) / torch.sum(torch.pow((y - torch.mean(y)), 2)))

            if stage == 'train':  # Optimize only in training dataset
                loss.backward()
                optimizer.step()

            metrics_epoch[f'{stage}_loss'].append(loss.item())
            metrics_epoch[f'{stage}_r2_score'].append(r_square.item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 100 == 0:
        plt.subplot(2, 1, 1)
        plt.title('loss l1')
        plt.plot(metrics['train_loss'])
        plt.plot(metrics['test_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train', 'Test'])

        plt.subplot(2, 1, 2)
        plt.title('r2_score')
        plt.plot(metrics['train_r2_score'])
        plt.plot(metrics['test_r2_score'])
        plt.xlabel('epoch')
        plt.ylabel('r2')
        plt.legend(['Train', 'Test'])

        plt.tight_layout()
        plt.show()