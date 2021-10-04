import time
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


BATCH_SIZE = 10
LEARNING_RATE = 1e-5


class DatasetPlacements(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = pd.read_csv('archive/Placement_Data_Full_Class.csv', sep=',')
        self.is_train = is_train
        self.features = None
        self.preprocess()
        self.standardize()
        self.load_data()

    def __len__(self):
        X, Y = self.data
        return len(X)

    def __getitem__(self, idx):
        x = self.data[0][idx]
        y = self.data[1][idx]

        return x, y

    def preprocess(self):
        self.data = self.data.drop(columns=['sl_no'])
        self.data["salary"].fillna(0.0, inplace=True)

        for col_name in self.data.columns.values:
            if self.data[col_name].dtype == 'object':
                category_vec = self.data[col_name].unique()
                one_hot_vec = np.arange(len(self.data[col_name].unique()))

                for el_index, el in enumerate(self.data[col_name]):
                    for cat_index, cat in enumerate(category_vec):
                        if el == cat:
                            # Categorize the variable to number
                            self.data.at[el_index, col_name] = one_hot_vec[cat_index]

        self.features  = self.data.columns.values
        # Convert to numpy
        self.data = self.data.to_numpy().astype(float)
        self.data = torch.FloatTensor(self.data)
        Y = self.data[:, 5].type(torch.IntTensor)
        X = torch.cat((self.data[:, :5], self.data[:, 6:]), 1)
        self.data = (X, Y)

    def standardize(self):
        x, y = self.data

        # Separate categorical (don't need to standardize them), only standardize continuous
        contin_col = [1, 3, 5, 8, 10, 12]
        for col in contin_col:
            # Standardize
            x_mean = torch.mean(x[:, col])
            x_std = torch.std(x[:, col])
            x[:, col] = (x[:, col] - x_mean) / x_std

        self.data = (x, y)

    def load_data(self):
        X, Y = self.data

        # shuffle
        torch.manual_seed(0)

        idxes_rand = torch.randperm(len(Y))
        X = X[idxes_rand]
        Y = Y[idxes_rand]

        # Slice off 5 samples to make batch of 10 possible to split
        X = X[:-5]
        Y = Y[:-5]

        Y_idxs = np.array(Y)
        Y = torch.zeros((len(Y), 3))
        Y[torch.arange(len(Y)), Y_idxs] = 1.0

        idx_split = int(len(X) * 0.8572)
        dataset_train = (X[:idx_split], Y[:idx_split])
        dataset_test = (X[idx_split:], Y[idx_split:])

        torch.manual_seed(int(time.time()))

        if self.is_train:
            self.data = dataset_train
        else:
            self.data = dataset_test


dataloader_train = torch.utils.data.DataLoader(
    dataset=DatasetPlacements(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=False
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=DatasetPlacements(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class Embedding(torch.nn.Module):
    """
    2d - means two different categories for given column, so 3d is 3 etc

    Dataset has categorical values in columns [0, 2, 4, 6, 7, 9, 11], but 6th has 3 different categories,
        so we separate it from calculation since it needs bigger embeddings matrix.

    offset_index is used to keep track of column position after inserting embeddings vector.
        For example: column 0 and column 2.
            vector before embedding     -> [0, 1, 0]
            and after embedding value 0 -> [0.0, 0.0, 0.0, 1, 0]
            now column 2 index has changed to 4
    """
    def __init__(self):
        super().__init__()
        self.num_embeddings_2d = 2
        self.embedding_dim_2d = 3
        self.num_embeddings_3d = 3
        self.embedding_dim_3d = 5
        self.categorical_columns_2d = [0, 2, 4, 7, 9, 11]
        self.categorical_columns_3d = [12]
        self.offset = 0
        self.calc_offset()
        """
            len(self.categorical_columns_2d) -> for every categorical data feature we need separate embeddings matrix
        """
        self.embedding_2d = torch.nn.Embedding(self.num_embeddings_2d, len(self.categorical_columns_2d) * self.embedding_dim_2d)
        self.embedding_3d = torch.nn.Embedding(self.num_embeddings_3d, len(self.categorical_columns_3d) * self.embedding_dim_3d)

    def calc_offset(self):
        # calculate the offset for 6th column after embedding 2d categorical
        for n in self.categorical_columns_2d:
            if n < 6:
                self.offset += 2 # Every embedding for 2 categorical values is adding 2 new columns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X_emb = []
        """
        Embed each sample in Batch replacing categorical data with embeddings vector.
        """
        for index, sample in enumerate(x):
            # Embedding for 2 categories
            embedded_sample = self.embed(sample, self.categorical_columns_2d, self.num_embeddings_2d, self.embedding_dim_2d, self.embedding_2d)
            # Embedding for 3 categories
            embedded_sample = self.embed(embedded_sample, self.categorical_columns_3d, self.num_embeddings_3d, self.embedding_dim_3d, self.embedding_3d)
            X_emb.append(embedded_sample)

        X_emb = torch.stack(X_emb)
        return X_emb

    """
    Helper function, that "embeds" replaces categorical value with embeddings vector, e.g: 1 with vector [0.0, 0.0, 0.0]
    Vector size depends on number of categories.
    """
    def embed(self, embed_sample, categorical_columns, num_embeddings, embedding_dim, emb_m):
        offset_index = 0
        embedded_sample = embed_sample
        for index, cat in enumerate(categorical_columns):  # for every column that has categorical data (here predefined)
            category = int(embedded_sample[cat + offset_index])
            category_one_hot = torch.zeros(num_embeddings, )
            category_one_hot[category] = 1.0
            category_one_hot = torch.unsqueeze(category_one_hot, 0)
            category_emb = torch.squeeze(category_one_hot @ emb_m.weight.data[:, index * embedding_dim:(index * embedding_dim) + embedding_dim])

            embedded_sample = torch.cat((embedded_sample[:cat + offset_index], category_emb, embedded_sample[cat + offset_index + 1:]), 0)
            offset_index += embedding_dim - 1
        return embedded_sample


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Embedding(),
            torch.nn.Linear(in_features=29, out_features=22),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=22, out_features=15),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=15, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=3),
            torch.nn.Softmax()
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
      'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 801):
    for data_loader in [dataloader_train, dataloader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataloader_test:
            stage = 'test'

        for x, y in data_loader:
            y_prim = model.forward(x)
            loss = torch.mean(-torch.sum(y * torch.log(y_prim + 1e-8)))

            y = y.detach()
            y_prim = y_prim.detach()
            # Accuracy
            pred_max = torch.argmax(y_prim, 1)
            actual_max = torch.argmax(y, 1)
            true_val = (pred_max == actual_max) * 1
            accuracy = torch.sum(true_val) / len(true_val)

            if stage == 'train':
                loss.backward()
                optimizer.step()

            metrics_epoch[f'{stage}_loss'].append(loss.item())
            metrics_epoch[f'{stage}_acc'].append(accuracy.item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 100 == 0:
        plt.subplot(2, 1, 1)
        plt.title('loss Cross Entropy')
        plt.plot(metrics['train_loss'])
        plt.plot(metrics['test_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train', 'Test'])

        plt.subplot(2, 1, 2)
        plt.title('accuracy')
        plt.plot(metrics['train_acc'])
        plt.plot(metrics['test_acc'])
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(['Train', 'Test'])

        plt.tight_layout()
        plt.show()
