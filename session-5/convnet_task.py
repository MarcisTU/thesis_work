import torch
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
TRAIN_MAX_LEN = 0
TEST_MAX_LEN = 0
IMAGE_SIZE = 200
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    TRAIN_MAX_LEN = 0
    TEST_MAX_LEN = 0


class DatasetBeans(torch.utils.data.Dataset):
    def __init__(self, is_train, max_length, rescale_size=None):
        super().__init__()
        self.split = 'train' if is_train else 'test'
        self.max_length = max_length
        self.data = []
        self.load_beans_dataset()

        self.rescale_size = rescale_size

    def __len__(self):
        if self.max_length:
            return self.max_length
        return len(self.data)

    def __getitem__(self, idx):
        img, y_idx = self.data[idx]
        np_img = np.array(img)
        img = torch.FloatTensor(np_img)

        # Change dimensions/permute (W, H, C) -> (C, W, H)
        img = img.permute(2, 0, 1)

        if self.rescale_size:
            img = torch.nn.functional.adaptive_avg_pool2d(
                img, output_size=(self.rescale_size, self.rescale_size))

        return img, y_idx

    def load_beans_dataset(self):
        ds, ds_info = tfds.load('beans', split=self.split, with_info=True)
        ds_extract = [(item['image'].numpy(), item['label'].numpy()) for item in ds]
        self.data = np.array(ds_extract, dtype=object)


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetBeans(is_train=True, max_length=TRAIN_MAX_LEN, rescale_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetBeans(is_train=False, max_length=TEST_MAX_LEN, rescale_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

def get_out_size(in_size, padding, kernel_size, stride):
    return int((in_size + 2*padding - kernel_size) / stride + 1)


class Conv2d(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = torch.nn.Parameter(
            torch.FloatTensor(kernel_size, kernel_size, in_channels, out_channels)
        )
        torch.nn.init.xavier_uniform_(self.K)

    def forward(self, x):
        batch_size = x.size(0)
        in_size = x.size(-1)  # last dim from (B, C, W, H)
        out_size = get_out_size(in_size, self.padding, self.kernel_size, self.stride)

        out = torch.zeros(batch_size, self.out_channels, out_size, out_size).to(DEVICE)

        x_padded_size = in_size+self.padding*2
        x_padded = torch.zeros(batch_size, self.in_channels, x_padded_size, x_padded_size).to(DEVICE)
        x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x

        K = self.K.view(-1, self.out_channels)  # self.kernel_size*self.kernel_size*self.in_channels

        i_out = 0
        for i in range(0, x_padded_size - self.kernel_size, self.stride):
            j_out = 0
            for j in range(0, x_padded_size - self.kernel_size, self.stride):
                x_part = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                x_part = x_part.reshape(batch_size, -1)  # self.kernel_size*self.kernel_size*self.in_channels

                out_part = x_part @ K  # (B, out_channels)
                out[:, :, i_out, j_out] = out_part

                j_out += 1
            i_out += 1

        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        out_channels = 8
        self.encoder = torch.nn.Sequential(
            Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            Conv2d(in_channels=8, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        o_1 = get_out_size(IMAGE_SIZE, kernel_size=5, stride=2, padding=1)
        o_2 = get_out_size(o_1, kernel_size=3, stride=1, padding=1)
        o_3 = get_out_size(o_2, kernel_size=3, stride=1, padding=1)
        o_3 = o_3//2  # After maxpooling

        self.fc = torch.nn.Linear(
            in_features=out_channels*o_3*o_3,
            out_features=3
        )

    def forward(self, x):
        batch_size = x.size(0)  # x.size() => (B, C_in, W_in, H_in)
        out = self.encoder.forward(x)  # out.size() => (B, C_out, W_out, H_out)
        out_flat = out.view(batch_size, -1)  # out_flat.size() => (B, F)
        logits = self.fc.forward(out_flat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 101):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        batch_count = 0
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y_idx in data_loader:
            batch_count += 1
            x = x.to(DEVICE)
            y_idx = y_idx.to(DEVICE)

            y_prim = model.forward(x)

            idx = range(len(y_idx))
            loss = torch.mean(-torch.log(y_prim + 1e-8)[idx, y_idx])

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y_idx = y_idx.cpu().data.numpy()
            idx_y_prim = np.argmax(np_y_prim, axis=1)
            acc = np.average((np_y_idx == idx_y_prim) * 1.0)

            print(f'Batch {batch_count} done.')

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 2 == 0:
        params = {
            'loss': 'Categorical Cross Entropy Loss',
            'acc': 'Accuracy'
        }

        fig, axs = plt.subplots(3, sharex=True)
        for param, ax in zip(params, axs):
            ax.set_title(params[param])
            ax.set_ylabel(param)
            ax.plot(metrics[f'train_{param}'], label='train')
            ax.plot(metrics[f'test_{param}'], label='test')

        lines, labels = axs[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper right')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.show()