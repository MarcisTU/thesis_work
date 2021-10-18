import random
import time
import argparse
import tensorflow_datasets as tfds
import tensorboard_utils
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional
# from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default=f'seq_default', type=str)
parser.add_argument('-is_cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-learning_rate', default=1e-5, type=float)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-epochs', default=40, type=int)
args = parser.parse_args()

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
TRAIN_MAX_LEN = 0
TEST_MAX_LEN = 0
IMAGE_SIZE = 250
DEVICE = 'cpu'

if args.is_cuda:
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
        t_img = torch.FloatTensor(np_img)

        # Change dimensions/permute (W, H, C) -> (C, W, H)
        t_img = t_img.permute(2, 0, 1)

        if self.rescale_size:
            t_img = torch.nn.functional.adaptive_avg_pool2d(
                t_img, output_size=(self.rescale_size, self.rescale_size))

        return t_img, y_idx

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


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 3
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=5,
                kernel_size=(5, 5),
                stride=(2, 2),
                padding=1
            ),
            torch.nn.BatchNorm2d(
                num_features=5
            ),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(
                in_channels=5,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1
            ),
            torch.nn.BatchNorm2d(
                num_features=8
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(
                num_features=8
            ),

            torch.nn.Conv2d(
                in_channels=8,
                out_channels=3,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        # x.shape = (B, 1, W, H)
        out = self.encoder.forward(x)
        y_prim = out.view(-1, self.out_channels)
        y_prim = torch.softmax(y_prim, dim=1)
        return y_prim


def f1_score(conf_m: np.ndarray):
    FP = np.sum(conf_m.sum(axis=0) - np.diag(conf_m))
    FN = np.sum(conf_m.sum(axis=1) - np.diag(conf_m))
    TP = np.sum(np.diag(conf_m))
    TN = np.sum(conf_m.sum() - (FP + FN + TP))

    f1_sc = (2 * TP) / (2 * TP + FP + FN)
    return f1_sc

model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}'
)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc',
        'f1-score'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, args.epochs + 1):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        class_count = 3
        conf_matrix = np.zeros((class_count, class_count))
        for x, y_idx in data_loader:
            #CAUTION random resize here!!! model must work regardless
            out_size = int(IMAGE_SIZE * (random.random() * 0.3 + 1.0))
            x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(out_size, out_size))

            x = x.to(DEVICE)
            y = y_idx.to(DEVICE)

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

            # Confusion matrix
            for idx_sample in range(len(y_idx)):
                conf_matrix[y_idx[idx_sample], idx_y_prim[idx_sample]] += 1

            # f1 score
            f1 = f1_score(conf_matrix)
            #f1 = f1_score(y_idx, idx_y_prim, average='micro')  ## Scikit learn metric

            metrics_epoch[f'{stage}_f1-score'].append(f1)
            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        """
        Update all the metrics with current value and best value so far
        """
        metrics_strs = []
        best_value = 0
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

                summary_writer.add_scalar(
                    tag=f'{key}',
                    scalar_value=round(value, 2),
                    global_step=epoch
                )

                if 'loss' in key:
                    best_value = np.min(metrics[key])
                else:
                    best_value = np.max(metrics[key])

                summary_writer.add_hparams(
                    hparam_dict=args.__dict__,
                    metric_dict={
                       f'best_{key}': best_value
                    },
                    name=args.run_name,
                    global_step=epoch
                )

        """
        Visualize confusion matrix and add it to Tensorboard
        """
        fig = plt.figure()
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
        plt.xticks([0, 1, 2], ['angular_leaf_spot', 'bean_rust', 'healthy'])
        plt.yticks([0, 1, 2], ['angular_leaf_spot', 'bean_rust', 'healthy'])
        for x in range(class_count):
            for y in range(class_count):
                plt.annotate(
                    str(round(100 * conf_matrix[x, y] / np.sum(conf_matrix[x]), 1)), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    backgroundcolor='white'
                )
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.tight_layout()

        summary_writer.add_figure(
            tag=f'{stage}_conf_matrix',
            figure=fig,
            global_step=epoch
        )

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    """
    Plot every 2 epochs
    """
    if epoch % 2 == 0:
        params = {
            'loss' : 'Categorical Cross Entropy Loss',
            'acc' :  'Accuracy',
            'f1-score' : 'F1-score'
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

    summary_writer.flush()
summary_writer.close()