import argparse
import copy
from datetime import datetime
import os

import itertools
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.utils import data
import torch.utils.data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

from loader import Dataset
from pretrained_classification import Model
from csvutils.file_utils import FileUtils
from csvutils.csv_utils_2 import CsvUtils2


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_name', default='emotion_transfer', type=str)
parser.add_argument('-num_epochs', default=10000, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-learning_rate_g', default=1e-4, type=float)
parser.add_argument('-learning_rate_d', default=1e-4, type=float)
parser.add_argument('-lambda_cyc', default=10, type=float)
parser.add_argument('-lambda_iden', default=5, type=float)
parser.add_argument('-d_iter', default=5, type=int)
parser.add_argument('-source_emotion', default="neutral", type=str)
parser.add_argument('-target_emotion', default="happiness", type=str)
parser.add_argument("--local_rank", default=0, type=int)
args, _ = parser.parse_known_args()

"""
Create every combination of runs from batch_size and learning_rate
"""
args_with_multiple_values = {}
for key, value in args.__dict__.items():
    if isinstance(value, list):
        if len(value) > 1:
            args_with_multiple_values[key] = value
grid_runs = list(ParameterGrid(args_with_multiple_values))


BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = 8
EPOCHS = args.num_epochs
INPUT_SIZE = 64
D_ITER = args.d_iter
SOURCE_EMOTION = args.source_emotion
TARGET_EMOTION = args.target_emotion
NUM_WORKERS = 8


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(num_features=channels),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(num_features=channels)
        )

    def forward(self, x):
        return x + self.block.forward(x)


class Generator(nn.Module):
    def __init__(self, channels, num_block=3):
        super().__init__()
        self.channels = channels

        model = [nn.ReflectionPad2d(padding=3)]
        model += self._create_layer(in_channels=self.channels, out_channels=64, kernel_size=7, stride=1, padding=0)
        # downsample
        model += self._create_layer(in_channels=64, out_channels=128, downsample=True)
        model += self._create_layer(in_channels=128, out_channels=128)
        model += self._create_layer(in_channels=128, out_channels=256, downsample=True)
        # residual blocks
        model += [ResidualBlock(channels=256) for _ in range(num_block)]
        # upsample
        model += self._create_layer(in_channels=256, out_channels=128, upsample=True)
        model += self._create_layer(in_channels=128, out_channels=128)
        model += self._create_layer(in_channels=128, out_channels=64, upsample=True)
        # output
        model += [nn.ReflectionPad2d(padding=3),
                  nn.Conv2d(in_channels=64, out_channels=self.channels, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def _create_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=False, upsample=False):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.InstanceNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
        return layers

    def forward(self, x):
        return self.model.forward(x)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, decay_rate=0.0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x


class Critic(nn.Module):
    def __init__(self, channels, std=0.1, std_decay_rate=0.0):
        super().__init__()
        self.std = std
        self.std_decay_rate = std_decay_rate
        self.channels = channels

        self.model = nn.Sequential(
            # out = (in + 2p - k)/s + 1
            # Receptive field size for Patch: (out - 1) * stride + kernel
            *self._create_layer(in_channels=self.channels, out_channels=64, img_size_out=INPUT_SIZE//2), # out 32
            *self._create_layer(in_channels=64, out_channels=64, img_size_out=INPUT_SIZE//2, downsample=False),
            *self._create_layer(in_channels=64, out_channels=128, img_size_out=INPUT_SIZE//4),  # out 16
            *self._create_layer(in_channels=128, out_channels=128, img_size_out=INPUT_SIZE//4, downsample=False),
            *self._create_layer(in_channels=128, out_channels=256, img_size_out=INPUT_SIZE//8), # out 8
            *self._create_layer(in_channels=256, out_channels=256, img_size_out=INPUT_SIZE//8, downsample=False),
            *self._create_layer(in_channels=256, out_channels=512, img_size_out=INPUT_SIZE//16),  # out 4
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def _create_layer(self, in_channels, out_channels, img_size_out, downsample=True):
        layers = [GaussianNoise(std=self.std, decay_rate=self.std_decay_rate)]
        if downsample:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LayerNorm(normalized_shape=[out_channels, img_size_out, img_size_out]))
        layers.append(nn.LeakyReLU())
        return layers

    def forward(self, x):
        return self.model.forward(x)


def gradient_penalty(critic, real_data, fake_data, penalty, device):
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
    critic_interpolates = critic(interpolates)

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

def decay_gauss_std(net):
    for m in net.modules():
        if isinstance(m, GaussianNoise):
            m.decay_step()

# model_D_s = Critic(channels=1)
# model_G_s_t = Generator(channels=1)
# def get_param_count(model):
#     params = list(model.parameters())
#     result = 0
#     for param in params:
#         count_param = np.prod(param.size()) # size[0] * size[1] ...
#         result += count_param
#     return result
#
# print(f'model_D params: {get_param_count(model_D_s)}')
# print(f'model_G params: {get_param_count(model_G_s_t)}')
# exit()

torch.distributed.init_process_group(backend='nccl')
DEVICE = torch.device('cuda', args.local_rank)

date_time = datetime.now()
date_str = date_time.strftime("%y-%m-%d_%H:%M:%S")

run_num = 0
for run in grid_runs:
    run_num += 1
    path_run = f'./results/{date_str}_{args.run_name}_{run_num}'
    path_images = f'./results/{date_str}_{args.run_name}_{run_num}/imgs'
    # path_sequence = './results'
    args_dict = copy.deepcopy(args.__dict__)
    args_dict["run_name"] = f"{date_str}_run_{run_num}"
    args_dict["sequence_name"] = f"{date_str}_seq_run_{run_num}"
    # args_dict["learning_rate_g"] = run["learning_rate_g"]
    # args_dict["learning_rate_d"] = run["learning_rate_d"]
    if args.local_rank == 0:
        FileUtils.createDir(path_run)
        FileUtils.createDir(path_images)
        FileUtils.writeJSON(f'{path_run}/args.json', args_dict)
        CsvUtils2.create_global(path_sequence=f'{path_run}')
        CsvUtils2.create_local(path_sequence=f'{path_run}', run_name=f'{args.run_name}_{run_num}')

    model_G_s_t = Generator(channels=1).to(DEVICE)
    model_G_s_t = torch.nn.parallel.DistributedDataParallel(model_G_s_t,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
    model_G_t_s = Generator(channels=1).to(DEVICE)
    model_G_t_s = torch.nn.parallel.DistributedDataParallel(model_G_t_s,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
    model_D_s = Critic(channels=1, std=0.1, std_decay_rate=0.01).to(DEVICE)
    model_D_s = torch.nn.parallel.DistributedDataParallel(model_D_s,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
    model_D_t = Critic(channels=1, std=0.1, std_decay_rate=0.01).to(DEVICE)
    model_D_t = torch.nn.parallel.DistributedDataParallel(model_D_t,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
    # Setup classificator model for metric
    if args.local_rank == 0:
        classifier_emo_dict = {
            'neutral': 0,
            'happiness': 1,
            'sadness': 2,
            'anger': 3
        }
        classificator = Model(CLASS_NUM=4, in_image_channels=1, pretrained=False).to(DEVICE)
        classificator.load_state_dict(torch.load('./model-best_4classes.pt', map_location=torch.device(DEVICE)))
        classificator.eval()

    transform = torchvision.transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip()
    ])

    # source
    source_dataset_train = Dataset(f'../data/fer_48_{SOURCE_EMOTION}.hdf5', mode='train', img_size=48, transform=transform)
    sampler_source_train = DistributedSampler(source_dataset_train)
    loader_params = {'batch_size': BATCH_SIZE, 'sampler': sampler_source_train, 'num_workers': NUM_WORKERS}
    dataloader_train_source = data.DataLoader(source_dataset_train, **loader_params)

    if args.local_rank == 0: # only create on one gpu
        source_dataset_test = Dataset(f'../data/fer_48_{SOURCE_EMOTION}.hdf5', mode='test', img_size=48, transform=transform)
        sampler_source_test = DistributedSampler(source_dataset_test)
        loader_params = {'batch_size': TEST_BATCH_SIZE, 'sampler': sampler_source_test, 'num_workers': NUM_WORKERS}
        dataloader_test_source = data.DataLoader(source_dataset_test, **loader_params)

    # target
    target_dataset_train = Dataset(f'../data/fer_48_{TARGET_EMOTION}.hdf5', mode='train', img_size=48, transform=transform)
    sampler_target_train = DistributedSampler(target_dataset_train)
    loader_params = {'batch_size': BATCH_SIZE, 'sampler': sampler_target_train, 'num_workers': NUM_WORKERS}
    dataloader_train_target = data.DataLoader(target_dataset_train, **loader_params)

    params_G = itertools.chain(model_G_s_t.parameters(), model_G_t_s.parameters())
    optimizer_G = torch.optim.Adam(params_G, lr=args_dict["learning_rate_g"], betas=(0.0, 0.9))
    optimizer_D_s = torch.optim.Adam(model_D_s.parameters(), lr=args_dict["learning_rate_d"], betas=(0.0, 0.9))
    optimizer_D_t = torch.optim.Adam(model_D_t.parameters(), lr=args_dict["learning_rate_d"], betas=(0.0, 0.9))

    if args.local_rank == 0:
        metrics = {}
        for stage in ['train']:
            for metric in [
                f'{TARGET_EMOTION}_acc',
                'loss_g',
                'loss_d',
                'loss_c',
                'loss_i',
                'source_grad_penalty',
                'target_grad_penalty'
            ]:
                metrics[f'{stage}_{metric}'] = []

    best_accuracy = 0.0
    for epoch in range(1, EPOCHS):
        sampler_target_train.set_epoch(epoch)
        sampler_source_train.set_epoch(epoch)
        if args.local_rank == 0:
            metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        iter_data_loader_target = iter(dataloader_train_target)

        n = 0
        for x_s, label_s in tqdm(dataloader_train_source, desc=stage):
            x_t, label_t = next(iter_data_loader_target)

            if x_s.size() != x_t.size():
                continue
            x_s = x_s.to(DEVICE)  # source
            x_t = x_t.to(DEVICE)  # target

            if n % args_dict["d_iter"] == 0:
                optimizer_G.zero_grad()
                for param in model_D_s.parameters():
                    param.requires_grad = False
                for param in model_D_t.parameters():
                    param.requires_grad = False
                ### Train G
                g_t = model_G_s_t.forward(x_s)
                g_s = model_G_t_s.forward(x_t)

                # generator loss
                y_g_t = model_D_t.forward(g_t)
                y_g_s = model_D_s.forward(g_s)
                loss_g = -torch.mean(y_g_t) - torch.mean(y_g_s)

                # cycle loss
                recov_s = model_G_t_s.forward(g_t)
                loss_c_s = torch.mean(torch.abs(recov_s - x_s)) * args_dict["lambda_cyc"]
                recov_t = model_G_s_t.forward(g_s)
                loss_c_t = torch.mean(torch.abs(recov_t - x_t)) * args_dict["lambda_cyc"]
                loss_c = loss_c_s + loss_c_t

                # identity loss
                i_t = model_G_s_t.forward(x_t)
                i_s = model_G_t_s.forward(x_s)
                loss_i_t = torch.mean(torch.abs(i_t - x_t)) * args_dict["lambda_iden"]
                loss_i_s = torch.mean(torch.abs(i_s - x_s)) * args_dict["lambda_iden"]
                loss_i = loss_i_s + loss_i_t

                loss = loss_g + loss_c + loss_i
                loss.backward()
                optimizer_G.step()

                if args.local_rank == 0:
                    with torch.no_grad():
                        # Normalize g_t from (-1, 1) back to (0, 1) for classifier
                        mid = 255.0 / 2
                        g_t = (g_t * mid) + mid
                        y_prim = classificator.forward(g_t)
                        y_idx = torch.argmax(y_prim, dim=1)
                        acc = torch.mean((y_idx == classifier_emo_dict[TARGET_EMOTION]) * 1.0)

                    metrics_epoch[f'{stage}_loss_i'].append(loss_i.cpu().item())
                    metrics_epoch[f'{stage}_loss_g'].append(loss_g.cpu().item())
                    metrics_epoch[f'{stage}_loss_c'].append(loss_c.cpu().item())
                    metrics_epoch[f'{stage}_{TARGET_EMOTION}_acc'].append(acc.cpu().item())
            else:
                ### Train D_s
                for param in model_D_s.parameters():
                    param.requires_grad = True
                optimizer_D_s.zero_grad()

                y_x_s = model_D_s.forward(x_s)
                g_s = model_G_t_s.forward(x_t)
                y_g_s = model_D_s.forward(g_s.detach())
                penalty_s = gradient_penalty(critic=model_D_s,
                                           real_data=x_s,
                                           fake_data=g_s,
                                           penalty=10,
                                           device=DEVICE)

                loss_ds = torch.mean(y_g_s) - torch.mean(y_x_s) + penalty_s
                loss_ds.backward()
                optimizer_D_s.step()

                ### Train D_t
                for param in model_D_t.parameters():
                    param.requires_grad = True
                optimizer_D_t.zero_grad()

                y_x_t = model_D_t.forward(x_t)
                g_t = model_G_s_t.forward(x_s)
                y_g_t = model_D_t.forward(g_t.detach())
                penalty_t = gradient_penalty(critic=model_D_t,
                                           real_data=x_t,
                                           fake_data=g_t,
                                           penalty=10,
                                           device=DEVICE)

                loss_dt = torch.mean(y_g_t) - torch.mean(y_x_t) + penalty_t
                loss_dt.backward()
                optimizer_D_t.step()

                loss_d = loss_ds + loss_dt

                if args.local_rank == 0:
                    metrics_epoch[f'{stage}_loss_d'].append(loss_d.cpu().item())
                    metrics_epoch[f'{stage}_source_grad_penalty'].append(penalty_s.cpu().item())
                    metrics_epoch[f'{stage}_target_grad_penalty'].append(penalty_t.cpu().item())
            n += 1

        if args.local_rank == 0:
            for key in metrics_epoch.keys():
                value = 0
                if len(metrics_epoch[key]):
                    value = np.mean(metrics_epoch[key])
                metrics[key].append(value)

            metrics_dict = {}
            for key, value in metrics.items():
                metrics_dict[key] = value[-1]

            CsvUtils2.add_hparams(
                path_sequence=path_run,
                run_name=f'{args.run_name}_{run_num}',
                args_dict=args.__dict__,
                metrics_dict=metrics_dict,
                global_step=epoch
            )

            if epoch % 4 == 0:
                with torch.no_grad():
                    sampler_source_test.set_epoch(epoch)
                    x_s, labels_s = next(iter(dataloader_test_source))
                    x_s = x_s.to(DEVICE)
                    fake_t = model_G_s_t.forward(x_s)
                    recovered_s = model_G_t_s.forward(fake_t)
                    viz_sample = torch.cat((x_s, fake_t, recovered_s), 0)
                    # Calculate accuracy for emotion transfer
                    mid = 255.0 / 2
                    fake_t = (fake_t * mid) + mid
                    y_prim = classificator.forward(fake_t)
                    y_idxs = torch.argmax(y_prim, dim=1)
                    acc = torch.mean((y_idxs == 1) * 1.0)
                    vutils.save_image(viz_sample,
                                      os.path.join(path_images, f'samples_{epoch}_test-acc_{acc}.png'),
                                      nrow=TEST_BATCH_SIZE,
                                      normalize=True)
                plt.clf()
                plt.cla()

                plts = []
                c = 0
                for key, value in metrics.items():
                    plts += plt.plot(value, f'C{c}', label=key)
                    ax = plt.twinx()
                    c += 1
                plt.legend(plts, [it.get_label() for it in plts])
                plt.tight_layout(pad=0.5)
                plt.savefig(f'{path_images}/plt-{epoch}.png')

        # Decay gaussian noise
        decay_gauss_std(model_D_s)
        decay_gauss_std(model_D_t)

    # Save generator for inference (and FID score calculation)
    torch.save(model_G_s_t.cpu().state_dict(), f'{path_run}/model_G_s_t_final.pt')
