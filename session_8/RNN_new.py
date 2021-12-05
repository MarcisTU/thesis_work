import os
import pickle
import time
import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import json
from tqdm import tqdm

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

plt.rcParams["figure.figsize"] = (10, 16)  # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 1000

EMBEDDING_SIZE = 32
RNN_HIDDEN_SIZE = 128
RNN_LAYERS = 3
RNN_NAME = "lstm"
RNN_IS_BIDIRECTIONAL = False

TRAIN_TEST_SPLIT = 0.8

MAX_LEN = 1000  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    # comment out this next line if you have nvidia GPU and you want to debug
    MAX_LEN = None
    pass


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/quotes.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/quotes.pkl',
                path_dataset,
                progress=True
            )

        with open(path_dataset, 'rb') as fp:
            (
                self.final_quotes_sentences, self.final_authors, self.final_categories,
                self.vocabulary_keys, self.vocabulary_counts, self.authors_keys, self.categories_keys
            ) = pickle.load(fp)
        self.max_sentence_length = np.max([len(it) for it in self.final_quotes_sentences])

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.final_quotes_sentences)

    def __getitem__(self, idx):
        x_raw = np.array(self.final_quotes_sentences[idx])

        y = np.roll(x_raw, -1)  # move all words by 1 position to left
        y = y[:-1]
        x = x_raw[:-1]
        x_length = len(x)

        pad_right = self.max_sentence_length - x_length
        pad_left = 0
        x_padded = np.pad(x, (pad_left, pad_right))
        y_padded = np.pad(y, (pad_left, pad_right))

        return x_padded, y_padded, x_length


dataset_full = Dataset()

train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1)
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    drop_last=(len(dataset_test) % BATCH_SIZE == 1)
)


class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        stdu = 1 / np.sqrt(RNN_HIDDEN_SIZE)
        self.W_x = torch.nn.Parameter(
            torch.FloatTensor(EMBEDDING_SIZE, RNN_HIDDEN_SIZE).uniform_(-stdu, stdu)
        )
        self.W_h = torch.nn.Parameter(
            torch.FloatTensor(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE).uniform_(-stdu, stdu)
        )
        self.b_h = torch.nn.Parameter(
            torch.FloatTensor(RNN_HIDDEN_SIZE).zero_()
        )

    def forward(self, x: PackedSequence, hidden=None):
        x_unpacked, x_lengths = pad_packed_sequence(x, batch_first=True)
        batch_size = x_unpacked.size(0)
        if hidden is None:
            hidden = torch.FloatTensor(batch_size, RNN_HIDDEN_SIZE).zero_().to(DEVICE)

        # x_unpacked.size() => (B, SEQ, EMBEDDING_SIZE)
        x_seq = x_unpacked.permute(1, 0, 2)  # => (SEQ, B, EMBEDDING_SIZE)
        list_out = []
        for x_t in x_seq:
            # x_t => (B, EMBEDDING_SIZE)
            hidden = torch.tanh(
                (self.W_x.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +
                (self.W_h.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +
                self.b_h
            )
            list_out.append(hidden)
        out_seq = torch.stack(list_out)
        out = out_seq.permute(1, 0, 2)
        out_packed = pack_padded_sequence(out, x_lengths, batch_first=True, enforce_sorted=False)

        return out_packed, hidden


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        stdu = 1 / np.sqrt(RNN_HIDDEN_SIZE)
        self.W_f = torch.nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.W_i = torch.nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.W_o = torch.nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.W_c = torch.nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).uniform_(-stdu, stdu)
        )

        self.U_f = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.U_i = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.U_o = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.U_c = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_(-stdu, stdu)
        )

        self.b_f = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size).fill_(value=1)
        )
        self.b_i = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size).zero_()
        )
        self.b_o = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size).zero_()
        )
        self.b_c = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size).zero_()
        )
        self.hidden = torch.nn.Parameter(
            torch.FloatTensor(BATCH_SIZE, self.hidden_size).zero_()
        )
        self.c_hidden = torch.nn.Parameter(
            torch.FloatTensor(BATCH_SIZE, self.hidden_size).zero_()
        )
        self.zoneout_prob = 0.3

        self.forget_layer_norm = torch.nn.LayerNorm(normalized_shape=[1, self.hidden_size])
        self.input_layer_norm = torch.nn.LayerNorm(normalized_shape=[1, self.hidden_size])
        self.internal_layer_norm = torch.nn.LayerNorm(normalized_shape=[1, self.hidden_size])
        self.output_layer_norm = torch.nn.LayerNorm(normalized_shape=[1, self.hidden_size])

        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x: PackedSequence, hidden=None):
        x_unpacked, x_lengths = pad_packed_sequence(x, batch_first=True)

        if hidden is None:
            hidden = self.hidden.to(DEVICE)
        c_hidden = self.c_hidden.to(DEVICE)

        # x_unpacked.size() => (B, SEQ, EMBEDDING_SIZE)
        x_seq = x_unpacked.permute(1, 0, 2)  # => (SEQ, B, EMBEDDING_SIZE)
        list_out = []
        for x_t in x_seq:
            # forget gate
            f_t = (self.W_f.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                  (self.U_f.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                  self.b_f
            f_t = self.forget_layer_norm.forward(f_t.unsqueeze(dim=1)).squeeze(dim=1)
            f_t = self.sigmoid.forward(f_t)
            # input
            i_t = (self.W_i.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                  (self.U_i.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                  self.b_i
            i_t = self.input_layer_norm.forward(i_t.unsqueeze(dim=1)).squeeze(dim=1)
            i_t = self.sigmoid.forward(i_t)
            # output
            o_t = (self.W_o.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                  (self.U_o.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                  self.b_o
            o_t = self.output_layer_norm.forward(o_t.unsqueeze(dim=1)).squeeze(dim=1)
            o_t = self.sigmoid.forward(o_t)
            # candidate memory
            c_hat_t = (self.W_c.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                      (self.U_c.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                      self.b_c
            c_hat_t = self.internal_layer_norm.forward(c_hat_t.unsqueeze(dim=1)).squeeze(dim=1)
            c_hat_t = self.sigmoid.forward(c_hat_t)

            # zoneout
            p = np.random.uniform(0, 1)
            c_hidden = (p <= self.zoneout_prob)*c_hidden + \
                       (1 - (p <= self.zoneout_prob))*(f_t * c_hidden + i_t * c_hat_t)
            hidden = (p <= self.zoneout_prob)*hidden + \
                     (1 - (p <= self.zoneout_prob))*(o_t * self.sigmoid.forward(c_hidden))

            # c_hidden = f_t * c_hidden + i_t * c_hat_t
            # hidden = o_t * self.sigmoid.forward(c_hidden)
            # dropout
            # c_hidden = self.dropout.forward(c_hidden)
            # hidden = self.dropout.forward(hidden)

            list_out.append(hidden)

        out_seq = torch.stack(list_out)
        out = out_seq.permute(1, 0, 2)
        out_packed = pack_padded_sequence(out, x_lengths, batch_first=True, enforce_sorted=False)

        return out_packed, hidden


class GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.sigmoid = torch.nn.Sigmoid()
        self.z_layer_norm = torch.nn.LayerNorm(normalized_shape=[1, self.hidden_size])
        self.r_layer_norm = torch.nn.LayerNorm(normalized_shape=[1, self.hidden_size])
        self.h_layer_norm = torch.nn.LayerNorm(normalized_shape=[1, self.hidden_size])

        self.zoneout_prob = 0.3

        stdu = 1 / np.sqrt(self.hidden_size)
        self.W_z = torch.nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.W_r = torch.nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.W_h = torch.nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.U_z = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.U_r = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.U_h = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_(-stdu, stdu)
        )
        self.b_z = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size).zero_()
        )
        self.b_r = torch.nn.Parameter(
            torch.FloatTensor(self.hidden_size).fill_(value=-1)
        )
        self.hidden = torch.nn.Parameter(
            torch.FloatTensor(BATCH_SIZE, self.hidden_size).zero_()
        )
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x: PackedSequence, hidden=None):
        x_unpacked, x_lengths = pad_packed_sequence(x, batch_first=True)

        if hidden is None:
            hidden = self.hidden.to(DEVICE)

        # x_unpacked.size() => (B, SEQ, EMBEDDING_SIZE)
        x_seq = x_unpacked.permute(1, 0, 2)  # => (SEQ, B, EMBEDDING_SIZE)
        list_out = []
        for x_t in x_seq:
            # x_t => (B, EMBEDDING_SIZE)
            # update gate allows us to control how much of the new state is just a copy of the old state
            z = (self.W_z.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                (self.U_z.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                self.b_z
            z = self.z_layer_norm.forward(z.unsqueeze(dim=1)).squeeze(dim=1)
            z = self.sigmoid.forward(z)
            # reset gate allows us to control how much of the previous state we might still want to remember
            r = (self.W_r.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                (self.U_r.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                self.b_r
            r = self.r_layer_norm.forward(r.unsqueeze(dim=1)).squeeze(dim=1)
            r = self.sigmoid.forward(r)
            # Candidate hidden state
            h_hat = (self.W_h.t() @ x_t.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                r * (self.U_h.t() @ hidden.unsqueeze(dim=-1)).squeeze(dim=-1) +\
                self.b_z
            h_hat = self.h_layer_norm.forward(h_hat.unsqueeze(dim=1)).squeeze(dim=1)
            h_hat = self.sigmoid.forward(h_hat)

            # zoneout
            # p = np.random.uniform(0, 1)
            # hidden = (p <= self.zoneout_prob) * hidden + \
            #          (1 - (p <= self.zoneout_prob)) * (z * hidden + (1 - z) * h_hat)

            # dropout
            hidden = z * hidden + (1 - z) * h_hat
            hidden = self.dropout.forward(hidden)

            list_out.append(hidden)

        out_seq = torch.stack(list_out)
        out = out_seq.permute(1, 0, 2)
        out_packed = pack_padded_sequence(out, x_lengths, batch_first=True, enforce_sorted=False)

        return out_packed, hidden


class LSTM_Block(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm1 = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.lstm2 = LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.lstm3 = LSTM(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, x, hidden):
        residual = x.data

        out, hidden = self.lstm1.forward(x, hidden)
        out_res = out.data + residual
        out_seq = PackedSequence(
            data=out_res,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        out, hidden = self.lstm2.forward(out_seq, hidden)
        out_res = out.data + residual
        out_seq = PackedSequence(
            data=out_res,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        out, hidden = self.lstm3.forward(out_seq, hidden)

        return out, hidden


class GRU_Block(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru1 = GRU(input_size=input_size, hidden_size=hidden_size)
        self.gru2 = GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.gru3 = GRU(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, x, hidden):
        residual = x.data

        out, hidden = self.gru1.forward(x, hidden)
        out_res = out.data + residual
        out_seq = PackedSequence(
            data=out_res,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        out, hidden = self.gru2.forward(out_seq, hidden)
        out_res = out.data + residual
        out_seq = PackedSequence(
            data=out_res,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        out, hidden = self.gru3.forward(out_seq, hidden)

        return out, hidden


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=len(dataset_full.vocabulary_keys),
            embedding_dim=EMBEDDING_SIZE
        )

        self.linear = torch.nn.Linear(in_features=EMBEDDING_SIZE, out_features=RNN_HIDDEN_SIZE)

        self.lstm_block_1 = LSTM_Block(input_size=RNN_HIDDEN_SIZE, hidden_size=RNN_HIDDEN_SIZE)
        # self.lstm_block_2 = LSTM_Block(input_size=RNN_HIDDEN_SIZE, hidden_size=RNN_HIDDEN_SIZE)
        # self.gru_block_1 = GRU_Block(input_size=RNN_HIDDEN_SIZE, hidden_size=RNN_HIDDEN_SIZE)
        # self.gru_block_2 = GRU_Block(input_size=RNN_HIDDEN_SIZE, hidden_size=RNN_HIDDEN_SIZE)

        self.fc = torch.nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=EMBEDDING_SIZE
        )

    def forward(self, x: PackedSequence, hidden=None):
        x_emb = self.emb.forward(x.data)  # x.data == Long sausage a_1, a_2, a_3, b_1, b_2 ...

        out_linear = self.linear.forward(x_emb)  # Add a linear layer for input data before RNN layers
        out_seq = PackedSequence(
            data=out_linear,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        ###
        out, hidden = self.lstm_block_1.forward(out_seq, hidden)
        # out, hidden = self.lstm_block_2.forward(out, hidden)
        ###

        out_fc = self.fc.forward(out.data)
        y_prim_logits = (self.emb.weight @ out_fc.unsqueeze(dim=-1)).squeeze(dim=-1)
        y_prim = torch.softmax(y_prim_logits, dim=-1)
        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        return y_prim_packed, hidden


model = Model()
model = model.to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_weights = torch.FloatTensor(1 - dataset_full.vocabulary_counts / np.sum(dataset_full.vocabulary_counts))
loss_weights = loss_weights.to(DEVICE)

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []

lowest_test_loss = float('inf')
best_model_state = None
total_epochs = 0
x_rollout = []

for epoch in range(1, EPOCHS):
    for dataloader in [dataloader_train, dataloader_test]:
        stage = 'train'
        if dataloader == dataloader_test:
            stage = 'test'
        losses = []
        accs = []
        for x_padded, y_padded, x_length in tqdm(dataloader, desc=stage):
            if x_padded.size(0) < BATCH_SIZE:  # if last batch size is not 32 skip
                continue
            x_padded = x_padded.to(DEVICE)
            y_padded = y_padded.to(DEVICE)
            x_rollout = x_padded

            x_packed = pack_padded_sequence(x_padded, x_length, batch_first=True, enforce_sorted=False)
            y_packed = pack_padded_sequence(y_padded, x_length, batch_first=True, enforce_sorted=False)

            y_prim_packed, _ = model.forward(x_packed)

            idxes_batch = range(len(y_packed.data))
            idxes_y = y_packed.data
            loss = -torch.mean(
                loss_weights[idxes_y] * torch.log(y_prim_packed.data[idxes_batch, idxes_y] + 1e-8)
            )
            losses.append(loss.cpu().item())

            idxes_y_prim = y_prim_packed.data.argmax(dim=-1)
            acc = torch.mean((idxes_y_prim == idxes_y) * 1.0)
            accs.append(acc.cpu().item())

            if stage == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
        else:
            loss_plot_test.append(np.mean(losses))
            if np.mean(losses) < lowest_test_loss:
                lowest_test_loss = np.mean(losses)
                torch.save(model.state_dict(), f'./model-{RNN_NAME}.pt')
            acc_plot_test.append(np.mean(accs))

    print(
        f'\n\nepoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]} '
        f'acc_plot_train: {acc_plot_train[-1]} '
        f'acc_plot_test: {acc_plot_test[-1]} '
    )

    if epoch % 4 == 0:
        fig, (ax1, ay1) = plt.subplots(2, 1)
        ax1.plot(loss_plot_train, 'r-', label='loss train')
        ax1.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='loss test')
        ax2.legend(loc='upper right')

        ay1.plot(acc_plot_train, 'r-', label='train acc')
        ay1.legend(loc='upper left')
        ay1.set_xlabel("Epoch")
        ay2 = ay1.twinx()
        ay2.plot(acc_plot_test, 'c-', label='test acc')
        ay2.legend(loc='upper right')

        plt.show()

        x_roll = x_rollout[:, :1]  # only first word as input
        hidden = torch.zeros((len(x_roll), RNN_HIDDEN_SIZE)).to(DEVICE)

        for t in range(dataset_full.max_sentence_length):
            lengths = torch.LongTensor([1] * x_roll.size(0))
            x_packed = pack_padded_sequence(x_roll[:, -1:], lengths, batch_first=True)

            y_prim, hidden = model.forward(x_packed, hidden)

            y_prim_unpacked, _ = pad_packed_sequence(y_prim, batch_first=True)
            y_prim_idx = y_prim_unpacked.argmax(dim=-1)
            x_roll = torch.cat((x_roll, y_prim_idx), dim=-1)

        np_x_roll = x_roll.cpu().data.numpy().tolist()
        for each_sent in np_x_roll:
            words_sent = [dataset_full.vocabulary_keys[it] for it in each_sent]

            if '[eos]' in words_sent:
                eos_idx = words_sent.index('[eos]')
                if eos_idx > 0:
                    words_sent = words_sent[:eos_idx]
            print(' '.join(words_sent))

    if len(loss_plot_test) > 20:
        # if test loss value converges stop training
        if np.abs((loss_plot_test[-1] - np.average(loss_plot_test[-20:-1]))) < 0.005:
            total_epochs = epoch
            print(np.abs((loss_plot_test[-1] - np.average(loss_plot_test[-20:-1]))))
            break
        # or test loss value is overfiting
        elif np.average(loss_plot_test[-20:-10]) < np.average(loss_plot_test[-10:-1]):
            break
