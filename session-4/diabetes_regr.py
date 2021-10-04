import time
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt


X, Y = sklearn.datasets.load_diabetes(return_X_y=True)
Y = np.expand_dims(Y, axis=1)

# Std #standardized_dataset = (dataset - mean(dataset)) / standard_deviation(dataset))
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y_std = (Y - np.mean(Y)) / np.std(Y)

# encode sex feature
X_std[:, 1] = X_std[:, 1] >= 0


class Variable(object):
    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class LayerLinear(object):
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value=np.random.random((out_features, in_features))
        )
        self.b = Variable(
            value=np.zeros((out_features,))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad = 1 * self.output.grad
        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.value, axis=1),
        )
        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable) -> Variable:
        self.x = x
        self.output = Variable(
            (x.value >= 0) * x.value
        )
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0) * self.output.grad


class LayerEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.x: Variable = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_m = Variable(np.zeros((num_embeddings, embedding_dim)))
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = []
        for index, sample in enumerate(x.value):
            category = int(sample[1])
            sex_one_hot = np.zeros(self.num_embeddings,)
            sex_one_hot[category] = 1.0
            sex_one_hot = np.expand_dims(sex_one_hot, axis=0)
            sex_emb = np.squeeze(sex_one_hot @ self.emb_m.value)  # [e_1, e_2, e_3] (3,)

            embedded_sample = np.concatenate((sample[:1], sex_emb, sample[2:]), axis=0)  # (12,)
            self.output.append(embedded_sample)

        self.output = Variable(np.array(self.output))
        return self.output

    def backward(self):
        # Grab only the gradient for category embedding
        out_grad = self.output.grad[:, 1:4]  # (B, 3)
        # Slice off the category feature from data
        sex_category = np.array(self.x.value[:, 1], dtype=int)

        sex_one_hot = np.zeros((len(sex_category), self.num_embeddings))
        sex_one_hot[np.arange(len(sex_one_hot)), sex_category] = 1.0

        self.emb_m.grad = np.matmul(
            np.expand_dims(sex_one_hot, axis=2),
            np.expand_dims(out_grad, axis=1)
        ) # (B, 2, 3)


class LossMAE:
    def __init__(self):
        self.y: Variable = None
        self.y_prim: Variable = None

    def forward(self, y: Variable, y_prim: Variable) -> float:
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad = (self.y_prim.value - self.y.value) / np.abs(self.y.value - self.y_prim.value)


class LossMSE:
    def __init__(self):
        self.y: Variable = None
        self.y_prim: Variable = None

    def forward(self, y: Variable, y_prim: Variable) -> float:
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.power((y.value - y_prim.value), 2))
        return loss

    def backward(self):
        self.y_prim.grad = -2 * (self.y.value - self.y_prim.value)


class Model:
    def __init__(self):
        self.layers = [
            LayerEmbedding(num_embeddings=2, embedding_dim=3),
            LayerLinear(in_features=12, out_features=8),
            LayerReLU(),
            LayerLinear(in_features=8, out_features=4),
            LayerReLU(),
            LayerLinear(in_features=4, out_features=1)
        ]

    def forward(self, x: Variable) -> Variable:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self): # List[Variables]
        variables = []
        for layer in self.layers:
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)
            if isinstance(layer, LayerEmbedding):
                variables.append(layer.emb_m)
        return variables


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters  # List[Variable]
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate


LEARNING_RATE = 5e-5
BATCH_SIZE = 8

model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    LEARNING_RATE
)
loss_fn = LossMAE()

np.random.seed(0)
# shuffle
idxes_rand = np.random.permutation(len(X_std))
X = X_std[idxes_rand]
Y = Y_std[idxes_rand]

idx_split = int(len(X) * 0.8)  # 80% for training and 20% for testing
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])

np.random.seed(int(time.time()))

losses_train = []
losses_test = []
nrmse_plot_train = []
nrmse_plot_test = []
y_max_train = np.max(dataset_train[1])
y_min_train = np.min(dataset_train[1])
y_max_test = np.max(dataset_test[1])
y_min_test = np.min(dataset_test[1])

for epoch in range(1, 101):
    for dataset in [dataset_train, dataset_test]:
        X, Y = dataset
        losses = []
        nrmses = []
        for idx in range(0, len(X)-BATCH_SIZE, BATCH_SIZE):
            x = X[idx:idx+BATCH_SIZE]
            y = Y[idx:idx+BATCH_SIZE]

            y_prim = model.forward(Variable(x))
            loss = loss_fn.forward(Variable(y), y_prim)

            losses.append(loss)

            # nrmse
            scaler = 1 / (y_max_test - y_min_test)
            if dataset == dataset_train:
                scaler = 1 / (y_max_train - y_min_train)
            nrmse = scaler * np.sqrt(np.mean(np.power((y - y_prim.value), 2)))
            nrmses.append(nrmse)

            if dataset == dataset_train:  # Optimize only in training dataset
                loss_fn.backward()
                model.backward()
                optimizer.step()


        if dataset == dataset_train:
            nrmse_plot_train.append(np.mean(nrmses))
            losses_train.append(np.mean(losses))
        else:
            nrmse_plot_test.append(np.mean(nrmses))
            losses_test.append(np.mean(losses))

    print(f"Epoch: {epoch} "
          f"losses_train: {losses_train[-1]} "
          f"losses_test: {losses_test[-1]} "
          f"nrmse_train: {nrmse_plot_train[-1]} "
          f"nrmse_test: {nrmse_plot_test[-1]} "
          )

    if epoch % 20 == 0:
        plt.subplot(2, 1, 1)
        plt.title('loss l1')
        plt.plot(losses_train)
        plt.plot(losses_test)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train', 'Test'])

        plt.subplot(2, 1, 2)
        plt.title('nrmse')
        plt.plot(nrmse_plot_train)
        plt.plot(nrmse_plot_test)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train', 'Test'])

        plt.tight_layout()
        plt.show()