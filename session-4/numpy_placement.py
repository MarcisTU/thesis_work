import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DatasetPlacements():
    def __init__(self):
        self.data = pd.read_csv('archive/Placement_Data_Full_Class.csv', sep=',')
        self.features = None
        self.preprocess()
        self.standardize()

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
        Y = self.data[:, 5]
        X = np.concatenate((self.data[:, :5], self.data[:, 6:]), 1)
        self.data = (X, Y)

    def standardize(self):
        x, y = self.data

        # Separate categorical (don't need to standardize them), only standardize continuous
        contin_col = [1, 3, 5, 8, 10, 12]
        for col in contin_col:
            # Standardize
            x_mean = np.mean(x[:, col])
            x_std = np.std(x[:, col])
            x[:, col] = (x[:, col] - x_mean) / x_std

        self.data = (x, y)

    def load_data(self):
        return self.data


class Variable():
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class LayerLinear():
    def __init__(self, in_features, out_features):
        self.W = Variable(
            value=np.random.random((out_features, in_features))
        )
        self.b = Variable(
            value=np.zeros(( out_features, ))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x):
        self.x: Variable = x
        self.output = Variable(
            np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad = 1 * self.output.grad

        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.value, axis=1))

        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


class LayerReLU():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable((x.value >= 0) * x.value)
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0) * self.output.grad


class LayerSoftmax():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        np_x = np.array(x.value)
        np_x -= np.max(np_x, axis=1, keepdims=True) #numerical stability hack
        self.output = Variable(np.exp(np_x) + 1e-8 / np.sum(np.exp(np_x), axis=1, keepdims=True))
        return self.output

    def backward(self):
        J = np.zeros((BATCH_SIZE, 3, 3))
        a = self.output.value

        for i in range(3):
            for j in range(3):
                if i == j:
                    J[:, i, j] = a[:, i] * (1 - a[:, j])  # identity part
                else:
                    J[:, i, j] = -a[:, i] * a[:, j]

        self.x.grad = np.matmul(J, np.expand_dims(self.output.grad, axis=2)).squeeze()


class LossCrossEntropy():
    def __init__(self):
        self.y_prim = None

    def forward(self, y, y_prim):
        self.y_prim = y_prim
        self.y = y
        return -np.sum(y.value * np.log(y_prim.value + 1e-8))

    def backward(self):
        self.y_prim.grad = -self.y.value / (self.y_prim.value + 1e-8)


class LayerEmbedding:
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
        self.x: Variable = None
        self.num_embeddings_2d = 2
        self.embedding_dim_2d = 3
        self.num_embeddings_3d = 3
        self.embedding_dim_3d = 5
        self.categorical_columns_2d = [0, 2, 4, 7, 9, 11]
        self.categorical_columns_3d = [12]
        self.offset = 0
        self.fix_offset()
        """
        len(self.categorical_columns_2d) -> for every categorical data feature we need separate embeddings matrix
        """
        self.emb_m_2d = Variable(np.zeros((len(self.categorical_columns_2d), self.num_embeddings_2d, self.embedding_dim_2d)))
        self.emb_m_3d = Variable(np.zeros((len(self.categorical_columns_3d), self.num_embeddings_3d, self.embedding_dim_3d)))
        self.output: Variable = None

    def fix_offset(self):
        # calculate the offset for 6th column after embedding 2d categorical
        for n in self.categorical_columns_2d:
            if n < 6:
                self.offset += 2 # Every embedding for 2 categorical values is adding 2 new columns

    def forward(self, x: Variable):
        self.x = x
        self.output = []
        """
        Embed each sample in Batch replacing categorical data with embeddings vector.
        """
        for index, sample in enumerate(x.value):
            # Embedding for 2 categories
            embedded_sample = self.embed(sample, self.categorical_columns_2d, self.num_embeddings_2d, self.embedding_dim_2d, self.emb_m_2d)
            # Embedding for 3 categories
            embedded_sample = self.embed(embedded_sample, self.categorical_columns_3d, self.num_embeddings_3d, self.embedding_dim_3d, self.emb_m_3d)
            self.output.append(embedded_sample)

        self.output = Variable(np.array(self.output))
        return self.output

    """
    Helper function, that "embeds" replaces categorical value with embeddings vector, e.g: 1 with vector [0.0, 0.0, 0.0]
    Vector size depends on number of categories.
    """
    def embed(self, embed_sample, categorical_columns, num_embeddings, embedding_dim, emb_m):
        offset_index = 0
        embedded_sample = embed_sample
        for index, cat in enumerate(categorical_columns):  # for every column that has categorical data (here predefined)
            category = int(embedded_sample[cat + offset_index])
            category_one_hot = np.zeros(num_embeddings, )
            category_one_hot[category] = 1.0
            category_one_hot = np.expand_dims(category_one_hot, axis=0)
            category_emb = np.squeeze(category_one_hot @ emb_m.value[index])

            embedded_sample = np.concatenate((embedded_sample[:cat + offset_index], category_emb, embedded_sample[cat + offset_index + 1:]), 0)
            offset_index += embedding_dim - 1
        return embedded_sample

    def backward(self):
        self.emb_m_2d.grad = self.get_embedding_grad(self.categorical_columns_2d, self.embedding_dim_2d, self.num_embeddings_2d, offset=0)
        self.emb_m_3d.grad = self.get_embedding_grad(self.categorical_columns_3d, self.embedding_dim_3d, self.num_embeddings_3d, offset=self.offset)

    def get_embedding_grad(self, categorical_col, embedding_dim, num_embeddings, offset):
        embed_m = np.zeros((len(self.output.grad), len(categorical_col), num_embeddings, embedding_dim))

        # Calculate grad for categorical data with 2 categories
        offset_index = 0
        for index, cat in enumerate(categorical_col):
            # Grab only the gradient for category embedding
            out_grad = self.output.grad[:, cat + offset_index:cat + offset_index + embedding_dim]
            # Slice off the category feature from data before embedding (so we need to offset the column 12 back to 6)
            category = np.array(self.x.value[:, cat-offset], dtype=int)

            category_one_hot = np.zeros((len(category), num_embeddings))
            category_one_hot[np.arange(len(category_one_hot)), category] = 1

            emb_m = np.matmul(
                np.expand_dims(category_one_hot, axis=2),
                np.expand_dims(out_grad, axis=1)
            )  # (16, self.num_embeddings, self.embedding_dim)
            # Insert calculated embedding matrix to its corresponding position
            embed_m[:, index] = emb_m

            offset_index += embedding_dim - 1

        return embed_m


class Model:
    def __init__(self):
        self.layers = [
            LayerEmbedding(),
            LayerLinear(in_features=29, out_features=20),
            LayerReLU(),
            LayerLinear(in_features=20, out_features=15),
            LayerReLU(),
            LayerLinear(in_features=15, out_features=10),
            LayerReLU(),
            LayerLinear(in_features=10, out_features=3),
            LayerSoftmax()
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)
            if isinstance(layer, LayerEmbedding):
                variables.append(layer.emb_m_2d)
                variables.append(layer.emb_m_3d)
        return variables


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate


BATCH_SIZE = 10
LEARNING_RATE=5e-4

model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossCrossEntropy()

dataset = DatasetPlacements()
X, Y = dataset.load_data()
Y = Y.astype(int)


# shuffle
np.random.seed(0)
idxes_rand = np.random.permutation(len(Y))
X = X[idxes_rand]
Y = Y[idxes_rand]

# Slice off 5 samples to make batch of 10 possible to split
X = X[:-5]
Y = Y[:-5]

Y_idxs = Y
Y = np.zeros((len(Y), 3))
Y[np.arange(len(Y)), Y_idxs] = 1.0

idx_split = int(len(X) * 0.8572)
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])

np.random.seed(int(time.time()))

train_losses_plot = []
test_losses_plot = []
train_acc_plot = []
test_acc_plot = []

for epoch in range(20):
    for dataset in [dataset_train, dataset_test]:
        X, Y = dataset
        losses = []
        accuracies = []
        for idx in range(0, len(X)-BATCH_SIZE, BATCH_SIZE):
            x = X[idx:idx+BATCH_SIZE]
            y = Y[idx:idx+BATCH_SIZE]

            y_prim = model.forward(Variable(value=x))
            loss = loss_fn.forward(Variable(value=y), y_prim)
            # Accuracy
            pred_max = np.argmax(y_prim.value, axis=1)
            actual_max = np.argmax(y, axis=1)
            true_val = (pred_max == actual_max) * 1
            accuracy = np.sum(true_val) / len(true_val)

            losses.append(loss)
            accuracies.append(accuracy)

            if len(dataset[0]) == len(dataset_train[0]):
                loss_fn.backward()
                model.backward()
                optimizer.step()

        if len(dataset[0]) == len(dataset_train[0]):
            train_losses_plot.append(np.mean(losses))
            train_acc_plot.append(np.mean(accuracies))
        else:
            test_losses_plot.append(np.mean(losses))
            test_acc_plot.append(np.mean(accuracies))

    print(f'Epoch: {epoch} '
          f'train_loss: {train_losses_plot[-1]} '
          f'train_accuracy: {train_acc_plot[-1]} '
          f'test_loss: {test_losses_plot[-1]} '
          f'test_accuracy: {test_acc_plot[-1]}')

plt.subplot(2, 1, 1)
plt.title('loss Cross Entropy')
plt.plot(train_losses_plot)
plt.plot(test_losses_plot)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train', 'Test'])

plt.subplot(2, 1, 2)
plt.title('accuracy')
plt.plot(train_acc_plot)
plt.plot(test_acc_plot)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['Train', 'Test'])

plt.tight_layout()
plt.show()
