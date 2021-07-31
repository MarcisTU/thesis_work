import numpy as np
import matplotlib.pyplot as plt

def linear(W, b, x):
    return W * x + b

def dW_linear(x):
    return x

def db_linear():
    return 1

def dx_linear(W):
    return W

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def da_sigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

def model(W, b, x):
    return sigmoid(linear(W, b, x)) * 20

def dW_model(W, b, x):
    return da_sigmoid(W * x + b) * dW_linear(W) * 20
def db_model(W, b, x):
    return da_sigmoid(W * x + b) * db_linear() * 20

def loss(y, y_prim):
    return np.mean(np.power((y - y_prim), 2))

def dW_loss(y, W, x, b):
    return np.mean(-2*dW_linear(x)*(y - (W * x + b)))

def db_loss(y, W, x, b):
    return np.mean(-2*db_linear()*(y - (W * x + b)))


X = np.array([1, 2, 3, 4, 5])
Y = np.array([0.7, 1.5, 4.5, 6.9, 9.5])

W = 0
b = 0
best_W = 0
best_b = 0
best_loss = np.inf
loss_history = []
Y_prim = np.zeros((4,))
dW_mse_loss = 0
db_mse_loss = 0

learning_rate = 0.0075
for epoch in range(50):
    # For every data point (Best results in this example ~1.2 loss score)
    for i in range(len(X)):
        dW_mse_loss += dW_loss(Y[i], W, X[i], b)
        db_mse_loss += db_loss(Y[i], W, X[i], b)

    # X and Y in batch
    # dW_mse_loss = dW_loss(Y, W, X, b)
    # db_mse_loss = db_loss(Y, W, X, b)

    W -= dW_mse_loss * learning_rate
    b -= db_mse_loss * learning_rate

    Y_prim = model(W, b, X)
    mse_loss = loss(Y, Y_prim)
    loss_history.append(mse_loss)

    print(f"Y_prim {Y_prim}")
    print(f"loss: {mse_loss}")

    # Save best bias and weight value obtained during training
    if  mse_loss < best_loss:
        best_loss = mse_loss
        best_W = W
        best_b = b

Y_prim = model(best_W, best_b, X)
print(f"Best loss: {best_loss}")

# Test model
test_y = model(best_W, best_b, 6)
print(f"Predicted price for 6 story house: ${np.round_(test_y * 1e5, 0)}")

# Plot results
plt.title("Regression model")
plt.plot(X, Y, 'o')
slope = np.polyfit(X, Y_prim, 1)
m = slope[0]
b = slope[1]
plt.plot(X, m*X + b)
plt.show()

plt.title("Loss function value")
plt.plot(loss_history, '-')
plt.show()