import numpy as np
from sklearn import datasets

np.random.seed(1)


# Nonlinear functions
def sigmoid(X, derive=False):
    if derive:
        return X * (1 - X)
    return 1 / (1 + np.exp(-X))


def relu(X, derive=False, alpha=0.1):
    if derive:
        return (X > 0).astype(float) * alpha
    return np.maximum(X, 0) * alpha


noline = relu

# Load iris dataset
iris = datasets.load_iris()

# Inputs.
X = iris.data
X -= X.min()
X /= X.max()

# Outputs.
Y = np.zeros((150, 3))
for i in range(150):
    Y[i, [iris.target[i]]] = 1

# Weights and bias.
W1 = np.random.randn(4, 5) / 150**0.5
b1 = 0.1 * np.ones((5, ))
W2 = np.random.randn(5, 3) / 150**0.5
b2 = 0.1 * np.ones((3, ))

# Training
train_times = 1800
for time in range(train_times):
    # Layer1
    A1 = np.dot(X, W1) + b1
    Z1 = noline(A1)

    # Layer2
    A2 = np.dot(Z1, W2) + b2
    _Y = Z2 = noline(A2)

    cost = (_Y - Y)  # cost = (Y - _Y)**2 / 2
    print('{} Error: {}'.format(time, np.mean(np.abs(cost))))

    # Calc deltas
    delta_A2 = cost * noline(Z2, derive=True)
    delta_b2 = delta_A2.sum(axis=0)
    delta_W2 = np.dot(Z1.T, delta_A2) + 0.01 / 150 * W2
    delta_A1 = np.dot(delta_A2, W2.T) * noline(Z1, derive=True)
    delta_b1 = delta_A1.sum(axis=0)
    delta_W1 = np.dot(X.T, delta_A1) + 0.01 / 150 * W1

    # Apply deltas
    rate = 0.1
    W2 -= rate * delta_W2
    b2 -= rate * delta_b2
    W1 -= rate * delta_W1
    b1 -= rate * delta_b1
else:
    print(np.mean((np.around(_Y) == Y).astype(int)))
