import numpy as np
from sklearn import datasets

np.random.seed(1)


# Nonlinear functions
def sigmoid(X, derive=False):
    if derive:
        return X * (1 - X)
    return 1 / (1 + np.exp(-X))


def relu(X, derive=False):
    if derive:
        return (X > 0).astype(float)
    return np.maximum(X, 0)


noline = relu

# Load iris dataset
iris = datasets.load_iris()

# Inputs.
X = iris.data
X -= X.min()
X /= X.max()

# Outputs.
Y = X

# Weights and bias.
W1 = np.random.randn(4, 3) / 150**0.5
b1 = 0.1 * np.ones((3, ))
W2 = np.random.randn(3, 2) / 150**0.5
b2 = 0.1 * np.ones((2, ))
W3 = np.random.randn(2, 4) / 150**0.5
b3 = 0.1 * np.ones((4, ))

# Training
train_times = 1000
for time in range(train_times):
    # Layer1
    A1 = np.dot(X, W1) + b1
    Z1 = noline(A1)

    # Layer2
    A2 = np.dot(Z1, W2) + b2
    Z2 = noline(A2)

    # Layer3
    A3 = np.dot(Z2, W3) + b3
    _Y = Z3 = noline(A3)

    cost = (_Y - Y)  # cost = (Y - _Y)**2 / 2
    print('{} Error: {}'.format(time, np.mean(np.abs(cost))))

    # Calc deltas
    delta_A3 = cost * noline(Z3, derive=True)
    delta_b3 = delta_A3.sum(axis=0)
    delta_W3 = np.dot(Z2.T, delta_A3)

    delta_A2 = np.dot(delta_A3, W3.T) * noline(Z2, derive=True)
    delta_b2 = delta_A2.sum(axis=0)
    delta_W2 = np.dot(Z1.T, delta_A2)

    delta_A1 = np.dot(delta_A2, W2.T) * noline(Z1, derive=True)
    delta_b1 = delta_A1.sum(axis=0)
    delta_W1 = np.dot(X.T, delta_A1)

    # Apply deltas
    rate = 0.001
    W3 -= rate * delta_W3
    b3 -= rate * delta_b3
    W2 -= rate * delta_W2
    b2 -= rate * delta_b2
    W1 -= rate * delta_W1
    b1 -= rate * delta_b1
else:
    print(Z2, 1 - np.mean(np.abs(cost)))
