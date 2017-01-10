import numpy as np

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

# Inputs.
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Outputs.
y = np.array([[0],
              [1],
              [1],
              [0]])

# Weights and bias.
W1 = 2 * np.random.random((3, 4)) - 1
b1 = 0.1 * np.ones((4,))
W2 = 2 * np.random.random((4, 1)) - 1
b2 = 0.1 * np.ones((1,))

# Training
train_times = 600
for time in range(train_times):
    # Layer1
    A1 = np.dot(X, W1) + b1
    Z1 = noline(A1)

    # Layer2
    A2 = np.dot(Z1, W2) + b2
    _y = Z2 = noline(A2)

    cost = (_y - y) # cost = (y - _y)**2 / 2
    print('{} Error: {}'.format(time, np.mean(np.abs(cost))))

    # Calc deltas
    delta_A2 = cost * noline(Z2, derive=True)
    delta_b2 = delta_A2.sum(axis=0)
    delta_W2 = np.dot(Z1.T, delta_A2)
    delta_A1 = np.dot(delta_A2, W2.T) * noline(Z1, derive=True)
    delta_b1 = delta_A1.sum(axis=0)
    delta_W1 = np.dot(X.T, delta_A1)

    # Apply deltas
    rate = 0.1
    W2 -= rate * delta_W2
    b2 -= rate * delta_b2
    W1 -= rate * delta_W1
    b1 -= rate * delta_b1
else:
    print(_y)
