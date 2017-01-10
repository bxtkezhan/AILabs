import numpy as np

np.random.seed(1)

# Softmax function.
def softmax(X):
    Ej = np.exp(X)
    Ek = Ej.sum(axis=1).reshape(X.shape[0], 1)
    return Ej / Ek

batchsize, col = 50, 10
# Inputs.
X = 2 * np.random.random((batchsize, col)) - 1

# Outputs.
W_n = 2 * np.random.random((batchsize, col)) - 1
b_n = np.random.random((col,))
Z_n = softmax(W_n * X + b_n)
Y = (Z_n == Z_n.max(axis=1).reshape(batchsize, 1)).astype(float)

# Weights, bias
W = 2 * np.random.random((batchsize, col)) - 1
b = 0.1 * np.ones((col,))

# Trainling
train_times = 1000
for time in range(train_times):
    # Layer
    A = W * X + b
    _Y = Z = softmax(A)

    cost = _Y - Y # cost = -log(_Y)
    print('{} Error: {}'.format(time, np.mean(np.abs(cost))))

    # Calc deltas
    delta_A = cost
    delta_b = delta_A.sum(axis=0)
    delta_W = delta_A * X

    # Apply deltas
    rate = 0.1
    W -= rate * delta_W
    b -= rate * delta_b
else:
    print(Y == np.around(_Y))
