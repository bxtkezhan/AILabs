import theano.tensor as T


sigmoid = T.nnet.sigmoid
softplus = T.nnet.softplus
tanh = T.tanh
relu = T.nnet.relu
softmax = T.nnet.softmax

def binary(X):
    return (X > 0).astype(X.dtype)

def lrelu(X, alpha=0.01):
    return T.maximum(0, X) + alpha * T.minimum(0, X)
