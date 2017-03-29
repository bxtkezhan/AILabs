import theano.tensor as T
from variable import Variable, Random
import numpy as np
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d


class Layer:
    def __init__(self, W_data, b_data, dtype=T.config.floatX, activation=None):

        VarShared = Variable(mode='shared', dtype=dtype)
        self.W = VarShared(value=W_data, name='W', borrow=True)
        self.b = VarShared(value=b_data, name='b', borrow=True)
        self.params = [self.W, self.b]
        self.activation = activation or (lambda X: X)

class Dense(Layer):
    def __init__(self, output_dim, input_dim, dtype=T.config.floatX,
                 initial_W=None, initial_b=0.1, W_scale=1, activation=None):

        W_data, b_data = initial_W, initial_b * np.ones((output_dim, ))
        if initial_W is None:
            W_data = np.random.uniform(
                low=-W_scale, high=W_scale,
                size=(input_dim, output_dim))

        Layer.__init__(self, W_data, b_data, dtype=dtype, activation=activation)

    def __call__(self, X):
        A = T.dot(X, self.W) + self.b
        return self.activation(A), self.params

class Conv2d(Layer):
    def __init__(self, output_dim, input_dim, k_scale,
                 border_mode='valid', subsample=(1,1), dtype=T.config.floatX,
                 initial_W=None, initial_b=0.1, W_scale=1, activation=None):

        W_data, b_data = initial_W, initial_b * np.ones((output_dim, ))
        if initial_W is None:
            W_data = np.random.uniform(
                low=-W_scale, high=W_scale,
                size=(output_dim, input_dim, k_scale[0], k_scale[1]))

        Layer.__init__(self, W_data, b_data, dtype=dtype, activation=activation)

        self.border_mode = border_mode
        self.subsample = subsample

    def __call__(self, X):
        A1 = conv2d(X, self.W, border_mode=self.border_mode,
                    subsample=self.subsample)
        A2 = A1 + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.activation(A2), self.params

class Pool2d(Layer):
    def __init__(self, sample_scale, mode='max',
                 ignore_border=True, activation=None):

        self.sample_scale = sample_scale
        self.mode = mode
        self.ignore_border = ignore_border
        self.activation = activation or (lambda X: X)

    def __call__(self, X):
        A = pool_2d(X, self.sample_scale, mode=self.mode,
                    ignore_border=self.ignore_border)
        return self.activation(A)

class Dropout(Layer):
    def __init__(self, size, rate=0.5, seed=None):
        self.rate = rate
        self.srng = Random(seed=seed)
        self.mask = self.srng.binomial(size=size, p=rate, dtype=T.config.floatX)

    def __call__(self, X):
        return self.mask * X
