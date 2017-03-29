from variable import Variable
from layers import Dense, Conv2d, Pool2d, Dropout
from functions import activations
from functions import losses
from model import Model
from optimizers import SGD

import theano.tensor as T
import numpy as np
import os, gzip, pickle

np.random.seed(1)
floatX = T.config.floatX

# Load and deal datasets
if not os.path.exists('./mnist.pkl.gz'):
    raise IOError('No such mnist.pkl.gz')
with gzip.open('./mnist.pkl.gz') as f:
    train_set, test_set = pickle.load(f)

train_X = train_set[0].reshape(-1, 1, 28, 28).astype(floatX)
train_Y = np.zeros((train_set[1].shape[0], 10), dtype=floatX)
for i in range(train_Y.shape[0]):
    train_Y[i, train_set[1][i]] = 1

test_X = test_set[0].reshape(-1, 1, 28, 28).astype(floatX)

Var = Variable()
X = Var('tensor4')
Y = Var('matrix')

H, params1 = Conv2d(10, 1, (5, 5), W_scale=0.06)(X)
H = Pool2d((2, 2), activation=activations.relu)(H)
H, params2 = Conv2d(20, 10, (5, 5), W_scale=0.03)(H)
H = Pool2d((2, 2), activation=activations.relu)(H)
H, params3 = Dense(128, 20 * 4**2, W_scale=0.1, activation=activations.relu)(H.flatten(2))
H = Dropout(size=(128, ), rate=0.25, seed=1)(H)
H, params4 = Dense(10, 128, W_scale=0.1, activation=activations.softmax)(H)

_Y = H
cost = losses.log_likelihood(Y, _Y).sum() / 128

model = Model(
        inputs=[X, Y], params=params1+params2+params3+params4, cost=cost,
        other_functions={'predict':{'inputs':[X], 'outputs':_Y.argmax(axis=1)}})

model.compile(SGD(lr=0.1))
model.train([train_X, train_Y], epochs_num=2, batch_size=128)
print((model.other_functions['predict'](test_X) == test_set[1]).mean())
