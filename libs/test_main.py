from variable import Variable
from layers import Dense
from functions import activations
from functions import losses
from model import Model
from optimizers import SGD

from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
train_X = iris.data.astype('float32')
train_Y = np.zeros((train_X.shape[0], 3), dtype='float32')
for i in range(train_Y.shape[0]):
    train_Y[i, iris.target[i]] = 1

Var = Variable()
X = Var('matrix')
Y = Var('matrix')

H1, params1 = Dense(5, train_X.shape[1], activation=activations.sigmoid)(X)
H2, params2 = Dense(5, 5, activation=activations.sigmoid)(H1)
H3, params3 = Dense(train_Y.shape[1], 5, activation=activations.sigmoid)(H2)

_Y = H3
cost = losses.binary_crossentropy(Y, _Y).mean()

model = Model(
        inputs=[X, Y], params=params1+params2+params3, cost=cost,
        other_functions={'predict':{'inputs':[X], 'outputs':_Y.argmax(axis=1)}})

model.compile(SGD(lr=0.8))
model.train([train_X, train_Y], epochs_num=10000, batch_size=50)
print(model.other_functions['predict'](train_X))
