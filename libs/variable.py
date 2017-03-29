import theano
import theano.tensor as T
import numpy as np


HASH_NP2T = {
    np.int8: 'b', np.int16: 'w', np.int32: 'i', np.int64: 'l',
    np.float32: 'f', np.float64: 'd', np.complex64: 'c',
}

class Variable:
    def __init__(self, mode='symbol', dtype=T.config.floatX):
        if mode not in ['symbol', 'shared']:
            raise TypeError('mode 必须是 symbol、shared 其中之一')
        self.mode = mode
        if isinstance(dtype, str):
            self.dtype = np.typeDict[dtype]
        else:
            self.dtype = dtype

    def __call__(self, value=None, name=None, borrow=False):
        if self.mode == 'symbol':
            return eval("T.{}{}(name='{}')".format(HASH_NP2T[self.dtype], value, name))
        elif self.mode == 'shared':
            return theano.shared(np.asarray(value, self.dtype), name=name, borrow=borrow)

Random = T.shared_randomstreams.RandomStreams
