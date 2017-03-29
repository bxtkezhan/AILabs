import theano.tensor as T

def squared_error(Y_true, Y_pred):
    return T.square(Y_true - Y_pred)

def absolute_error(Y_true, Y_pred):
    return abs(Y_true, Y_pred)

def binary_crossentropy(Y_true, Y_pred):
    return -Y_true * T.log(Y_pred) - (1 - Y_true) * T.log(1 - Y_pred)

def log_likelihood(Y_true, Y_pred):
    return -Y_true * T.log(Y_pred)
