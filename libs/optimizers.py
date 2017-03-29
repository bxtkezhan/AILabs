import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.0, decay=0.0, nesterov=False,
                 maximum=None, minimum=None):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
        self.idx = None

        self.maximum = maximum or lr
        self.minimum = minimum or 0.0
        if self.maximum <= self.minimum:
            raise TypeError('maximum 必须大于 minimum')

    def __call__(self, sample_size, batch_size, status='begin'):
        if status == 'begin':
            self.idx = np.arange(sample_size)
        elif status == 'time':
            self.idx = np.random.permutation(self.idx)
            self.lr = self.lr - self.decay
            self.lr = min(self.lr, self.maximum)
            self.lr = max(self.lr, self.minimum)
        elif status == 'epoch':
            pass

        return self.idx, self.lr
