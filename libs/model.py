import theano
import theano.tensor as T
from variable import Variable


class Model:
    def __init__(self, inputs, params, cost, outputs=None,
                 other_functions={}, metrics=[]):

        if not isinstance(inputs, list):
            raise TypeError('inputs 必须为 list 类型')
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.cost = cost

        self.grads = T.grad(self.cost, self.params)

        Var = Variable('symbol', dtype=self.inputs[0].dtype)
        self.shared_lr = Var('scalar', name='lr')
        
        self.inputs.append(theano.In(self.shared_lr, value=0.01))
        self.updates = [(p, p - self.shared_lr * g)
                        for p, g in zip(self.params, self.grads)]

        self.other_functions = other_functions
        self.metrics = metrics

    def compile(self, optimizer=None):
        self.optimizer = optimizer

        train_function_outputs = [self.cost]
        if self.outputs is not None:
            train_function_outputs.append(self.outputs)
        self.trainFunction = theano.function(
            inputs=self.inputs, outputs=train_function_outputs,
            updates=self.updates)

        for function_name in self.other_functions.keys():
            self.other_functions[function_name] = theano.function(
                inputs=self.other_functions[function_name]['inputs'],
                outputs=self.other_functions[function_name]['outputs'])

    def trainOnBatch(self, inputs, lr=0.01):
        if not isinstance(inputs, list):
            raise TypeError('inputs 必须为 list 类型')
        return self.trainFunction(*(inputs + [lr]))

    def train(self, inputs, epochs_num=1, batch_size=100):
        if not isinstance(inputs, list):
            raise TypeError('inputs 必须为 list 类型')
        sample_size = len(inputs[0])
        times = sample_size // batch_size
        outputs = None
        idx, lr = self.optimizer(sample_size, batch_size, status='begin')
        for epoch in range(epochs_num):
            for time in range(times):
                batch_idx = idx[time * batch_size: (time+1) * batch_size]
                batch_inputs = [data[batch_idx] for data in inputs]
                outputs = self.trainOnBatch(batch_inputs, lr)
                idx, lr = self.optimizer(sample_size, batch_size, status='time')
                print(time, outputs[0])
            idx, lr = self.optimizer(sample_size, batch_size, status='epoch')
