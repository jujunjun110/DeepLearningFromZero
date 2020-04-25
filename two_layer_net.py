from collections import OrderedDict
import numpy as np

from util import cross_entropy_error, softmax, numerical_gradient, sigmoid
from layers import Affine, Relu, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weith_init_std=0.01):
        self.params = {}
        self.params['W1'] = weith_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weith_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda w: self.loss(x, t)

        grads = {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'b2': numerical_gradient(loss_W, self.params['b2']),
        }

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values()).reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {
            'W1': self.layers['Affine1'].dW,
            'W2': self.layers['Affine1'].dW,
            'b1': self.layers['Affine2'].db,
            'b2': self.layers['Affine2'].db,
        }

        return grads
