from util import cross_entropy_error, softmax, numerical_gradient, sigmoid
import numpy as np


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weith_init_std=0.01):
        self.params = {}
        self.params['W1'] = weith_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weith_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2

        return softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

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
