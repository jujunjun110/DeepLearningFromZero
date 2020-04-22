import numpy as np
import sys
import os
from util import softmax, cross_entropy_error, numerical_grad


class SimpleNet:
    def __init__(self):
        self.w = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.w)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
