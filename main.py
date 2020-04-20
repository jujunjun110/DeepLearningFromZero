import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import IPython


def main():
    sys.path.append(os.pardir)
    from dataset.mnist import load_mnist

    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=True)
    IPython.embed()

    # network = init_network()
    # x = np.array([1.0, 0.5])
    # y = forward(network, x)
    # print(y)


def init_network():
    network = {}

    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['b2'] = np.array([0.1, 0.2])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3

    y = identity_function(a3)
    return y


def identity_function(x):
    return x


def AND(x1, x2):
    w1, w2 = 0.5, 0.5
    b = -0.7

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(w * x) + b

    if tmp > 0:
        return 1

    return 0


def NAND(x1, x2):
    w1, w2 = -0.5, -0.5
    b = 0.7

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(w * x) + b

    if tmp > 0:
        return 1

    return 0


def OR(x1, x2):
    w1, w2 = 0.5, 0.5
    b = -0.2

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(w * x) + b

    if tmp > 0:
        return 1

    return 0


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


if __name__ == "__main__":
    main()
