import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from PIL import Image


def main():
    exec_parallel()
    exec_parallel2()
    exec_sequentially()


def exec_parallel():
    # 100画像ずつ並列
    x, t = get_data()
    network = init_network()
    batch_size = 100
    accuracy_count = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i: i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_count += np.sum(p == t[i:i+batch_size])

    print(f"Accuracy: {accuracy_count / len(x)}")


def exec_parallel2():
    # 10000画像並列
    x, t = get_data()
    network = init_network()
    accuracy_count = 0

    y = predict(network, x)
    p = np.argmax(y, axis=1)
    accuracy_count += np.sum(p == t)

    print(f"Accuracy: {accuracy_count / len(x)}")


def exec_sequentially():
    x, t = get_data()
    network = init_network()
    accuracy_count = 0

    for img, label in zip(x, t):
        y = predict(network, img)
        p = np.argmax(y)
        if p == label:
            accuracy_count += 1

    print(f"Accuracy: {accuracy_count / len(x)}")


def get_data():
    sys.path.append(os.pardir)
    from dataset.mnist import load_mnist
    # (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    (_, _), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"],
    b1, b2, b3 = network["b1"], network["b2"], network["b3"],

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    return softmax(a3)


def img_show(img, mode="L"):
    pil_img = Image.fromarray(np.uint8(img), mode=mode)
    pil_img.show()


def exec_network():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


# def init_network():
#     network = {}

#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['b2'] = np.array([0.1, 0.2])
#     network['b3'] = np.array([0.1, 0.2])

#     return network


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
