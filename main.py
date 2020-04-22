import sys
import os
import pickle
import datetime

import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from PIL import Image

from util import softmax, cross_entropy_error, numerical_grad


def main():
    calc_time_endpoint()


def calc_time_endpoint():
    for f in ["exec_parallel", "exec_parallel2", "exec_sequentially"]:
        calc_time(f)


def calc_time(func_name):
    start = datetime.datetime.now()
    globals()[func_name]()
    end = datetime.datetime.now()
    print(f"Function: {func_name}, Time: {end - start}")


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

    y = predict(network, x)
    p = np.argmax(y, axis=1)
    accuracy_count = np.sum(p == t)

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


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def mean_squared_error(y, t):
    # 二乗和誤差
    return 0.5 * np.sum((y - t) ** 2)


def numerical_diff(f, x):
    h = 1e-4
    return(f(x + h) - f(x - h)) / (2 * h)


def gradient_descent(f, init_x, lr=0.01, step_num=10000):
    # lr = Learning Rate
    x = init_x

    for _ in range(step_num):
        grad = numerical_grad(f, x)
        x -= lr * grad

    return x


if __name__ == "__main__":
    main()
