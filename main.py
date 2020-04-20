import numpy as np
import matplotlib.pyplot as plt


def main():
    pass


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
    tmp1 = NAND(x1, x2)
    tmp2 = OR(x1, x2)
    return AND(tmp1, tmp2)


if __name__ == "__main__":
    main()
