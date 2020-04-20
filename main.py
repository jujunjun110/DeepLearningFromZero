import numpy as np
import matplotlib.pyplot as plt


def plotsample():
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, label="cos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin&cos")
    plt.legend()
    plt.show()


def main():
    pass


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    if w1 * x1 + w2 * x2 > theta:
        return 1

    return 0


if __name__ == "__main__":
    main()
