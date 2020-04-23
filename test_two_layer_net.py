import unittest
import numpy as np
from IPython import embed

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


class TestTwoLayerNet(unittest.TestCase):
    def test_assert_true(self):
        (x_train, t_train), (_x_test, _t_test) = load_mnist(normalize=True, one_hot_label=True)

        train_loss_list = []

        iters_num = 100
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = 1

        network = TwoLayerNet(784, 50, 10)

        for i in range(iters_num):
            print(i)
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grad = network.numerical_gradient(x_batch, t_batch)

            for key in ("W1", "b1", "W2", "b2"):
                network.params[key] -= learning_rate * grad[key]

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            print(loss)

        print(train_loss_list)


if __name__ == "__main__":
    unittest.main()
