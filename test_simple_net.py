import unittest
from simple_net import SimpleNet
import numpy as np


class TestSimpleNet(unittest.TestCase):
    def test_assert_true(self):
        net = SimpleNet()
        print(net.w)

        x = np.array([0.6, 0.9])
        p = net.predict(x)
        print(p)

        print(np.argmax(p))

        t = np.array([0, 0, 1])

        loss = net.loss(x, t)
        print(loss)

        self.assertTrue(True)


if __name__ == "__main__":
    pass
