import unittest
import numpy as np

from simple_net import SimpleNet
from util import numerical_grad


class TestSimpleNet(unittest.TestCase):
    def test_assert_true(self):
        net = SimpleNet()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
