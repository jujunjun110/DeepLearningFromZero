import unittest
import main
import numpy as np


class TestMain(unittest.TestCase):
    def test_softmax(self):
        case = np.array([0.3, 2.9, 4.0])
        actual = main.softmax(case)
        expected = [0.01821127, 0.24519181, 0.73659691]

        self.assertTrue(np.allclose(actual, expected))
        self.assertEqual(actual.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
