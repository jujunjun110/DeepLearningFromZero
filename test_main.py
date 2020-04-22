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

    def test_numerical_grad(self):
        def function_2(x):
            return (x[0] ** 2 + x[1] ** 2)

        case = np.array([3., 4.])
        expected = np.array([6., 8.])
        actual = main.numerical_grad(function_2, case)
        self.assertTrue(np.allclose(actual, expected))

    def test_gradient_descent(self):
        def function_2(x):
            return (x[0] ** 2 + x[1] ** 2)

        case = np.array([-3., 4.])
        expected = np.array([0., 0.])

        actual = main.gradient_descent(function_2, case, 0.1)
        print(actual)
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
