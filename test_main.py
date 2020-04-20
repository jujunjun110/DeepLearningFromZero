import unittest
import main
import numpy as np


class TestMain(unittest.TestCase):
    def test_AND(self):
        cases = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
        ]

        for case in cases:
            with self.subTest(case=case):
                x1, x2, expected = case
                actual = main.AND(x1, x2)
                self.assertEqual(actual, expected)

    def test_OR(self):
        cases = [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]

        for case in cases:
            with self.subTest(case=case):
                x1, x2, expected = case
                actual = main.OR(x1, x2)
                self.assertEqual(actual, expected)

    def test_NAND(self):
        cases = [
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ]

        for case in cases:
            with self.subTest(case=case):
                x1, x2, expected = case
                actual = main.NAND(x1, x2)
                self.assertEqual(actual, expected)

    def test_XOR(self):
        cases = [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ]

        for case in cases:
            with self.subTest(case=case):
                x1, x2, expected = case
                actual = main.XOR(x1, x2)
                self.assertEqual(actual, expected)

    def test_softmax(self):
        case = np.array([0.3, 2.9, 4.0])
        actual = main.softmax(case)
        expected = [0.01821127, 0.24519181, 0.73659691]

        self.assertTrue(np.allclose(actual, expected))
        self.assertEqual(actual.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
