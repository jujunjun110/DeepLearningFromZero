import unittest
import bool
import numpy as np


class TestBool(unittest.TestCase):
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
                actual = bool.AND(x1, x2)
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
                actual = bool.OR(x1, x2)
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
                actual = bool.NAND(x1, x2)
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
                actual = bool.XOR(x1, x2)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
