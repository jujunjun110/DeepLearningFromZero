import unittest
import main


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


if __name__ == "__main__":
    unittest.main()
