import unittest
import numpy as np
from src.metrics import Metric

class TestRecall(unittest.TestCase):
    """
    Unit tests for the Recall metric, used to evaluate classification models.
    Recall (also called sensitivity or true positive rate) measures the proportion
    of actual positive instances that were correctly identified.

    Recall = True Positives / (True Positives + False Negatives)
    """

    def setUp(self):
        """
        Initializes the Recall metric before each test.
        """
        self.recall = Metric.use("recall")

    def test_perfect_recall(self):
        """
        Tests the case where all actual positives are predicted positive.
        For perfect recall the expected result should be 1.0.

        True positives = 3, false negatives = 0 => recall = 3 / (3 + 0) = 1.0.
        """
        y = np.array([1, 0, 1, 1])
        yp = np.array([1, 0, 1, 1])
        result = self.recall.value(y, yp)
        expected = 1.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_partial_recall(self):
        """
        Tests the case where some actual positives are missed.
        Here there are 3 actual positives and 2 are predicted positive,
        so true positives = 2, false negatives = 1 => recall = 2 / (2 + 1) = 0.666666...
        """
        y = np.array([1, 0, 1, 1])
        yp = np.array([1, 0, 0, 1])
        result = self.recall.value(y, yp)
        expected = 2.0 / 3.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_zero_recall(self):
        """
        Tests the case where no actual positive is predicted positive.
        True positives = 0, false negatives > 0 => recall = 0.0.
        """
        y = np.array([1, 1, 1])
        yp = np.array([0, 0, 0])
        result = self.recall.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_no_actual_positives(self):
        """
        Tests the edge case where there are no actual positive instances in y.
        The implementation adds a small epsilon to avoid division by zero.
        The expected recall in this situation should be 0.0 because there are
        no true positives and no actual positives to detect.
        """
        y = np.array([0, 0, 0, 0])
        yp = np.array([0, 1, 0, 1])
        result = self.recall.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_shape_mismatch(self):
        """
        Tests that the `value` method raises a ValueError when the shapes of y and yp do not match.
        """
        y = np.array([1, 0])
        yp = np.array([1, 0, 1])
        with self.assertRaises(ValueError):
            self.recall.value(y, yp)

if __name__ == '__main__':
    unittest.main()
