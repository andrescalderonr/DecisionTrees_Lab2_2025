import unittest
import numpy as np
from src.metrics import Metric

class TestPrecision(unittest.TestCase):
    """
    Unit tests for the Precision metric.
    Precision measures the proportion of predicted positive instances that are actually positive.

    Precision = True Positives / (True Positives + False Positives)
    """

    def setUp(self):
        """
        Initializes the Precision metric before each test.
        """
        self.precision = Metric.use("precision")

    def test_perfect_precision(self):
        """
        All predicted positives are true positives.
        true_positive = 3, false_positive = 0 => precision = 3 / (3 + 0) = 1.0.
        """
        y = np.array([1, 0, 1, 1, 0])
        yp = np.array([1, 0, 1, 1, 0])
        result = self.precision.value(y, yp)
        expected = 1.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_partial_precision(self):
        """
        Some predicted positives are incorrect.
        true_positive = 2, false_positive = 1 => precision = 2 / (2 + 1) = 0.666666...
        """
        y = np.array([1, 0, 1, 0])
        yp = np.array([1, 1, 1, 0])
        result = self.precision.value(y, yp)
        expected = 2.0 / 3.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_zero_precision(self):
        """
        No predicted positive is actually positive.
        true_positive = 0, false_positive > 0 => precision = 0.0.
        """
        y = np.array([0, 0, 0])
        yp = np.array([1, 1, 1])
        result = self.precision.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_no_predicted_positives(self):
        """
        Edge case where the model predicts no positives.
        true_positive = 0, false_positive = 0 => implementation adds epsilon to denominator,
        expected precision should be 0.0 because there are no positive predictions.
        """
        y = np.array([1, 0, 1, 0])
        yp = np.array([0, 0, 0, 0])
        result = self.precision.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_shape_mismatch(self):
        """
        The method should raise ValueError when shapes of y and yp differ.
        """
        y = np.array([1, 0])
        yp = np.array([1, 0, 1])
        with self.assertRaises(ValueError):
            self.precision.value(y, yp)

if __name__ == '__main__':
    unittest.main()
