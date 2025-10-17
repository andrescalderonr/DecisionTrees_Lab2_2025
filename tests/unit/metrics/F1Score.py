import unittest
import numpy as np
from src.metrics import Metric

class TestF1Score(unittest.TestCase):
    """
    Unit tests for the F1Score metric.
    F1 Score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)
    """

    def setUp(self):
        """
        Initializes the F1Score metric before each test.
        """
        self.f1 = Metric.use("f1score")

    def test_perfect_f1(self):
        """
        All predictions are correct so precision = 1.0 and recall = 1.0 => F1 = 1.0.
        true_positive = 3, false_positive = 0, false_negative = 0 => F1 = 1.0.
        """
        y = np.array([1, 0, 1, 1, 0])
        yp = np.array([1, 0, 1, 1, 0])
        result = self.f1.value(y, yp)
        expected = 1.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_partial_f1(self):
        """
        Mixed correctness producing intermediate precision and recall.
        Example:
        y = [1, 0, 1, 1]
        yp = [1, 1, 0, 1]
        true_positive = 2, false_positive = 1, false_negative = 1
        precision = 2 / (2 + 1) = 2/3
        recall = 2 / (2 + 1) = 2/3
        F1 = 2 * (2/3)*(2/3) / ((2/3)+(2/3)) = 2/3
        """
        y = np.array([1, 0, 1, 1])
        yp = np.array([1, 1, 0, 1])
        result = self.f1.value(y, yp)
        expected = 2.0 / 3.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_zero_f1_no_true_positives(self):
        """
        No true positives while there are positive predictions.
        true_positive = 0, false_positive > 0, false_negative > = 0 => F1 = 0.0.
        """
        y = np.array([0, 0, 0])
        yp = np.array([1, 1, 1])
        result = self.f1.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_f1_when_no_predicted_positives(self):
        """
        Model predicts no positives.
        true_positive = 0, false_positive = 0.
        precision = 0.0, recall depends on actual positives; F1 should be 0.0.
        """
        y = np.array([1, 0, 1, 0])
        yp = np.array([0, 0, 0, 0])
        result = self.f1.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_f1_when_no_actual_positives(self):
        """
        No actual positive instances.
        true_positive = 0, false_negative = 0.
        precision may be >0 if model predicts positives, but recall is 0 by definition here.
        Using the implementation's epsilon to avoid division by zero, F1 should evaluate to 0.0.
        """
        y = np.array([0, 0, 0, 0])
        yp = np.array([0, 1, 0, 1])
        result = self.f1.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_shape_mismatch(self):
        """
        The method should raise ValueError when shapes of y and yp differ.
        """
        y = np.array([1, 0])
        yp = np.array([1, 0, 1])
        with self.assertRaises(ValueError):
            self.f1.value(y, yp)

if __name__ == '__main__':
    unittest.main()
