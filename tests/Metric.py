import unittest
import numpy as np
from src.metrics import Metric

class TestAccuracy(unittest.TestCase):
    """
    Unit tests for the Accuracy metric, commonly used to evaluate classification models.
    The Accuracy metric calculates the proportion of correct predictions made by the model.
    It is defined as:

    Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
    """

    def setUp(self):
        """
        Initializes the Accuracy metric before each test.
        """
        self.accuracy = Metric.use("accuracy")

    def test_perfect_accuracy(self):
        """
        Tests the case where the predicted values exactly match the true labels.
        For perfect accuracy, the expected result should be 1.0.

        In this test, all predictions are correct, so accuracy = 4 correct predictions / 4 total predictions = 1.0.
        """
        y = np.array([1, 0, 1, 1])
        yp = np.array([1, 0, 1, 1])
        result = self.accuracy.value(y, yp)
        expected = 1.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_partial_accuracy(self):
        """
        Tests the case where some of the predictions match the true labels.
        In this case, the accuracy should be less than 1.0.

        Here, 3 out of 4 predictions are correct, so accuracy = 3 correct predictions / 4 total predictions = 0.75.
        """
        y = np.array([1, 0, 1, 0])
        yp = np.array([1, 1, 1, 0])
        result = self.accuracy.value(y, yp)
        expected = 0.75
        self.assertAlmostEqual(float(result[0]), expected, places=5)


    def test_zero_accuracy(self):
        """
        Tests the case where none of the predictions match the true labels.
        For zero accuracy, the expected result should be 0.0.

        In this case, 0 out of 3 predictions are correct, so accuracy = 0 correct predictions / 3 total predictions = 0.0.
        """
        y = np.array([0, 0, 1])
        yp = np.array([1, 1, 0])
        result = self.accuracy.value(y, yp)
        expected = 0.0
        self.assertAlmostEqual(float(result[0]), expected, places=5)

    def test_shape_mismatch(self):
        """
        Tests that the `value` method raises a ValueError when the shapes of the true labels `y` and
        the predicted labels `yp` do not match.

        The true labels and predicted labels must have the same length for accuracy to be calculated.
        """
        y = np.array([1, 0])
        yp = np.array([1, 0, 1])
        with self.assertRaises(ValueError):
            self.accuracy.value(y, yp)

if __name__ == '__main__':
    unittest.main()