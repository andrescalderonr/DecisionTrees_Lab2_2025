import numpy as np
from .Metric import Metric

class Accuracy(Metric):
    """
    Accuracy metric for evaluating classification performance.
    Implements the metric interface for neural network evaluation.

    Methods:
       - value(y: np.ndarray, yp: np.ndarray) -> np.ndarray: Computes the accuracy score between true and predicted labels.
    """
    def __init__(self):
        super().__init__()

    def value(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculates the accuracy of predictions.

        Accuracy is defined as the proportion of correctly predicted labels
        over the total number of predictions.

        Args:
        - y (np.ndarray): Array of true labels.
        - yp (np.ndarray): Array of predicted labels.

        Returns:
        - np.ndarray: A single-element array containing the accuracy score.

        Raises:
        - ValueError: If the shapes of y and yp do not match.
        """
        if y.shape != yp.shape:
            raise ValueError("True labels and predicted labels must have the same shape.")

        accuracy = np.mean(y == yp)
        return np.array([accuracy])