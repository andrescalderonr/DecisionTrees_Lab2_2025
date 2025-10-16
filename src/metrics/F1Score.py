import numpy as np
from .Metric import Metric

class F1Score(Metric):
    """
    F1 Score metric for evaluating classification performance.
    Implements the metric interface for neural network evaluation.

    Methods:
       - value(y: np.ndarray, yp: np.ndarray) -> np.ndarray: Computes the F1 score between true and predicted labels.
    """
    def __init__(self):
        super().__init__()

    def value(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculates the F1 score of predictions.

        F1 Score is the harmonic mean of precision and recall,
        providing a balance between the two metrics.

        Args:
        - y (np.ndarray): Array of true labels.
        - yp (np.ndarray): Array of predicted labels.

        Returns:
        - np.ndarray: A single-element array containing the F1 score.

        Raises:
        - ValueError: If the shapes of y and yp do not match.
        """
        if y.shape != yp.shape:
            raise ValueError("True labels and predicted labels must have the same shape.")

        true_positive = np.sum((yp == 1) & (y == 1))
        false_positive = np.sum((yp == 1) & (y == 0))
        false_negative = np.sum((yp == 0) & (y == 1))

        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return np.array([f1])
