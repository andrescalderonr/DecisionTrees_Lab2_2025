import numpy as np
from .Metric import Metric

class Precision(Metric):
    """
    Precision metric for evaluating classification performance.
    Implements the metric interface for neural network evaluation.

    Methods:
       - value(y: np.ndarray, yp: np.ndarray) -> np.ndarray: Computes the precision score between true and predicted labels.
    """
    def __init__(self):
        super().__init__()

    def value(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculates the precision of predictions.

        Precision is defined as the proportion of true positive predictions
        over all positive predictions made by the model.

        Args:
        - y (np.ndarray): Array of true labels.
        - yp (np.ndarray): Array of predicted labels.

        Returns:
        - np.ndarray: A single-element array containing the precision score.

        Raises:
        - ValueError: If the shapes of y and yp do not match.
        """
        if y.shape != yp.shape:
            raise ValueError("True labels and predicted labels must have the same shape.")

        true_positive = np.sum((yp == 1) & (y == 1))
        false_positive = np.sum((yp == 1) & (y == 0))

        precision = true_positive / (true_positive + false_positive + 1e-10)
        return np.array([precision])
