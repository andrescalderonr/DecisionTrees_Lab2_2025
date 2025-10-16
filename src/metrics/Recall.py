import numpy as np
from .Metric import Metric

class Recall(Metric):
    """
    Recall metric for evaluating classification performance.
    Implements the metric interface for neural network evaluation.

    Methods:
       - value(y: np.ndarray, yp: np.ndarray) -> np.ndarray: Computes the recall score between true and predicted labels.
    """
    def __init__(self):
        super().__init__()

    def value(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculates the recall of predictions.

        Recall is defined as the proportion of true positive predictions
        over all actual positive instances in the dataset.

        Args:
        - y (np.ndarray): Array of true labels.
        - yp (np.ndarray): Array of predicted labels.

        Returns:
        - np.ndarray: A single-element array containing the recall score.

        Raises:
        - ValueError: If the shapes of y and yp do not match.
        """
        if y.shape != yp.shape:
            raise ValueError("True labels and predicted labels must have the same shape.")

        true_positive = np.sum((yp == 1) & (y == 1))
        false_negative = np.sum((yp == 0) & (y == 1))

        recall = true_positive / (true_positive + false_negative + 1e-10)
        return np.array([recall])
