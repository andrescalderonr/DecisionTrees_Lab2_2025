import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract class that defines the core behavior for evaluation metrics.
    It represents the performance metric used to assess the output of a neural network.

    Attributes:
        metric_map (dict): A dictionary mapping metric names to their corresponding class names.

    Methods:
        use(cls, name: str) -> "Metric":
            Given the name of a metric, returns an instance of the corresponding metric class.

        value(self, y: np.ndarray, yp: np.ndarray) -> float:
            Computes the performance score based on the true labels and predicted outputs.
    """

    metric_map = {}

    @classmethod
    def use(cls, name: str) -> "Metric":
        """
        Returns the metric object based on the provided name.

        This method implements the Factory Design Pattern, allowing dynamic instantiation
        of the appropriate metric class based on the input name.

        Args:
            name (str): The name of the metric (e.g., 'accuracy').
        Returns:
            Metric: An instance of the corresponding metric class (e.g., `Accuracy`).
        Raises:
            ValueError: If the provided name doesn't match any known metrics.
        """
        metric_class = cls.metric_map.get(name.lower())
        if metric_class:
            return metric_class()
        else:
            raise ValueError(f"Metric '{name}' not supported")

    @abstractmethod
    def value(self, y: np.ndarray, yp:np.ndarray) -> np.ndarray:
        """
        Computes the performance score of the neural network.

        This method should be implemented by subclasses to define how the metric is calculated.
        For example, accuracy might return 1 if the score is above a threshold (e.g., 0.6).

        Args:
            y (np.ndarray): Array of true labels.
            yp (np.ndarray): Array of predicted outputs.
        Returns:
            float: The computed performance score.
        """
        pass