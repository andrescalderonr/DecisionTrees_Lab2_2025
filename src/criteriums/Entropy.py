import numpy as np
import pandas as pd
from .Criterium import Criterium

class Entropy(Criterium):
    """
    Entropy criterium for evaluating attribute selection in decision trees.
    Implements the impurity interface using Shannon entropy.

    Methods:
       - impurity(V: pd.DataFrame) -> float: Computes the entropy of a node.
       - gain(a: str, X: pd.DataFrame, Y: list[pd.DataFrame]) -> float: Computes the information gain from splitting a node.
       - treeImpurity(nodes: list[pd.DataFrame]) -> float: Computes the total entropy across all nodes in a tree.
    """
    def __init__(self):
        super().__init__()

    def impurity(self, V: pd.DataFrame) -> float:
        """
        Calculates the entropy of a node.

        Entropy measures the uncertainty in the distribution of class labels.
        It is defined as: -∑(p_i * log2(p_i)) for each class i.

        Args:
        - V (pd.DataFrame): Data samples for a node. The last column must contain class labels.

        Returns:
        - float: The entropy value of the node.

        Raises:
        - ValueError: If the input DataFrame is empty.
        """
        if V.empty:
            raise ValueError("Input DataFrame is empty.")

        labels = V.iloc[:, -1]
        probabilities = labels.value_counts(normalize=True)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def gain(self, a: str, X: pd.DataFrame, Y: list[pd.DataFrame]) -> float:
        """
        Calculates the information gain from splitting a node using a given attribute.

        Information gain is the reduction in entropy after the split.

        Args:
        - a (str): The attribute used for splitting.
        - X (pd.DataFrame): Original dataset before the split.
        - Y (list[pd.DataFrame]): List of datasets after the split.

        Returns:
        - float: The information gain value.

        Raises:
        - ValueError: If the input dataset is empty or Y is not a list of DataFrames.
        """
        if X.empty or not all(isinstance(df, pd.DataFrame) for df in Y):
            raise ValueError("Invalid input data for gain calculation.")

        total_entropy = self.impurity(X)
        weighted_entropy = sum((len(subset) / len(X)) * self.impurity(subset) for subset in Y)
        gain = total_entropy - weighted_entropy
        return gain

    def treeImpurity(self, nodes: list[pd.DataFrame]) -> float:
        """
        Calculates the total entropy across all nodes in a decision tree.

        This is useful for evaluating the overall impurity of a tree structure.

        Args:
        - nodes (list[pd.DataFrame]): List of data samples for each node in the tree.

        Returns:
        - float: The total entropy of the tree.

        Raises:
        - ValueError: If the node list is empty or contains invalid elements.
        """
        if not nodes or not all(isinstance(node, pd.DataFrame) for node in nodes):
            raise ValueError("Invalid node list for tree impurity calculation.")

        total_samples = sum(len(node) for node in nodes)
        total_entropy = sum((len(node) / total_samples) * self.impurity(node) for node in nodes)
        return total_entropy
