import pandas as pd
from abc import ABC, abstractmethod

class Criterium(ABC):
    """
    Abstract class that defines the core behavior for attribute selection criteria.
    It represents the impurity measure used to evaluate and prioritize features in decision trees.

    Attributes:
        criterium_map (dict): A dictionary mapping criterium names to their corresponding class types.

    Methods:
        use(cls, name: str) -> "Criterium":
            Given the name of a criterium, returns an instance of the corresponding criterium class.

        impurity(self, V: pd.DataFrame) -> float:
            Computes the impurity of a single node based on its samples.

        gain(self, a: str, X: pd.DataFrame, Y: list[pd.DataFrame]) -> float:
            Computes the information gain of splitting a node using a given attribute.

        treeImpurity(self, nodes: list[pd.DataFrame]) -> float:
            Computes the overall impurity of a tree based on its nodes.
    """

    criterium_map = {}

    @classmethod
    def use(cls, name: str) -> "Criterium":
        """
        Returns the criterium object based on the provided name.

        This method implements the Factory Design Pattern, allowing dynamic instantiation
        of the appropriate criterium class based on the input name.

        Args:
            name (str): The name of the criterium (e.g., 'entropy', 'gini').
        Returns:
            Criterium: An instance of the corresponding criterium class.
        Raises:
            ValueError: If the provided name doesn't match any known criteria.
        """
        criterium_class = cls.criterium_map.get(name.lower())
        if criterium_class:
            return criterium_class()
        else:
            raise ValueError(f"Criterium '{name}' not supported")

    @abstractmethod
    def impurity(self, values: pd.DataFrame) -> float:
        """
        Computes the impurity of a node based on its samples.

        This method should be implemented by subclasses to define how impurity is calculated,
        such as using entropy or Gini index.

        Args:
            values(pd.DataFrame): Data samples for a node.

        Returns:
            float: The impurity value of the node.
        """
        pass

    @abstractmethod
    def gain(self, a: str, x: pd.DataFrame, y: list[pd.DataFrame]) -> float:
        """
        Computes the information gain of splitting a node using a given attribute.

        This method should be implemented by subclasses to define how gain is calculated
        based on the impurity reduction across child nodes.

        Args:
            a (str): The attribute being evaluated.
            x (pd.DataFrame): Input samples.
            y (list[pd.DataFrame]): List of output samples for each child node.

        Returns:
            float: The information gain value.
        """
        pass

    @abstractmethod
    def tree_impurity(self, nodes: list[pd.DataFrame]) -> float:
        """
        Computes the overall impurity of a tree based on its nodes.

        This method should be implemented by subclasses to define how the total impurity
        is aggregated across all nodes in the tree.

        Args:
            nodes (list[pd.DataFrame]): List of data samples for each node in the tree.

        Returns:
            float: The total impurity of the tree.
        """
        pass
