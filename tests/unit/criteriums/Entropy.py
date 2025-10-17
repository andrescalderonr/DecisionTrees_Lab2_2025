import unittest
import pandas as pd
from src.criteriums import Criterium

class TestEntropy(unittest.TestCase):
    """
    Unit tests for the Entropy criterium, used to measure impurity in decision tree nodes.

    Entropy quantifies the uncertainty in the distribution of class labels using Shannon entropy:
    Entropy = -sum(p_i * log2(p_i)) over classes i.

    These tests validate:
    - Correct entropy values for pure and balanced splits.
    - Correct information gain computation for simple splits.
    - Correct weighted tree impurity aggregation.
    - Proper error handling for invalid inputs.
    """

    def setUp(self):
        """
        Instantiates the Entropy criterium before each test.

        Uses the Criterium factory to obtain the 'entropy' implementation so tests
        exercise the same registration/lookup mechanism used in production code.
        """
        self.entropy = Criterium.use("entropy")

    def test_impurity_pure_node(self):
        """
        Verifies that a node with identical labels has zero entropy.

        A pure node has no uncertainty, therefore expected entropy = 0.0.
        Example:
        - labels = [1, 1, 1, 1] -> entropy = 0.0
        """
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "label": [1, 1, 1, 1]
        })
        result = self.entropy.impurity(df)
        expected = 0.0
        self.assertAlmostEqual(result, expected, places=7)

    def test_impurity_uniform_binary(self):
        """
        Verifies that a binary split with equal class frequencies yields entropy = 1.0.

        For two classes with p=0.5 each, entropy (base 2) equals 1.0.
        Example:
        - labels = [0, 0, 1, 1] -> entropy = 1.0
        """
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "label": [0, 0, 1, 1]
        })
        result = self.entropy.impurity(df)
        expected = 1.0
        self.assertAlmostEqual(result, expected, places=7)

    def test_gain_simple_split(self):
        """
        Validates information gain for a perfectly separable split.

        Parent node: labels [0,0,1,1] -> entropy = 1.0
        Children:
         - left: labels [0,0] -> entropy = 0.0
         - right: labels [1,1] -> entropy = 0.0

        Weighted child entropy = 0.0, so information gain = 1.0.
        """
        parent = pd.DataFrame({
            "a": [0, 0, 1, 1],
            "label": [0, 0, 1, 1]
        })
        left = pd.DataFrame({
            "a": [0, 0],
            "label": [0, 0]
        })
        right = pd.DataFrame({
            "a": [1, 1],
            "label": [1, 1]
        })
        gain = self.entropy.gain("a", parent, [left, right])
        expected = 1.0
        self.assertAlmostEqual(gain, expected, places=7)

    def test_tree_impurity_weighted(self):
        """
        Checks weighted tree impurity calculation.

        Two nodes:
         - node1: 3 samples with labels [0,0,0] -> entropy = 0.0
         - node2: 1 sample with label [1] -> entropy = 0.0

        Total samples = 4, weighted entropy = 0.0.
        """
        node1 = pd.DataFrame({"f": [1,2,3], "label": [0,0,0]})
        node2 = pd.DataFrame({"f": [4], "label": [1]})
        total = self.entropy.tree_impurity([node1, node2])
        expected = 0.0
        self.assertAlmostEqual(total, expected, places=7)

    def test_impurity_empty_raises(self):
        """
        Ensures impurity raises ValueError for an empty DataFrame.

        The method requires at least one sample to compute class probabilities.
        Passing an empty DataFrame must raise a ValueError to signal invalid input.
        """
        empty = pd.DataFrame(columns=["f", "label"])
        with self.assertRaises(ValueError):
            self.entropy.impurity(empty)

    def test_gain_invalid_inputs_raise(self):
        """
        Ensures gain raises ValueError for invalid inputs.

        Cases covered:
        - x is empty (no parent samples) should raise ValueError.
        - y contains non-DataFrame elements should raise ValueError.
        """
        parent_empty = pd.DataFrame(columns=["a", "label"])
        invalid_child = "not_a_df"
        with self.assertRaises(ValueError):
            self.entropy.gain("a", parent_empty, [pd.DataFrame(), pd.DataFrame()])
        with self.assertRaises(ValueError):
            self.entropy.gain("a", pd.DataFrame({"a":[1], "label":[0]}), [pd.DataFrame(), invalid_child])

    def test_tree_impurity_invalid_nodes_raise(self):
        """
        Ensures tree_impurity validates its input list.

        Cases covered:
        - empty list should raise ValueError.
        - list containing non-DataFrame elements should raise ValueError.
        """
        with self.assertRaises(ValueError):
            self.entropy.tree_impurity([])
        with self.assertRaises(ValueError):
            self.entropy.tree_impurity([pd.DataFrame(), "not_a_df"])

if __name__ == '__main__':
    unittest.main()
