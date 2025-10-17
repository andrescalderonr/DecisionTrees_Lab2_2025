import unittest
import pandas as pd
import numpy as np
from src.tree import DecisionTree
from src.exceptions import NotTrainedError, InvalidInputError


class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.x_simple = pd.DataFrame({
            "age": [25, 30, 45, 35, 22],
            "income": [50000, 60000, 80000, 70000, 48000],
            "student": ["no", "yes", "no", "yes", "yes"]
        })
        self.y_simple = pd.DataFrame({
            "buys": [0, 1, 0, 1, 1]
        })

        self.tree = DecisionTree(max_depth=5, min_categories=3)

    def test_train_and_predict(self):
        self.tree.train(self.x_simple, self.y_simple)
        preds = self.tree.predict(self.x_simple)

        self.assertIsInstance(preds, pd.DataFrame)
        self.assertEqual(preds.shape, self.y_simple.shape)

    def test_predict_without_training_raises(self):
        with self.assertRaises(NotTrainedError):
            self.tree.predict(self.x_simple)

    def test_invalid_input_empty_x(self):
        with self.assertRaises(InvalidInputError):
            self.tree.train(pd.DataFrame(), self.y_simple)

    def test_invalid_input_mismatched_shapes(self):
        x_wrong = self.x_simple.iloc[:-1]
        with self.assertRaises(InvalidInputError):
            self.tree.train(x_wrong, self.y_simple)

    def test_tree_depth(self):
        self.tree.train(self.x_simple, self.y_simple)
        depth = self.tree.depth()
        self.assertGreaterEqual(depth, 1)
        self.assertLessEqual(depth, self.tree.max_depth)

    def test_rules_not_empty_after_training(self):
        self.tree.train(self.x_simple, self.y_simple)
        rules = self.tree.rules()
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)

    def test_to_string_representation(self):
        self.tree.train(self.x_simple, self.y_simple)
        tree_str = self.tree.to_string()
        self.assertIsInstance(tree_str, str)
        self.assertIn("depth=", tree_str)

    def test_prediction_output_nan_if_no_prediction(self):
        self.tree.tree = {
            "is_leaf": True,
            "prediction": None,
            "depth": 0,
            "n_samples": 0
        }
        self.tree._y_columns = ["buys"]
        self.tree._n_output_cols = 1

        preds = self.tree.predict(self.x_simple)
        self.assertTrue(preds.isna().all().all())

    def test_train_with_nan_values(self):
        x_nan = self.x_simple.copy()
        x_nan.loc[0, "age"] = np.nan
        x_nan.loc[2, "student"] = np.nan
        self.tree.train(x_nan, self.y_simple)
        preds = self.tree.predict(x_nan)
        self.assertEqual(preds.shape, self.y_simple.shape)

    def test_min_categories_respected(self):
        small_tree = DecisionTree(max_depth=5, min_categories=10)
        small_tree.train(self.x_simple, self.y_simple)
        self.assertTrue(small_tree.tree["is_leaf"])

    def test_perfect_split(self):
        x_perfect = pd.DataFrame({
            "feature": [0, 0, 1, 1]
        })
        y_perfect = pd.DataFrame({
            "label": [0, 0, 1, 1]
        })
        tree = DecisionTree(max_depth=2, min_categories=1)
        tree.train(x_perfect, y_perfect)
        preds = tree.predict(x_perfect)
        self.assertTrue((preds["label"] == y_perfect["label"]).all())


if __name__ == '__main__':
    unittest.main()
