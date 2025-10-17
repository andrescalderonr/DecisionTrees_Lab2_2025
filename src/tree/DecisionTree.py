import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.metrics import Metric
from src.criteriums import Criterium
from src.exceptions import NotTrainedError, InvalidInputError, GainCalculationError

# --- Utility/Format Methods ---

def _format_continuous_node(n, prefix, lines, recurse_fn):
    """
    Formats a continuous internal decision node in the tree.

    Args:
        n (dict): Node dictionary.
        prefix (str): String used for indentation/formatting.
        lines (list): List of lines to which formatted output is appended.
        recurse_fn (function): Recursive function to traverse the tree.
    """
    lines.append(f"{prefix}Node depth={n['depth']} attr={n['attr']} <= {n['threshold']}")
    if n.get("left"):
        recurse_fn(n["left"], prefix + "  ")
    if n.get("right"):
        recurse_fn(n["right"], prefix + "  ")

def _format_categorical_node(n, prefix, lines, recurse_fn):
    """
    Formats a categorical internal decision node in the tree.

    Args:
        n (dict): Node dictionary.
        prefix (str): String used for indentation/formatting.
        lines (list): List of lines to which formatted output is appended.
        recurse_fn (function): Recursive function to traverse the tree.
    """
    lines.append(f"{prefix}Node depth={n['depth']} attr={n['attr']} categorical")
    for k, child in n["children"].items():
        lines.append(f"{prefix}  if {n['attr']} == {k}:")
        recurse_fn(child, prefix + "    ")

def _format_prediction(prediction):
    """
    Formats prediction output (Series or scalar).

    Args:
        prediction: The prediction value (can be None, scalar, or pd.Series).

    Returns:
        Formatted prediction (dictionary if Series).
    """
    if prediction is None:
        return None
    if isinstance(prediction, pd.Series):
        return prediction.to_dict()
    return prediction

def _format_leaf(n, prefix):
    """
    Formats a leaf node for string representation.

    Args:
        n (dict): Node dictionary.
        prefix (str): String used for indentation/formatting.

    Returns:
        str: Formatted string representing the leaf node.
    """
    pred = _format_prediction(n.get("prediction"))
    return f"{prefix}LEAF depth={n['depth']} samples={n.get('n_samples', 0)} pred={pred}"

def _make_leaf(y, depth):
    """
    Creates a leaf node from the current data.

    Args:
        y (pd.DataFrame): Target values at this node.
        depth (int): Current depth of the node.

    Returns:
        dict: Leaf node with prediction and metadata.
    """
    node = {
        "is_leaf": True,
        "prediction": None,
        "attr": None,
        "threshold": None,
        "children": {},
        "left": None,
        "right": None,
        "depth": depth,
        "n_samples": len(y)
    }
    if len(y) > 0:
        node["prediction"] = y.mode(dropna=True).iloc[0]
    return node

def _get_child_node_continuous(node, row):
    """
    Traverses to the next child node for a continuous attribute.

    Args:
        node (dict): Current node.
        row (pd.Series): A single input sample.

    Returns:
        dict: Child node to continue traversal.
    """
    attr = node["attr"]
    val = row[attr]

    if pd.isna(val):
        child = node["children"].get(None)
        return child if child is not None else (node.get("left") or node.get("right"))

    return node.get("left") if val <= node["threshold"] else node.get("right")

def _get_child_node_categorical(node, row):
    """
    Traverses to the next child node for a categorical attribute.

    Args:
        node (dict): Current node.
        row (pd.Series): A single input sample.

    Returns:
        dict: Child node to continue traversal.
    """
    attr = node["attr"]
    val = row[attr]
    key = val if not pd.isna(val) else None

    child = node["children"].get(key)
    if child is None:
        if not node["children"]:
            return None
        return max(node["children"].values(), key=lambda c: c.get("n_samples", 0))
    return child

class DecisionTree:
    """
    Represents a decision tree classifier.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_categories (int): Minimum number of samples to consider for splitting.
        criterium (Criterium): Impurity measure (e.g., Entropy).
        metric_obj (Metric): Evaluation metric (e.g., Accuracy).
        tree (dict): Root node of the built tree.
        _rules (list): Human-readable splitting rules.
        _impurity_history (list): Impurity values at each split.
        _y_columns (list): Output column names from training labels.
        _n_output_cols (int): Number of output columns.
        print_impurity (bool): Flag to enable impurity printing during training.
    """

    def __init__(self, max_depth, min_categories):
        """
        Initializes the DecisionTree.

        Args:
            max_depth (int): Maximum tree depth allowed.
            min_categories (int): Minimum sample size for splits.
        """
        self.print_impurity = None
        self.max_depth = int(max_depth)
        self.min_categories = max(1, int(min_categories))
        self.criterium = Criterium.use("entropy")
        self.metric_obj = Metric.use("accuracy")
        self.tree = None
        self._rules = []
        self._impurity_history = []
        self._y_columns = []
        self._n_output_cols = 0

    # --- Core Methods ---

    def metric(self, y, yp):
        """
        Computes the evaluation metric between true and predicted values.

        Args:
            y (pd.DataFrame): True labels.
            yp (pd.DataFrame): Predicted labels.

        Returns:
            float: Evaluation score.
        """
        y = y.to_numpy()
        yp = yp.to_numpy()
        return float(self.metric_obj.value(y, yp))

    def predict(self, x):
        """
        Predicts the output labels for input data using the trained tree.

        Args:
            x (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Predicted labels.

        Raises:
            NotTrainedError: If the tree hasn't been trained.
        """
        if self.tree is None:
            raise NotTrainedError("The tree has not been trained")

        rows = []
        for _, row in x.iterrows():
            node = self.tree
            while not node["is_leaf"]:
                if node["threshold"] is not None:
                    child = _get_child_node_continuous(node, row)
                else:
                    child = _get_child_node_categorical(node, row)
                if child is None:
                    break
                node = child

            pred = node.get("prediction")
            rows.append([np.nan] * self._n_output_cols if pred is None else pred.values.tolist())

        return pd.DataFrame(rows, columns=self._y_columns)

    def train(self, x, y, print_impurity=False, do_graphic=False):
        """
        Trains the decision tree on the given dataset.

        Args:
            x (pd.DataFrame): Input features.
            y (pd.DataFrame): Target labels.
            print_impurity (bool): Whether to print impurity after each split.
            do_graphic (bool): Whether to plot impurity over splits.

        Raises:
            InvalidInputError: If inputs are empty or mismatched.
        """
        if len(x) == 0:
            raise InvalidInputError("X is empty")
        if len(x) != len(y):
            raise InvalidInputError("X and Y must have the same number of rows")

        self._y_columns = list(y.columns)
        self._n_output_cols = len(self._y_columns)
        self._rules = []
        self._impurity_history = []
        self.print_impurity = print_impurity
        self.tree = self._build_node_recursive(x.copy(), y.copy(), depth=0)
        self._fill_predictions(self.tree)

        if do_graphic:
            self._plot_impurity()

    def depth(self):
        """
        Returns the depth of the trained tree.

        Returns:
            int: Maximum depth of the tree.
        """
        if self.tree is None:
            return 0

        def max_depth(n):
            if n["is_leaf"]:
                return n["depth"]
            depths = []
            if n.get("left"):
                depths.append(max_depth(n["left"]))
            if n.get("right"):
                depths.append(max_depth(n["right"]))
            for c in n["children"].values():
                if c is not n.get("left") and c is not n.get("right"):
                    depths.append(max_depth(c))
            return max(depths) if depths else n["depth"]

        return max_depth(self.tree)

    def rules(self):
        """
        Returns the list of split rules derived during training.

        Returns:
            list: List of human-readable rules.
        """
        return list(self._rules)

    def to_string(self):
        """
        Returns a string representation of the entire tree.

        Returns:
            str: Formatted string tree view.
        """
        if self.tree is None:
            return "<empty tree>"

        lines = []

        def recurse(n, prefix=""):
            if n["is_leaf"]:
                lines.append(_format_leaf(n, prefix))
            elif n.get("threshold") is not None:
                _format_continuous_node(n, prefix, lines, recurse)
            else:
                _format_categorical_node(n, prefix, lines, recurse)

        recurse(self.tree, "")
        return "\n".join(lines)

    # --- Helper Methods ---

    def _fallback_gain(self, y_splits):
        """
        Manually computes gain if the primary gain function fails.

        Args:
            y_splits (list): List of label partitions.

        Returns:
            float: Computed gain.

        Raises:
            GainCalculationError: If manual gain calculation fails.
        """
        try:
            all_y = pd.concat(y_splits, ignore_index=True)
            base = self.criterium.impurity(all_y)
            n_all = len(all_y) if len(all_y) > 0 else 1
            weighted = sum((len(part) / n_all) * self.criterium.impurity(part) for part in y_splits)
            return base - weighted
        except (ValueError, ZeroDivisionError) as e:
            raise GainCalculationError(
                f"Manual gain calculation failed: {e}"
            ) from e

    def _find_best_split_continuous(self, samples, y, attr, best_gain, best_split_data):
        """
        Finds the best threshold to split a continuous attribute.

        Args:
            samples (pd.DataFrame): Input features.
            y (pd.DataFrame): Target labels.
            attr (str): Attribute name.
            best_gain (float): Current best gain.
            best_split_data (tuple): Current best split configuration.

        Returns:
            tuple: Updated best gain and split data.
        """
        col = samples[attr]
        uniq = np.unique(col.dropna())
        if len(uniq) <= 1:
            return best_gain, best_split_data

        thresholds = (uniq[:-1] + uniq[1:]) / 2.0

        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            y_left = y[left_mask]
            y_right = y[right_mask]

            try:
                gain = self.criterium.gain(attr, samples[[attr]], [y_left, y_right])
            except (ValueError, ZeroDivisionError):
                try:
                    gain = self._fallback_gain([y_left, y_right])
                except GainCalculationError:
                    gain = 0.0

            if gain > best_gain:
                best_gain = gain
                best_split_data = (attr, float(t), (left_mask, right_mask))

        return best_gain, best_split_data

    def _find_best_split_categorical(self, samples, y, attr, best_gain, best_split_data):
        """
        Finds the best way to split a categorical attribute.

        Args:
            samples (pd.DataFrame): Input features.
            y (pd.DataFrame): Target labels.
            attr (str): Attribute name.
            best_gain (float): Current best gain.
            best_split_data (tuple): Current best split configuration.

        Returns:
            tuple: Updated best gain and split data.
        """
        col = samples[attr]
        filled = col.fillna("__MISSING__")
        values = filled.unique()
        masks = []
        y_parts = []

        for v in values:
            mask = (filled == v)
            masks.append(mask)
            y_parts.append(y[mask])

        try:
            gain = self.criterium.gain(attr, samples[[attr]], y_parts)
        except (ValueError, ZeroDivisionError):
            try:
                gain = self._fallback_gain(y_parts)
            except GainCalculationError:
                gain = 0.0

        if gain > best_gain:
            best_gain = gain
            best_split_data = (attr, None, (values, masks))

        return best_gain, best_split_data

    def _find_optimal_split(self, samples, y):
        """
        Iterates over all attributes to find the optimal split.

        Args:
            samples (pd.DataFrame): Input features.
            y (pd.DataFrame): Target labels.

        Returns:
            tuple: Best gain and corresponding split info.
        """
        best_gain = -np.inf
        best_split_data = (None, None, None)

        for attr in samples.columns:
            col = samples[attr]
            if pd.api.types.is_numeric_dtype(col):
                best_gain, best_split_data = self._find_best_split_continuous(
                    samples, y, attr, best_gain, best_split_data
                )
            else:
                best_gain, best_split_data = self._find_best_split_categorical(
                    samples, y, attr, best_gain, best_split_data
                )

        return best_gain, best_split_data

    def _build_node_recursive(self, samples, y, depth):
        """
        Core recursive method to build the tree node-by-node.

        Args:
            samples (pd.DataFrame): Input features at this node.
            y (pd.DataFrame): Target values at this node.
            depth (int): Current depth in the tree.

        Returns:
            dict: Constructed node.
        """
        if depth >= self.max_depth:
            return _make_leaf(y, depth)

        if len(y) < self.min_categories:
            return _make_leaf(y, depth)

        if len(y.drop_duplicates()) == 1:
            return _make_leaf(y, depth)

        best_gain, (best_attr, best_threshold, best_split_info) = self._find_optimal_split(samples, y)

        if best_attr is None or best_split_info is None:
            return _make_leaf(y, depth)

        if best_gain <= 0 and depth >= self.max_depth - 1:
            return _make_leaf(y, depth)

        node = {
            "is_leaf": False, "prediction": None, "attr": best_attr,
            "threshold": best_threshold, "children": {}, "left": None,
            "right": None, "depth": depth, "n_samples": len(y)
        }

        if best_threshold is not None:
            self._split_continuous_node(node, samples, y, depth, best_split_info)
        else:
            self._split_categorical_node(node, samples, y, depth, best_split_info)

        try:
            self._impurity_history.append(float(self.criterium.tree_impurity([])))
        except (ValueError, ZeroDivisionError):
            self._impurity_history.append(float("nan"))

        if self.print_impurity:
            print(f"Depth {depth} split {best_attr} gain {best_gain} impurity {self._impurity_history[-1]}")

        return node

    def _split_continuous_node(self, node, samples, y, depth, best_split_info):
        """
        Splits a decision tree node based on a continuous attribute and recursively builds child nodes.
    
        Args:
            node (dict): Current node dictionary to update with children.
            samples (pd.DataFrame): Input features at this node.
            y (pd.DataFrame): Target values at this node.
            depth (int): Current depth in the tree.
            best_split_info (tuple): Tuple containing left and right masks for the best split.

        Side effects:
            - Updates the `node` dictionary with left and right children.
            - Records the split rule in `self._rules`.
        """
        left_mask, right_mask = best_split_info

        node["left"] = self._build_node_recursive(samples[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build_node_recursive(samples[right_mask], y[right_mask], depth + 1)

        threshold = node["threshold"]
        attr = node["attr"]

        node["children"] = {"<=%.6g" % threshold: node["left"], ">%.6g" % threshold: node["right"]}
        self._rules.append(f"if {attr} <= {threshold} then ... (depth {depth})")

    def _split_categorical_node(self, node, samples, y, depth, best_split_info):
        """
        Splits a decision tree node based on a categorical attribute and recursively builds child nodes.
    
        Args:
            node (dict): Current node dictionary to update with children.
            samples (pd.DataFrame): Input features at this node.
            y (pd.DataFrame): Target values at this node.
            depth (int): Current depth in the tree.
            best_split_info (tuple): Tuple containing unique values and their corresponding masks.
    
        Side effects:
            - Updates the `node` dictionary with children for each category value.
            - Records each split rule in `self._rules`.
        """
        values, masks = best_split_info
        attr = node["attr"]

        for v, mask in zip(values, masks):
            key = v if v != "__MISSING__" else None
            node["children"][key] = self._build_node_recursive(samples[mask], y[mask], depth + 1)
            self._rules.append(f"if {attr} == {v} then ... (depth {depth})")

    def _fill_predictions(self, n):
        """
        Recursively traverses the tree to fill prediction values (no-op placeholder).

        Args:
            n (dict): Current node.
        """
        if n["is_leaf"]:
            return
        for c in n["children"].values():
            self._fill_predictions(c)

    def _plot_impurity(self):
        """
        Plots the impurity history if available.
        """
        if len(self._impurity_history) == 0:
            print("No impurity history to plot")
        else:
            xs = list(range(len(self._impurity_history)))
            ys = self._impurity_history
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, marker="o", linestyle="-")
            plt.title("Tree impurity per iteration")
            plt.xlabel("Split iteration")
            plt.ylabel("Impurity")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
