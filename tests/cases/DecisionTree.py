import unittest
import pandas as pd
from src.tree import DecisionTree

class TestCases(unittest.TestCase):

    def test_and_operator(self):
        x = pd.DataFrame({
            "A": [0, 0, 1, 1],
            "B": [0, 1, 0, 1]
        })
        y = pd.DataFrame({
            "AND": [0, 0, 0, 1]
        })

        tree = DecisionTree(max_depth=3, min_categories=1)
        tree.train(x, y)
        preds = tree.predict(x)
        self.assertTrue((preds["AND"] == y["AND"]).all())

    def test_or_operator(self):
        x = pd.DataFrame({
            "A": [0, 0, 1, 1],
            "B": [0, 1, 0, 1]
        })
        y = pd.DataFrame({
            "OR": [0, 1, 1, 1]
        })

        tree = DecisionTree(max_depth=3, min_categories=1)
        tree.train(x, y)
        preds = tree.predict(x)
        self.assertTrue((preds["OR"] == y["OR"]).all())

    def test_xor_operator(self):
        x = pd.DataFrame({
            "A": [0, 0, 1, 1],
            "B": [0, 1, 0, 1]
        })
        y = pd.DataFrame({
            "XOR": [0, 1, 1, 0]
        })

        tree = DecisionTree(max_depth=5, min_categories=1)
        tree.train(x, y)
        preds = tree.predict(x)
        self.assertTrue((preds["XOR"] == y["XOR"]).all())

    def test_adult_dataset_manual_split(self):
        column_names = [

            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",

            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",

            "hours-per-week", "native-country", "income"

        ]

        df = pd.read_csv("resources/adult.data", names=column_names, na_values=" ?", skipinitialspace=True)

        df = df.dropna()

        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        n_train = int(0.8 * len(df))

        df_train = df.iloc[:n_train]

        df_test = df.iloc[n_train:]

        # The failing line is here:
        x_train = df_train.drop("income", axis=1)
        y_train = df_train[["income"]]

        x_test = df_test.drop("income", axis=1)
        y_test = df_test[["income"]]

        # --- REVISED FIX: Convert to category and explicitly add the missing token ---
        MISSING_TOKEN = "__MISSING__"

        for col in x_train.select_dtypes(include='object').columns:
            # Convert 'object' column to 'category' first.
            # This will auto-discover categories present in the current slice.
            x_train[col] = x_train[col].astype('category')
            x_test[col] = x_test[col].astype('category')

            x_train[col] = x_train[col].cat.add_categories(MISSING_TOKEN)
            x_test[col] = x_test[col].cat.add_categories(MISSING_TOKEN)

        y_train = y_train.copy()
        y_test = y_test.copy()
        y_train.loc[:, "income"] = y_train["income"].astype('category')
        y_test.loc[:, "income"] = y_test["income"].astype('category')

        # ----------------------------------------------------------------------

        tree = DecisionTree(max_depth=10, min_categories=5)
        tree.train(x_train, y_train)

        preds = tree.predict(x_test)

        accuracy = tree.metric(y_test, preds)

        print(f"\nPrecisión en dataset Adult (sin sklearn): {accuracy:.4f}")
        self.assertGreater(accuracy, 0.75)

if __name__ == "__main__":
    unittest.main()
