import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


class PushModel:
    """
    A class representing a logistic regression model with feature scaling and binary
    classification.
    """

    def __init__(
        self, penalty_param: str, c_param: float, prediction_threshold: float
    ) -> None:
        """
        Initialize PushModel with logistic regression pipeline and prediction threshold.

        Parameters:
        - penalty_param: Regularization penalty type for logistic regression.
        - c_param: Inverse of regularization strength for logistic regression.
        - prediction_threshold: Threshold for binary classification predictions.
        """
        self.FEATURE_COLS = [
            "ordered_before",
            "abandoned_before",
            "global_popularity",
            "set_as_regular",
        ]

        self.TARGET_COL = "outcome"

        self.lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LogisticRegression(penalty=penalty_param, C=c_param)),
            ]
        )
        self.prediction_threshold = prediction_threshold

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.FEATURE_COLS]

    def _extract_target(self, df: pd.DataFrame) -> pd.Series:
        return df[self.TARGET_COL]

    def _feature_target_split(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._extract_features(df), self._extract_target(df)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit logistic regression model to the data.

        Parameters:
        - df: DataFrame containing both the features and the target.
        """
        X, y = self._feature_target_split(df)
        self.lr.fit(X, y)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Make binary classification predictions using the fitted model.

        Parameters:
        - df: DataFrame containing the features.
        """
        X = self._extract_features(df)
        probabilities = self.lr.predict_proba(X)[:, 1]
        bool_predictions = probabilities > self.prediction_threshold
        predictions = pd.Series(bool_predictions.astype(int))
        return predictions

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the fitted model and return probabilities of each
        class.

        Parameters:
        - df: DataFrame containing the features.
        """
        X = self._extract_features(df)
        probabilities = self.lr.predict_proba(X)
        return probabilities

    def evaluate_metrics(self, test: pd.DataFrame) -> tuple[float, float]:
        """
        Evaluate the performance metrics of the model on the test data.

        Parameters:
        - test: The test DataFrame containing both the features and the target.

        Returns:
        - A tuple containing the precision-recall AUC and ROC AUC scores.
        """
        X_test, y_test = self._feature_target_split(test)
        y_pred = self.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred)

        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc = auc(recall, precision)

        return (pr_auc, roc_auc)
