import os
import logging
import pandas as pd
import joblib
from datetime import datetime

# from collections import defaultdict
# import matplotlib.pyplot as plt

from data_processing import load_training_feature_frame
from model import PushModel

OUTPUT_PATH = os.path.join(os.getcwd(), "src", "module_3", "models")

PENALTY_PARAM = "l2"
C_PARAM = 1e-6
PREDICTION_THRESHOLD = 0.01821537339520326


def train_test_split(
    df: pd.DataFrame,
    test_proportion: float,
):
    """
    Split train and test by date as dataset contains timeseries
    """
    train_proportion = 1 - test_proportion

    df_sorted = df.sort_values(by="order_date")

    # Consider only orders and not pairs of order, item
    df_orders = df_sorted.drop_duplicates(subset=["order_id"])[
        ["order_id", "order_date"]
    ]
    total_orders = len(df_orders)

    train_count = int(total_orders * train_proportion)

    train_orders = df_orders.iloc[:train_count]["order_id"]
    test_orders = df_orders.iloc[train_count:]["order_id"]

    train = df[df["order_id"].isin(train_orders)]
    test = df[df["order_id"].isin(test_orders)]
    return train, test


def train(data: pd.DataFrame) -> PushModel:
    logging.info("Training model")
    train, test = train_test_split(data, 0.2)

    # Train the model with train, test split to get its performance
    evaluated_model = PushModel(
        penalty_param=PENALTY_PARAM,
        c_param=C_PARAM,
        prediction_threshold=PREDICTION_THRESHOLD,
    )
    evaluated_model.fit(train)
    pr_auc, roc_auc = evaluated_model.evaluate_metrics(test)
    logging.info(f"Model performance: PR AUC:{pr_auc:.6f}, ROC AUC: {roc_auc:.6f}")

    # Train the model with entire dataset
    model = PushModel(
        penalty_param=PENALTY_PARAM,
        c_param=C_PARAM,
        prediction_threshold=PREDICTION_THRESHOLD,
    )
    model.fit(data)

    # # predictions = infer("hi", "hi").to_dict()
    # predictions = model.predict_proba(data)[:, 1]
    # plt.hist(predictions, bins=100)
    # plt.show()

    # myMap = defaultdict(int)
    # # print(predictions)
    # for value in predictions:
    #     myMap[value] += 1
    # print(myMap)

    return model


def generate_model_filename(model_name: str) -> str:
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{current_time_str}_{model_name}.joblib"
    return filename


def save_model(model: PushModel, model_name: str) -> str:
    """
    Saves the model into a file using joblib.
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    filename = generate_model_filename(model_name)
    model_path = os.path.join(OUTPUT_PATH, filename)

    logging.info(f"Saving model in {model_path}")

    try:
        joblib.dump(model, model_path)
        return model_path
    except Exception as e:
        logging.error(f"Error occured while saving the model: {e}")
        return None


def main():
    df = load_training_feature_frame()
    model = train(df)
    save_model(model, "logistic_regression")


if __name__ == "__main__":
    main()
