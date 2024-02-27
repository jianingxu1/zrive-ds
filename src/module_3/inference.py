import logging
import joblib
import os
import pandas as pd
from typing import Optional
from collections import defaultdict

from data_processing import build_feature_frame
from model import PushModel

MODEL_PATH = os.path.join(
    os.getcwd(),
    "src",
    "module_3",
    "models",
    "2024-02-27_19-41-49_logistic_regression.joblib",
)
DATA_PATH = os.path.join(os.getcwd(), "data", "feature_frame.csv")


def load_csv_data(data_path: str) -> Optional[pd.DataFrame]:
    if not data_path or not os.path.exists(data_path):
        logging.warning("Invalid path to data")
        return None
    df = pd.read_csv(data_path)
    feature_frame = build_feature_frame(df)
    return feature_frame


def load_model(model_path: str) -> Optional[PushModel]:
    """
    Load a model from a joblib file.

    Parameters:
    - model_path: The path to the joblib file containing the model.
    """
    if not model_path or not os.path.exists(model_path):
        logging.warning("Invalid path to model")
        return None
    try:
        logging.info("Loading model")
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        logging.warning(f"Error: File '{model_path}' not found.")
    except Exception as e:
        logging.warning(f"An error occurred while loading the model: {e}")


def infer(model_path: str, data_to_predict_path: str) -> pd.Series:
    data_to_predict = load_csv_data(data_to_predict_path)
    model = load_model(model_path)
    predictions = model.predict(data_to_predict)
    return predictions


def main():
    predictions = infer(MODEL_PATH, DATA_PATH).to_dict()
    results = defaultdict(int)
    for key, value in predictions.items():
        results[value] += 1
    logging.info(f"Predictions obtained: {results.items()}")


if __name__ == "__main__":
    main()
