import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)

DATA_PATH = os.path.join(os.getcwd(), "data", "feature_frame.csv")


def load_raw_data() -> pd.DataFrame:
    logging.info(f"Loading dataset from {DATA_PATH}")
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return None


def filter_orders_with_minimum_size(
    df: pd.DataFrame, min_size: int = 5
) -> pd.DataFrame:
    """
    Filter orders that have at least 5 products
    """
    filt = df.query("outcome == 1.0").groupby("order_id").size() >= min_size
    order_ids = set(filt[filt].index)
    filtered_df = df[df["order_id"].isin(order_ids)]
    return filtered_df


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building feature frame")
    preprocessed_data = (
        data.pipe(filter_orders_with_minimum_size, min_size=5)
        .assign(created_at=lambda x: pd.to_datetime(x.created_at))
        .assign(order_date=lambda x: pd.to_datetime(x.order_date))
    )
    return preprocessed_data


def load_training_feature_frame() -> pd.DataFrame:
    logging.info("Loading feature frame")
    raw_data = load_raw_data()
    data = build_feature_frame(raw_data)
    return data
