import pandas as pd

from src.module_3.data_processing import filter_orders_with_minimum_size


def test_orders_with_minimum_size() -> None:
    df = pd.DataFrame(
        {"order_id": [1, 1, 2, 2, 3, 3, 3, 4], "outcome": [0, 1, 1, 1, 1, 1, 1, 0]}
    )
    result = filter_orders_with_minimum_size(df, 2).reset_index(drop=True)
    expected = pd.DataFrame(
        {"order_id": [2, 2, 3, 3, 3], "outcome": [1, 1, 1, 1, 1]}
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(result, expected)
    assert result.order_id.nunique() <= expected.order_id.nunique()
    assert len(result) <= len(df)
