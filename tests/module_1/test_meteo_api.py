import pandas as pd
import requests
import pytest
from pytest import MonkeyPatch, LogCaptureFixture
from unittest.mock import Mock

from src.module_1.module_1_meteo_api import compute_yearly_mean_and_std, fetch_url


def test_compute_yearly_mean_and_std() -> None:
    test_parameter = "temperature"

    data = {
        "daily": {
            "time": ["1950-01-01", "1950-01-02", "1951-01-01", "1951-01-02"],
            f"{test_parameter}_model1": [1, 2, 3, 4],
            f"{test_parameter}_model2": [30, 80, 50, 40],
            f"{test_parameter}_model3": [10, 18, 17, 19],
        }
    }

    expected = pd.DataFrame(
        {
            ("year", ""): {0: 1950, 1: 1951},
            (f"{test_parameter}_model1", "mean"): {0: 1.5, 1: 3.5},
            (f"{test_parameter}_model1", "std"): {
                0: 0.7071067811865476,
                1: 0.7071067811865476,
            },
            (f"{test_parameter}_model2", "mean"): {0: 55.0, 1: 45.0},
            (f"{test_parameter}_model2", "std"): {
                0: 35.35533905932738,
                1: 7.0710678118654755,
            },
            (f"{test_parameter}_model3", "mean"): {0: 14.0, 1: 18.0},
            (f"{test_parameter}_model3", "std"): {
                0: 5.656854249492381,
                1: 1.4142135623730951,
            },
        }
    )

    pd.testing.assert_frame_equal(
        compute_yearly_mean_and_std(data), expected, check_dtype=False
    )


class MockResponse:
    def __init__(
        self, json_data: str, status_code: int, reason: str, headers: dict = {}
    ) -> None:
        self.json_data = json_data
        self.status_code = status_code
        self.reason = reason
        self.headers = headers

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise requests.exceptions.RequestException(response=self)

    def json(self) -> str:
        return self.json_data


def test_fetch_url_200(monkeypatch: MonkeyPatch) -> None:
    mock_response = Mock(
        return_value=MockResponse("mocked_response", 200, "mocked_reason")
    )
    monkeypatch.setattr("requests.get", mock_response)

    response = fetch_url("mock_url")

    assert response == "mocked_response"


def test_fetch_url_404(monkeypatch: MonkeyPatch) -> None:
    mock_response = Mock(
        return_value=MockResponse("mocked_response", 404, "mocked_reason")
    )
    monkeypatch.setattr("requests.get", mock_response)

    with pytest.raises(requests.RequestException):
        fetch_url("mock_url")


def test_fetch_url_429(monkeypatch: MonkeyPatch, caplog: LogCaptureFixture) -> None:
    mock_response = Mock(
        return_value=MockResponse("mocked_response", 429, "mocked_reason")
    )
    monkeypatch.setattr("requests.get", mock_response)
    monkeypatch.setattr("time.sleep", lambda x: None)

    fetch_url("mock_url")

    expected_logs = [
        "Error 429: mocked_reason. Retrying after 1 seconds.",
        "Error 429: mocked_reason. Retrying after 2 seconds.",
        "Error 429: mocked_reason. Data fetching unsuccessful.",
    ]

    assert [r.msg for r in caplog.records] == expected_logs


def test_fetch_url_not_200_404_429(monkeypatch: MonkeyPatch) -> None:
    mock_response = Mock(
        return_value=MockResponse("mocked_response", 500, "mocked_reason")
    )
    monkeypatch.setattr("requests.get", mock_response)

    response = fetch_url("mock_url")

    assert response is None


def test_main():
    pass
