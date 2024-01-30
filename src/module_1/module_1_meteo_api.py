import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"


def fetch_url(url: str, params: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch data from an API given a URL and optional parameters.
    """
    MAX_ATTEMPTS = 3
    exponential_backoff = 1
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(url, params)
            response.raise_for_status()
            logging.info("Data fetched successfully!")
            return response.json()

        except requests.exceptions.RequestException as error:
            if error.response.status_code == 404:
                logging.error(f"404 Not Found: {url}")
                raise
            elif error.response.status_code == 429 and attempt != MAX_ATTEMPTS - 1:
                retry_after = int(
                    response.headers.get("Retry-After", exponential_backoff)
                )
                exponential_backoff *= 2
                logging.warning(
                    f"Error {error.response.status_code}: {error.response.reason}. "
                    + f"Retrying after {retry_after} seconds."
                )
                time.sleep(retry_after)
                continue
            logging.warning(
                f"Error {error.response.status_code}: {error.response.reason}. "
                + "Data fetching unsuccessful."
            )
            return None


@dataclass
class MeteoApiRequestParameters:
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    climate_models: List[str]


def fetch_data_meteo_api(
    requestParams: MeteoApiRequestParameters,
) -> Optional[Dict[str, Any]]:
    """
    Fetch weather data from meteo API given a specified location,
    time range in "YYYY-MM-DD" format and climate models.
    """
    params = {
        "latitude": requestParams.latitude,
        "longitude": requestParams.longitude,
        "daily": VARIABLES,
        "start_date": requestParams.start_date,
        "end_date": requestParams.end_date,
        "models": requestParams.climate_models,
    }

    return fetch_url(API_URL, params)


def compute_yearly_mean_and_std(data: Dict[str, Any]) -> pd.DataFrame:
    daily_data = data.get("daily", {})

    df = pd.DataFrame(daily_data)

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df = df.drop("time", axis=1)

    yearly_mean_and_std_df = df.groupby("year").agg(["mean", "std"])

    return yearly_mean_and_std_df.reset_index()


def plot_mean_and_std(
    climate_models: List[str],
    parameter: str,
    city: str,
    climate_data: Dict[str, Any],
    df: pd.DataFrame,
) -> None:
    plt.figure()
    plt.style.use("ggplot")
    plt.subplots(figsize=(10, 6))
    for model in climate_models:
        column = f"{parameter}_{model}"
        plt.errorbar(
            df["year"],
            df[column]["mean"],
            yerr=df[column]["std"],
            linestyle="-",
            elinewidth=0.5,
            marker="o",
            markersize=2,
            label=f"{model}",
            capsize=1,
        )
        plt.fill_between(
            df["year"],
            df[column]["mean"] - df[column]["std"],
            df[column]["mean"] + df[column]["std"],
            alpha=0.3,
        )

    parameter_unit = climate_data["daily_units"][f"{parameter}_{climate_models[0]}"]
    parameter_name = parameter.replace("_", " ").capitalize()
    plt.xlabel("Year")
    plt.ylabel(f"{parameter_name} ({parameter_unit})")
    plt.title(
        "Evolution of the Mean and Standard Deviation of\n "
        + f"{parameter_name} using different climate models for {city}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    start_date = "1950-01-01"
    end_date = "2050-01-01"
    climate_models = [
        "CMCC_CM2_VHR4",
        "FGOALS_f3_H",
        "HiRAM_SIT_HR",
        "MRI_AGCM3_2_S",
        "EC_Earth3P_HR",
        "MPI_ESM1_2_XR",
        "NICAM16_8S",
    ]
    parameter_to_plot = "temperature_2m_mean"

    for city, coordinates in COORDINATES.items():
        latitude, longitude = coordinates["latitude"], coordinates["longitude"]
        requestParams = MeteoApiRequestParameters(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            climate_models=climate_models,
        )
        data = fetch_data_meteo_api(requestParams)

        if data is None:
            continue

        processed_data = compute_yearly_mean_and_std(data)

        plot_mean_and_std(climate_models, parameter_to_plot, city, data, processed_data)


if __name__ == "__main__":
    main()
