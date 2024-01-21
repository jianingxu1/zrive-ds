import requests
import pandas as pd
import time
import matplotlib.pyplot as plt

API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

START_DATE = "1950-01-01"
END_DATE = "2050-01-01"

MAX_SECONDS_TO_WAIT = 3
MAX_ATTEMPTS = 5


def fetch_url(url, params=None):
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        response = requests.get(url, params)
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            if error.response.status_code == 429 and attempts < MAX_ATTEMPTS:
                retry_after = int(response.headers.get("Retry-After", 1))
                time.sleep(retry_after)
                continue
            raise


def get_data_meteo_api(city_name):
    params = {
        "latitude": COORDINATES[city_name]["latitude"],
        "longitude": COORDINATES[city_name]["longitude"],
        "daily": VARIABLES,
        "start_date": START_DATE,
        "end_date": END_DATE,
    }
    # TODO schema validation
    return fetch_url(API_URL, params)


def get_mean_and_std_by_year_dataframe(data: dict):
    df = pd.DataFrame(data)

    # Convert 'time' to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Extract year and create a new column for it
    df["year"] = df["time"].dt.year

    # Group by year and calculate mean and standard deviation
    statistics_per_year = df.groupby("year").agg(["mean", "std"])
    return statistics_per_year


def main():
    city_data = {}
    for city in COORDINATES:
        try:
            response = get_data_meteo_api(city)
            city_data[city] = response
            print(f"{city} data fetched successfully!")
        except requests.exceptions.RequestException as error:
            print(
                f"Error {error.response.status_code}: {error.response.reason}."
                + f"Could not fetch data for {city}."
            )

    # Transform data into dataframes and perform mean and std calculations
    city_dataframe = {}
    for city_name in city_data:
        city_dataframe[city_name] = get_mean_and_std_by_year_dataframe(
            city_data[city_name]["daily"]
        )


if __name__ == "__main__":
    main()
