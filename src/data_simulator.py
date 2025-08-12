import os
import pandas as pd
import numpy as np
from datetime import datetime


class SalesDataSimulator:
    """
    Simulate synthetic daily sales data for multiple stores over a specified date range.

    This class generates sales data considering store characteristics, seasonality, holidays,
    promotions, weekend effects, competitor impact, and random noise, then can save/load the dataset.
    """

    def __init__(
        self,
        start_date="2021-01-01",
        end_date="2024-12-31",
        output_dir="data",
        filename="simulated_sales_data.csv",
        seed=42,
    ):
        """
        Initialize simulation settings including date range, output file info, and random seed.

        Args:
            start_date (str): Start date of simulation period.
            end_date (str): End date of simulation period.
            output_dir (str): Directory to save data file.
            filename (str): Filename for saved dataset CSV.
            seed (int): Random seed for reproducibility.
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.output_dir = output_dir
        self.filename = filename
        self.seed = seed

        # Metadata of stores with size and city population info
        self.stores = [
            {"store_id": 1, "store_size": "small", "city_population": 50000},
            {"store_id": 2, "store_size": "medium", "city_population": 150000},
            {"store_id": 3, "store_size": "large", "city_population": 300000},
            {"store_id": 4, "store_size": "medium", "city_population": 200000},
            {"store_id": 5, "store_size": "large", "city_population": 400000},
        ]

        # List of specific holiday dates affecting sales
        self.holidays = [
            datetime(2021, 3, 14),
            datetime(2021, 10, 18),
            datetime(2022, 4, 20),
            datetime(2022, 8, 22),
            datetime(2023, 5, 24),
            datetime(2023, 9, 26),
        ]

        # Fixed promotion periods defined as date ranges per store
        self.fixed_promotions = {
            1: [
                (pd.Timestamp("2021-08-20"), pd.Timestamp("2021-08-25")),
                (pd.Timestamp("2022-09-18"), pd.Timestamp("2022-09-24")),
                (pd.Timestamp("2023-11-19"), pd.Timestamp("2023-11-23")),
            ],
            2: [(pd.Timestamp("2021-05-10"), pd.Timestamp("2021-05-15"))],
            3: [
                (pd.Timestamp("2021-03-15"), pd.Timestamp("2021-03-20")),
                (pd.Timestamp("2022-04-14"), pd.Timestamp("2022-04-19")),
                (pd.Timestamp("2023-02-13"), pd.Timestamp("2023-02-18")),
            ],
            4: [(pd.Timestamp("2021-10-01"), pd.Timestamp("2021-10-05"))],
            5: [(pd.Timestamp("2021-09-10"), pd.Timestamp("2021-09-12"))],
        }

    def simulate(self) -> pd.DataFrame:
        """
        Simulate daily sales data for all stores over the date range.

        Returns:
            pd.DataFrame: DataFrame containing simulated sales data with features.
        """
        np.random.seed(self.seed)
        dates = pd.date_range(self.start_date, self.end_date)
        data = []

        for store in self.stores:
            store_id = store["store_id"]

            # Define base sales scaled by store size and city population
            base_sales = {
                "small": 1000,
                "medium": 2000,
                "large": 3000,
            }[store["store_size"]] * (1 + store["city_population"] / 500000)

            # Random annual growth rate between 5% and 15%
            annual_growth = np.random.uniform(0.05, 0.15)
            daily_growth = (1 + annual_growth) ** (1 / 365) - 1

            for i, date in enumerate(dates):
                month = date.month

                # Apply compounded daily growth on base sales
                sales = base_sales * (1 + daily_growth) ** i

                # Weekend sales boost on Fridays and Saturdays
                is_weekend = 1 if date.weekday() >= 4 else 0
                sales *= np.random.uniform(1.3, 1.5) if is_weekend else np.random.uniform(0.9, 1.1)

                # Seasonal uplift in November and December
                if month in [11, 12]:
                    sales *= np.random.uniform(1.4, 1.8)

                # Competitor impact: permanent sales drop for store 3 after 2022-06-15
                if store_id == 3 and date >= datetime(2022, 6, 15):
                    sales *= 0.65

                # Determine if promotion is active on current date for the store
                promotion_active = 0
                for start_promo, end_promo in self.fixed_promotions.get(store_id, []):
                    if start_promo <= date <= end_promo:
                        promotion_active = 1
                        break

                # Apply promotion sales boost
                if promotion_active == 1:
                    sales *= np.random.uniform(1.5, 2.0)

                # Add noise with normal and uniform distributions
                noise_normal = np.random.normal(1, 0.03)
                noise_uniform = np.random.uniform(0.97, 1.03)
                sales *= noise_normal * noise_uniform

                # Ensure sales are positive integers
                sales = max(1, int(round(sales)))

                # Flag if the date is a holiday
                is_holiday = 1 if date in self.holidays else 0

                # Collect entry for this date, store, and sales data
                data.append(
                    {
                        "Date": date,
                        "store_id": str(store_id),
                        "daily_sales": float(sales),
                        "promotion_active": str(promotion_active),
                        "store_size": store["store_size"],
                        "city_population": float(store["city_population"]),
                        "is_weekend": str(is_weekend),
                        "is_holiday": str(is_holiday),
                        "month": str(month),
                        "year": date.year,
                    }
                )

        df = pd.DataFrame(data)

        # Convert selected columns to categorical dtype for efficiency and consistency
        cat_cols = ["store_id", "store_size", "promotion_active", "is_weekend", "is_holiday", "month"]
        for col in cat_cols:
            df[col] = df[col].astype("category")

        # Compute time index as days since first date
        df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days

        return df

    def save(self, df: pd.DataFrame):
        """
        Save the simulated DataFrame to a CSV file in the output directory.

        Args:
            df (pd.DataFrame): DataFrame to save.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, self.filename)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")

    def load(self) -> pd.DataFrame or None:
        """
        Load the simulated sales DataFrame from the CSV file if it exists.

        Returns:
            pd.DataFrame or None: Loaded DataFrame or None if file does not exist.
        """
        path = os.path.join(self.output_dir, self.filename)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            cat_cols = ["store_id", "store_size", "promotion_active", "is_weekend", "is_holiday", "month"]
            for col in cat_cols:
                df[col] = df[col].astype(str).astype("category")
            df["daily_sales"] = df["daily_sales"].astype(float)
            df["city_population"] = df["city_population"].astype(float)
            return df
        else:
            return None
