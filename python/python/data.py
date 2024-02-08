# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os

import pandas as pd

### Manually Generated


def dt_simulated_weekly():
    csv_file = os.getcwd() + "/../data/dt_simulated_weekly.csv"
    return pd.read_csv(csv_file)


def dt_prophet_holidays():
    csv_file = os.getcwd() + "/../data/dt_prophet_holidays.csv"
    return pd.read_csv(csv_file)
