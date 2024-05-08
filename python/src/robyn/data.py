# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import os

import pandas as pd

### Manually Generated


def dt_simulated_weekly():
    csv_file = os.getcwd() + "../data/dt_simulated_weekly.csv"
    return pd.read_csv(csv_file)


def dt_prophet_holidays():
    csv_file = os.getcwd() + "../data/dt_prophet_holidays.csv"
    return pd.read_csv(csv_file)

# def dt_simulated_weekly():
#     csv_file = os.getcwd() + "/python/src/data/dt_simulated_weekly.csv"
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     csv_file = os.path.join(dir_path, '..', 'data', 'dt_simulated_weekly.csv')
#     return pd.read_csv(csv_file)

# def dt_prophet_holidays():
#     csv_file = os.getcwd() + "/python/src/data/dt_prophet_holidays.csv"
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     csv_file = os.path.join(dir_path, '..', 'data', 'dt_prophet_holidays.csv')
#     return pd.read_csv(csv_file)
