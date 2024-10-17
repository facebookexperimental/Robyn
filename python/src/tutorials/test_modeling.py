# test_modeling.py

import os
import sys

import sys
import os
import numpy as np
import pandas as pd
import pyreadr
from typing import Dict, Any
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.model_executor import ModelExecutor
from robyn.modeling.ridge_model_builder import RidgeModelBuilder
from robyn.modeling.entities.enums import NevergradAlgorithm, Models
from robyn.modeling.feature_engineering import FeaturizedMMMData, FeatureEngineering


def load_data() -> Dict[str, pd.DataFrame]:
    base_path = os.getenv("ROBYN_BASE_PATH")
    if not base_path:
        raise EnvironmentError("Please set the ROBYN_BASE_PATH environment variable")

    simulated_weekly_path = os.path.join(base_path, "dt_simulated_weekly.RData")
    prophet_holidays_path = os.path.join(base_path, "dt_prophet_holidays.RData")

    result = pyreadr.read_r(simulated_weekly_path)
    dt_simulated_weekly = result["dt_simulated_weekly"]
    result_holidays = pyreadr.read_r(prophet_holidays_path)
    dt_prophet_holidays = result_holidays["dt_prophet_holidays"]

    return {"dt_simulated_weekly": dt_simulated_weekly, "dt_prophet_holidays": dt_prophet_holidays}


data = load_data()


def setup_mmm_data(data: Dict[str, pd.DataFrame]) -> MMMData:
    dt_simulated_weekly = data["dt_simulated_weekly"]

    mmm_data_spec = MMMData.MMMDataSpec(
        dep_var="revenue",
        dep_var_type="revenue",
        date_var="DATE",
        context_vars=["competitor_sales_B", "events"],
        paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
        paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],
        organic_vars=["newsletter"],
        window_start="2016-01-01",
        window_end="2018-12-31",
    )

    return MMMData(data=dt_simulated_weekly, mmmdata_spec=mmm_data_spec)


mmm_data = setup_mmm_data(data)
mmm_data.data.head()


def setup_hyperparameters() -> Hyperparameters:
    return Hyperparameters(
        hyperparameters={
            "facebook_S_alphas": [0.5, 3],
            "facebook_S_gammas": [0.3, 1],
            "facebook_S_thetas": [0, 0.3],
            "print_S_alphas": [0.5, 3],
            "print_S_gammas": [0.3, 1],
            "print_S_thetas": [0.1, 0.4],
            "tv_S_alphas": [0.5, 3],
            "tv_S_gammas": [0.3, 1],
            "tv_S_thetas": [0.3, 0.8],
            "search_S_alphas": [0.5, 3],
            "search_S_gammas": [0.3, 1],
            "search_S_thetas": [0, 0.3],
            "ooh_S_alphas": [0.5, 3],
            "ooh_S_gammas": [0.3, 1],
            "ooh_S_thetas": [0.1, 0.4],
            "newsletter_alphas": [0.5, 3],
            "newsletter_gammas": [0.3, 1],
            "newsletter_thetas": [0.1, 0.4],
        },
        adstock="geometric",
        lambda_=0.0,
        train_size=[0.5, 0.8],
    )


# Setup Hyperparameters
hyperparameters = setup_hyperparameters()
print("Hyperparameters setup complete.")

# Create HolidaysData object
holidays_data = HolidaysData(
    dt_holidays=data["dt_prophet_holidays"],
    prophet_vars=["trend", "season", "holiday"],
    prophet_country="DE",
    prophet_signs=["default", "default", "default"],
)
# Setup FeaturizedMMMData
feature_engineering = FeatureEngineering(mmm_data, hyperparameters, holidays_data)

featurized_mmm_data = feature_engineering.perform_feature_engineering()

print("Shape of dt_mod:", featurized_mmm_data.dt_mod.shape)
featurized_mmm_data.dt_mod.head()

# Setup ModelExecutor
model_executor = ModelExecutor(
    mmmdata=mmm_data,
    holidays_data=holidays_data,
    hyperparameters=hyperparameters,
    calibration_input=None,  # Add calibration input if available
    featurized_mmm_data=featurized_mmm_data,
)

# Setup TrialsConfig
trials_config = TrialsConfig(iterations=2000, trials=5)  # Set to the number of cores you want to use

print(
    f">>> Starting {trials_config.trials} trials with {trials_config.iterations} iterations each using {NevergradAlgorithm.TWO_POINTS_DE.value} nevergrad algorithm on x cores..."
)

# Run the model

output_models = model_executor.model_run(
    trials_config=trials_config,
    ts_validation=True,
    add_penalty_factor=False,
    rssd_zero_penalty=True,
    cores=8,
    nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
    intercept=True,
    intercept_sign="non_negative",
    model_name=Models.RIDGE,
)
print("Model training complete.")
