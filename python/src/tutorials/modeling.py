import sys
import numpy as np
import pandas as pd
import pyreadr
from typing import Dict, Any

# Adjust these paths as needed
sys.path.append("/Users/yijuilee/project_robyn/robynpy_interfaces/Robyn/python/src")

from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.model_executor import ModelExecutor
from robyn.modeling.ridge_model_builder import RidgeModelBuilder
from robyn.modeling.entities.enums import NevergradAlgorithm, Models
from robyn.modeling.feature_engineering import FeaturizedMMMData, FeatureEngineering

from tutorials.utils.python_helper import render_spendexposure


def load_data() -> Dict[str, pd.DataFrame]:
    # Load the RData files
    result = pyreadr.read_r("/Users/yijuilee/project_robyn/modelling/Robyn/R/data/dt_simulated_weekly.RData")
    dt_simulated_weekly = result["dt_simulated_weekly"]

    result_holidays = pyreadr.read_r("/Users/yijuilee/project_robyn/modelling/Robyn/R/data/dt_prophet_holidays.RData")
    dt_prophet_holidays = result_holidays["dt_prophet_holidays"]

    return {"dt_simulated_weekly": dt_simulated_weekly, "dt_prophet_holidays": dt_prophet_holidays}


def setup_mmm_data(data: Dict[str, pd.DataFrame]) -> MMMData:
    dt_simulated_weekly = data["dt_simulated_weekly"]

    mmm_data_spec = MMMData.MMMDataSpec(
        dep_var="revenue",
        dep_var_type="revenue",
        date_var="DATE",
        prophet_vars=["trend", "season", "holiday"],
        prophet_country="DE",
        context_vars=["competitor_sales_B", "events"],
        paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
        paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],
        organic_vars=["newsletter"],
        window_start="2016-01-01",
        window_end="2018-12-31",
    )

    return MMMData(data=dt_simulated_weekly, mmmdata_spec=mmm_data_spec)


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


def main():
    # Load data
    data = load_data()
    print("Data loaded successfully.")

    # Setup MMMData
    mmm_data = setup_mmm_data(data)
    print("MMMData setup complete.")

    # Setup HolidaysData
    holidays_data = HolidaysData(
        dt_holidays=data["dt_prophet_holidays"], prophet_vars=["trend", "season", "holiday"], prophet_country="DE"
    )
    print("HolidaysData setup complete.")

    # Setup Hyperparameters
    hyperparameters = setup_hyperparameters()
    print("Hyperparameters setup complete.")

    # Setup FeaturizedMMMData
    feature_engineering = FeatureEngineering(mmm_data, hyperparameters)
    featurized_mmm_data = feature_engineering.perform_feature_engineering()
    print("Feature engineering complete.")

    print("FeaturizedMMMData, dt_mod:", featurized_mmm_data.dt_mod.shape)
    print("FeaturizedMMMData, dt_mod:", featurized_mmm_data.dt_mod.columns)
    print("FeaturizedMMMData, dt_modRollWind:", featurized_mmm_data.dt_modRollWind.shape)
    print("FeaturizedMMMData, dt_modRollWind:", featurized_mmm_data.dt_modRollWind.columns)
    print("FeaturizedMMMData, modNLS:", featurized_mmm_data.modNLS)

    return
    # Setup ModelExecutor
    model_executor = ModelExecutor(
        mmm_data,
        holidays_data,
        hyperparameters,
        None,  # Calibration input is None for this example
        featurized_mmm_data,
    )
    print("ModelExecutor setup complete.")

    # Setup TrialsConfig
    trials_config = TrialsConfig(trials=5, iterations=2000)

    # Run the model
    model_outputs = model_executor.model_run(
        ts_validation=True,
        add_penalty_factor=False,
        seed=123,
        trials_config=trials_config,
        rssd_zero_penalty=True,
        nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
        intercept=True,
        intercept_sign="non_negative",
        model_name=Models.RIDGE,
    )

    # Print results
    print("Model run completed.")
    print(f"Number of trials: {len(model_outputs.trials)}")
    print(f"Best NRMSE: {min(trial.nrmse for trial in model_outputs.trials)}")

    # Render spend exposure plots
    render_spendexposure(mmm_data)


if __name__ == "__main__":
    main()
