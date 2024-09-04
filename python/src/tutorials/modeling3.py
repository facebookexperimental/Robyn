import sys

import numpy as np
import pandas as pd
import pyreadr

sys.path.append("/Users/yijuilee/project_robyn/modelling/Robyn/python/src")

from robyn.modeling.entities.mmmdata_collection import MMMDataCollection
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.mmm_model_executor import MMMModelExecutor

sys.path.append("/Users/yijuilee/project_robyn/modelling/Robyn/python/")
from tutorials.utils.python_helper import render_spendexposure


def main():
    # Load the RData file
    # Step 1: Load data
    result = pyreadr.read_r(
        "/Users/yijuilee/project_robyn/modelling/Robyn/R/data/dt_simulated_weekly.RData"
    )
    dt_simulated_weekly = result["dt_simulated_weekly"]
    print(dt_simulated_weekly.head())
    result_holidays = pyreadr.read_r(
        "/Users/yijuilee/project_robyn/modelling/Robyn/R/data/dt_prophet_holidays.RData"
    )
    dt_prophet_holidays = result_holidays["dt_prophet_holidays"]
    print(dt_prophet_holidays.head())

    # Step 2: Model specification
    mmmdata_collection = MMMDataCollection()

    # 2-1: Specify input variables
    mmmdata_collection.set_input_data(dt_simulated_weekly)
    mmmdata_collection.set_holidays_data(dt_prophet_holidays)
    mmmdata_collection.set_date_variable("DATE")
    mmmdata_collection.set_dependent_variable("revenue", "revenue")
    mmmdata_collection.set_prophet_variables(
        prophet_vars=["trend", "season", "holiday"], prophet_country="DE"
    )
    mmmdata_collection.set_context_variables(
        context_vars=["competitor_sales_B", "events"],
        context_signs=["default", "default"],
    )
    mmmdata_collection.set_paid_media_variables(
        paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
        paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],
        paid_media_signs=["positive", "positive", "positive", "positive", "positive"],
    )
    mmmdata_collection.set_organic_variables(
        organic_vars=["newsletter"], organic_signs=["positive"]
    )
    mmmdata_collection.set_window("2016-01-01", "2018-12-31")
    mmmdata_collection.set_adstock("geometric")

    # 2-2 and 2-3: Define and add hyperparameters
    hyperparameters = {
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
        "train_size": [0.5, 0.8],
    }
    mmmdata_collection.set_hyperparameters(hyperparameters)
    # After setting all variables
    print("MMMDataCollection state before prepare_modeling_data:")
    print(f"dt_input shape: {mmmdata_collection.dt_input.shape}")
    print(f"dt_input columns: {mmmdata_collection.dt_input.columns}")
    print(f"paid_media_vars: {mmmdata_collection.paid_media_vars}")
    print(f"paid_media_spends: {mmmdata_collection.paid_media_spends}")
    print(f"exposure_vars: {mmmdata_collection.exposure_vars}")

    # Prepare modeling data
    mmmdata_collection.prepare_modeling_data()

    print("\nMMMDataCollection state after prepare_modeling_data:")
    print(f"dt_mod shape: {mmmdata_collection.dt_mod.shape}")
    print(f"dt_mod columns: {mmmdata_collection.dt_mod.columns}")
    print(f"dt_modRollWind shape: {mmmdata_collection.dt_modRollWind.shape}")
    print(f"dt_modRollWind columns: {mmmdata_collection.dt_modRollWind.columns}")

    # Render spend exposure plots
    render_spendexposure(mmmdata_collection)

    # 2-4: Model calibration (optional, commented out for now)
    # calibration_input = pd.DataFrame({
    #     "channel": ["facebook_S", "tv_S", "facebook_S+search_S", "newsletter"],
    #     "liftStartDate": ["2018-05-01", "2018-04-03", "2018-07-01", "2017-12-01"],
    #     "liftEndDate": ["2018-06-10", "2018-06-03", "2018-07-20", "2017-12-31"],
    #     "liftAbs": [400000, 300000, 700000, 200],
    #     "spend": [421000, 7100, 350000, 0],
    #     "confidence": [0.85, 0.8, 0.99, 0.95],
    #     "metric": ["revenue", "revenue", "revenue", "revenue"],
    #     "calibration_scope": ["immediate", "immediate", "immediate", "immediate"]
    # })
    # mmmdata_collection.set_calibration_input(calibration_input)

    # Step 3: Build initial model
    model_executor = MMMModelExecutor()
    trials_config = TrialsConfig(
        num_trials=5,
        num_iterations_per_trial=2000,
        timeseries_validation=True,
        add_penalty_factor=False,
        rssd_zero_penalty=True,  # Add this line
    )

    model_output = model_executor.model_run(
        mmmdata_collection=mmmdata_collection,
        trials_config=trials_config,
        seed=123,
        quiet=True,
        nevergrad_algo="TwoPointsDE",
        intercept=True,
        intercept_sign="non_negative",
    )

    # Print some results
    print("Model run completed.")
    print(f"Number of trials: {len(model_output.model_output.trials)}")
    print(
        f"Best NRMSE: {min(trial.nrmse for trial in model_output.model_output.trials)}"
    )

    # Further steps like model selection, budget allocation, etc. would go here


if __name__ == "__main__":
    main()
