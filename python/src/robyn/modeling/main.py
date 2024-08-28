# main.py

import pandas as pd
from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.entities.modelrun_trials_config import TrialsConfig
from robyn.mmm_model_executor import MMMModelExecutor


def load_simulated_data():
    # Replace this with actual data loading logic
    return pd.DataFrame(
        {
            "DATE": pd.date_range(start="2016-01-01", periods=100, freq="W"),
            "revenue": np.random.rand(100) * 1000,
            "tv_S": np.random.rand(100) * 100,
            "ooh_S": np.random.rand(100) * 50,
            "print_S": np.random.rand(100) * 30,
            "facebook_S": np.random.rand(100) * 80,
            "search_S": np.random.rand(100) * 60,
        }
    )


def load_prophet_holidays():
    # Replace this with actual holiday data loading logic
    return pd.DataFrame(
        {
            "ds": pd.date_range(start="2016-01-01", periods=10, freq="M"),
            "holiday": [
                "New Year's Day",
                "Valentine's Day",
                "Easter",
                "Mother's Day",
                "Father's Day",
                "Independence Day",
                "Labor Day",
                "Halloween",
                "Thanksgiving",
                "Christmas",
            ],
            "country": ["US"] * 10,
            "year": [2016] * 10,
        }
    )


def main():
    # Load and prepare data
    dt_simulated_weekly = load_simulated_data()
    dt_prophet_holidays = load_prophet_holidays()

    mmmdata_collection = MMMDataCollection(
        dt_input=dt_simulated_weekly,
        dt_holidays=dt_prophet_holidays,
        date_var="DATE",
        dep_var="revenue",
        dep_var_type="revenue",
        prophet_vars=["trend", "season", "holiday"],
        prophet_country="US",
        context_vars=["competitor_sales_B", "events"],
        paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
        paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],
        organic_vars=["newsletter"],
        factor_vars=["events"],
        window_start="2016-11-23",
        window_end="2018-08-22",
        adstock="geometric",
    )

    # Configure model run
    trials_config = TrialsConfig(
        num_trials=5,
        num_iterations_per_trial=2000,
        timeseries_validation=True,
        add_penalty_factor=False,
    )

    # Initialize and run the model
    model_executor = MMMModelExecutor()
    output = model_executor.model_run(
        mmmdata_collection=mmmdata_collection,
        trials_config=trials_config,
        ts_validation=True,
        add_penalty_factor=False,
        refresh=False,
        seed=123,
        quiet=False,
        cores=None,
        nevergrad_algo="TwoPointsDE",
        intercept=True,
        intercept_sign="non_negative",
    )

    # Process and display results
    print(output)


if __name__ == "__main__":
    main()
