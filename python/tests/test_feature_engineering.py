import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import (
    DependentVarType,
    AdstockType,
    ProphetVariableType,
    PaidMediaSigns,
    ProphetSigns,
)


@pytest.fixture
def sample_data():
    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    df = pd.DataFrame(
        {
            "date": date_range,
            "sales": np.random.randint(1000, 5000, size=len(date_range)),
            "tv_spend": np.random.randint(500, 2000, size=len(date_range)),
            "radio_spend": np.random.randint(200, 1000, size=len(date_range)),
            "tv_impressions": np.random.randint(10000, 50000, size=len(date_range)),
            "radio_impressions": np.random.randint(5000, 20000, size=len(date_range)),
        }
    )
    return df


@pytest.fixture
def mmm_data(sample_data):
    mmm_data_spec = MMMData.MMMDataSpec(
        dep_var="sales",
        dep_var_type=DependentVarType.REVENUE,
        date_var="date",
        paid_media_spends=["tv_spend", "radio_spend"],
        paid_media_vars=["tv_impressions", "radio_impressions"],
        paid_media_signs=[PaidMediaSigns.POSITIVE, PaidMediaSigns.POSITIVE],
    )
    return MMMData(sample_data, mmm_data_spec)


@pytest.fixture
def hyperparameters():
    channel_hyperparameters = {
        "tv_spend": ChannelHyperparameters(
            thetas=[0.3],
            alphas=[0.5],
            gammas=[0.3],
        ),
        "radio_spend": ChannelHyperparameters(
            thetas=[0.2],
            alphas=[0.4],
            gammas=[0.2],
        ),
    }
    return Hyperparameters(hyperparameters=channel_hyperparameters, adstock=AdstockType.GEOMETRIC)


@pytest.fixture
def holidays_data():
    holidays_df = pd.DataFrame(
        {
            "ds": ["2023-01-01", "2023-12-25"],
            "holiday": ["New Year", "Christmas"],
        }
    )
    return HolidaysData(
        dt_holidays=holidays_df,
        prophet_vars=[ProphetVariableType.HOLIDAY],
        prophet_signs=[ProphetSigns.POSITIVE],
        prophet_country="US",
    )


def test_feature_engineering_initialization(mmm_data, hyperparameters, holidays_data):
    fe = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    assert isinstance(fe, FeatureEngineering)
    assert fe.mmm_data == mmm_data
    assert fe.hyperparameters == hyperparameters
    assert fe.holidays_data == holidays_data


def test_prepare_data(mmm_data, hyperparameters, holidays_data):
    fe = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    prepared_data = fe._prepare_data()
    assert isinstance(prepared_data, pd.DataFrame)
    assert "ds" in prepared_data.columns
    assert "dep_var" in prepared_data.columns
    assert len(prepared_data) == len(mmm_data.data)
    assert prepared_data["ds"].dtype == "datetime64[ns]"
    assert prepared_data["dep_var"].equals(mmm_data.data[mmm_data.mmmdata_spec.dep_var])


def test_create_rolling_window_data(mmm_data, hyperparameters, holidays_data):
    fe = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    dt_mod = fe._prepare_data()
    rolling_window_data = fe._create_rolling_window_data(dt_mod)
    assert isinstance(rolling_window_data, pd.DataFrame)
    assert len(rolling_window_data) == len(dt_mod)

    # Test with custom window
    mmm_data.mmmdata_spec.window_start = datetime(2023, 3, 1)
    mmm_data.mmmdata_spec.window_end = datetime(2023, 9, 30)
    fe_windowed = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    dt_mod_windowed = fe_windowed._prepare_data()
    rolling_window_data_windowed = fe_windowed._create_rolling_window_data(dt_mod_windowed)
    assert len(rolling_window_data_windowed) < len(dt_mod_windowed)
    assert rolling_window_data_windowed["ds"].min() >= mmm_data.mmmdata_spec.window_start
    assert rolling_window_data_windowed["ds"].max() <= mmm_data.mmmdata_spec.window_end


def test_calculate_media_cost_factor(mmm_data, hyperparameters, holidays_data):
    fe = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    dt_mod = fe._prepare_data()
    dt_modRollWind = fe._create_rolling_window_data(dt_mod)
    media_cost_factor = fe._calculate_media_cost_factor(dt_modRollWind)
    assert isinstance(media_cost_factor, pd.Series)
    assert len(media_cost_factor) == len(mmm_data.mmmdata_spec.paid_media_spends)
    assert media_cost_factor.index.tolist() == mmm_data.mmmdata_spec.paid_media_spends
    assert np.isclose(media_cost_factor.sum(), 1)  # Should sum to 1


def test_fit_spend_exposure(mmm_data, hyperparameters, holidays_data):
    fe = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    dt_mod = fe._prepare_data()
    dt_modRollWind = fe._create_rolling_window_data(dt_mod)
    media_cost_factor = fe._calculate_media_cost_factor(dt_modRollWind)

    for paid_media_var in mmm_data.mmmdata_spec.paid_media_spends:
        result = fe._fit_spend_exposure(dt_modRollWind, paid_media_var, media_cost_factor)
        assert isinstance(result, dict)
        assert "res" in result
        assert "plot" in result
        assert "yhat" in result
        assert result["res"]["channel"] == paid_media_var
        assert result["res"]["model_type"] in ["nls", "lm"]
        assert 0 <= result["res"]["rsq"] <= 1  # R-squared should be between 0 and 1


def test_perform_feature_engineering(mmm_data, hyperparameters, holidays_data):
    fe = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    result = fe.perform_feature_engineering(quiet=True)

    assert isinstance(result, FeaturizedMMMData)
    assert isinstance(result.dt_mod, pd.DataFrame)
    assert isinstance(result.dt_modRollWind, pd.DataFrame)
    assert isinstance(result.modNLS, dict)

    # Check if dt_mod contains expected columns
    expected_columns = (
        ["ds", "dep_var"] + mmm_data.mmmdata_spec.paid_media_spends + mmm_data.mmmdata_spec.paid_media_vars
    )
    assert all(col in result.dt_mod.columns for col in expected_columns)

    # Check if dt_modRollWind has the same number of rows as dt_mod
    assert len(result.dt_mod) == len(result.dt_modRollWind)

    # Check if modNLS contains entries for each paid media channel
    assert set(result.modNLS.keys()) == set(mmm_data.mmmdata_spec.paid_media_spends)

    # Check if the feature engineering process has created any new features
    assert len(result.dt_mod.columns) > len(mmm_data.data.columns)


if __name__ == "__main__":
    pytest.main()
