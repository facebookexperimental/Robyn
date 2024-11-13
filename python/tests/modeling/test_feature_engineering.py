# # pyre-strict

# import pytest
# import pandas as pd
# import numpy as np
# from unittest.mock import MagicMock, patch
# from pathlib import Path
# from prophet import Prophet
# from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData
# from robyn.data.entities.mmmdata import MMMData
# from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
# from robyn.data.entities.holidays_data import HolidaysData
# from robyn.data.entities.enums import AdstockType


# @pytest.fixture(scope="session")
# def test_data():
#     resources_path = Path("python/src/robyn/tutorials/resources")
#     dt_simulated = pd.read_csv(resources_path / "dt_simulated_weekly.csv", parse_dates=["DATE"])
#     dt_holidays = pd.read_csv(resources_path / "dt_prophet_holidays.csv", parse_dates=["ds"])
#     return dt_simulated, dt_holidays


# @pytest.fixture
# def feature_engineering_setup(test_data):
#     dt_simulated, dt_holidays = test_data

#     mmm_data = MagicMock(spec=MMMData)
#     hyperparameters = MagicMock(spec=Hyperparameters)
#     holidays_data = MagicMock(spec=HolidaysData)

#     mmm_data.mmmdata_spec = MagicMock()
#     mmm_data.mmmdata_spec.date_var = "DATE"
#     mmm_data.mmmdata_spec.dep_var = "revenue"
#     mmm_data.mmmdata_spec.paid_media_spends = ["facebook_S"]
#     mmm_data.mmmdata_spec.paid_media_vars = ["facebook_I"]
#     mmm_data.mmmdata_spec.window_start = pd.Timestamp("2015-11-23")
#     mmm_data.mmmdata_spec.window_end = pd.Timestamp("2015-12-21")
#     mmm_data.mmmdata_spec.interval_type = "week"
#     mmm_data.mmmdata_spec.day_interval = 7
#     mmm_data.mmmdata_spec.context_vars = ["competitor_sales_B"]
#     mmm_data.mmmdata_spec.organic_vars = ["events", "newsletter"]
#     mmm_data.mmmdata_spec.factor_vars = []
#     mmm_data.data = dt_simulated.copy()
#     mmm_data.mmmdata_spec.get_paid_media_var = MagicMock(return_value="facebook_I")

#     holidays_data.prophet_vars = ["trend", "season", "holiday"]
#     holidays_data.dt_holidays = dt_holidays.copy()
#     holidays_data.prophet_country = "AD"

#     hyperparameters.adstock = AdstockType.GEOMETRIC

#     feature_engineering = FeatureEngineering(mmm_data, hyperparameters, holidays_data)

#     return feature_engineering, mmm_data, hyperparameters, holidays_data, dt_simulated, dt_holidays


# def test_initialization(feature_engineering_setup):
#     """Test initialization of FeatureEngineering class"""
#     feature_engineering, mmm_data, hyperparameters, holidays_data, _, _ = feature_engineering_setup

#     assert isinstance(feature_engineering.mmm_data, MMMData)
#     assert isinstance(feature_engineering.hyperparameters, Hyperparameters)
#     assert isinstance(feature_engineering.holidays_data, HolidaysData)
#     assert feature_engineering.logger is not None


# def test_perform_feature_engineering(feature_engineering_setup):
#     """Test the complete feature engineering pipeline"""
#     feature_engineering, *_ = feature_engineering_setup
#     result = feature_engineering.perform_feature_engineering(quiet=True)

#     assert isinstance(result, FeaturizedMMMData)
#     assert isinstance(result.dt_mod, pd.DataFrame)
#     assert isinstance(result.dt_modRollWind, pd.DataFrame)
#     assert isinstance(result.modNLS, dict)
#     assert "results" in result.modNLS
#     assert "yhat" in result.modNLS
#     assert "plots" in result.modNLS


# def test_prepare_data(feature_engineering_setup):
#     """Test data preparation method"""
#     feature_engineering, _, _, _, dt_simulated, _ = feature_engineering_setup
#     result = feature_engineering._prepare_data()

#     assert "ds" in result.columns
#     assert "revenue" in result.columns
#     assert "competitor_sales_B" in result.columns
#     assert len(result) == len(dt_simulated)

#     if not pd.api.types.is_datetime64_any_dtype(result["ds"]):
#         result["ds"] = pd.to_datetime(result["ds"])
#     assert pd.api.types.is_datetime64_any_dtype(result["ds"])


# def test_create_rolling_window_data_with_both_windows(feature_engineering_setup):
#     """Test rolling window creation with both start and end dates"""
#     feature_engineering, mmm_data, _, _, dt_simulated, _ = feature_engineering_setup
#     dt_transform = pd.DataFrame({"ds": pd.to_datetime(dt_simulated["DATE"]), "revenue": dt_simulated["revenue"]})
#     result = feature_engineering._create_rolling_window_data(dt_transform)

#     assert all(result["ds"] >= mmm_data.mmmdata_spec.window_start)
#     assert all(result["ds"] <= mmm_data.mmmdata_spec.window_end)


# def test_create_rolling_window_data_with_no_windows(feature_engineering_setup):
#     """Test rolling window creation with no window constraints"""
#     feature_engineering, mmm_data, _, _, dt_simulated, _ = feature_engineering_setup
#     mmm_data.mmmdata_spec.window_start = None
#     mmm_data.mmmdata_spec.window_end = None

#     dt_transform = pd.DataFrame({"ds": pd.to_datetime(dt_simulated["DATE"]), "revenue": dt_simulated["revenue"]})
#     result = feature_engineering._create_rolling_window_data(dt_transform)
#     assert len(result) == len(dt_transform)


# def test_create_rolling_window_invalid_dates(feature_engineering_setup):
#     """Test rolling window creation with invalid date ranges"""
#     feature_engineering, mmm_data, _, _, dt_simulated, _ = feature_engineering_setup
#     dt_transform = pd.DataFrame({"ds": pd.to_datetime(dt_simulated["DATE"]), "revenue": dt_simulated["revenue"]})

#     # Test start after end
#     mmm_data.mmmdata_spec.window_start = pd.Timestamp("2015-12-31")
#     mmm_data.mmmdata_spec.window_end = pd.Timestamp("2015-01-01")

#     with pytest.raises(ValueError):
#         feature_engineering._create_rolling_window_data(dt_transform)


# def test_calculate_media_cost_factor(feature_engineering_setup):
#     """Test media cost factor calculation"""
#     feature_engineering, _, _, _, dt_simulated, _ = feature_engineering_setup
#     dt_input = dt_simulated[["facebook_S"]].copy()
#     result = feature_engineering._calculate_media_cost_factor(dt_input)

#     assert len(result) == 1  # One media channel
#     assert pytest.approx(result.sum()) == 1.0  # Should sum to 1
#     assert all(0 <= x <= 1 for x in result)  # Values between 0 and 1


# def test_run_models(feature_engineering_setup):
#     """Test model running for media variables"""
#     feature_engineering, _, _, _, dt_simulated, _ = feature_engineering_setup
#     dt_modRollWind = dt_simulated[["facebook_S", "facebook_I"]].copy()
#     result = feature_engineering._run_models(dt_modRollWind, 1.0)

#     assert "results" in result
#     assert "yhat" in result
#     assert "plots" in result


# def test_hill_function(feature_engineering_setup):
#     """Test Hill function transformation"""
#     feature_engineering, _, _, _, _, _ = feature_engineering_setup
#     x = np.array([1, 2, 3, 4, 5])
#     alpha = 2
#     gamma = 3
#     result = feature_engineering._hill_function(x, alpha, gamma)

#     assert len(result) == len(x)
#     assert all(0 <= r <= 1 for r in result)


# @pytest.mark.parametrize(
#     "adstock_type,params",
#     [
#         (AdstockType.GEOMETRIC, {"thetas": [0.5]}),
#         (AdstockType.WEIBULL_CDF, {"shapes": [1.0], "scales": [1.0]}),
#         (AdstockType.WEIBULL_PDF, {"shapes": [1.0], "scales": [1.0]}),
#     ],
# )
# def test_apply_adstock(feature_engineering_setup, adstock_type, params):
#     """Test different adstock transformations"""
#     feature_engineering, _, hyperparameters, _, _, _ = feature_engineering_setup
#     x = pd.Series([1, 2, 3, 4, 5])
#     params_mock = MagicMock(spec=ChannelHyperparameters)
#     for key, value in params.items():
#         setattr(params_mock, key, value)

#     hyperparameters.adstock = adstock_type
#     result = feature_engineering._apply_adstock(x, params_mock)
#     assert len(result) == len(x)


# def test_apply_adstock_invalid(feature_engineering_setup):
#     """Test invalid adstock type handling"""
#     feature_engineering, _, hyperparameters, _, _, _ = feature_engineering_setup
#     x = pd.Series([1, 2, 3, 4, 5])
#     params = MagicMock(spec=ChannelHyperparameters)
#     hyperparameters.adstock = "INVALID"

#     with pytest.raises(ValueError):
#         feature_engineering._apply_adstock(x, params)


# @pytest.mark.parametrize(
#     "prophet_vars",
#     [
#         ["trend", "season", "holiday"],
#         ["trend", "season"],
#         ["trend", "holiday"],
#     ],
# )
# def test_prophet_decomposition(feature_engineering_setup, prophet_vars):
#     """Test Prophet decomposition with different variables"""
#     feature_engineering, _, _, holidays_data, dt_simulated, _ = feature_engineering_setup
#     holidays_data.prophet_vars = prophet_vars

#     dt_mod = pd.DataFrame(
#         {
#             "ds": pd.to_datetime(dt_simulated["DATE"]),
#             "dep_var": dt_simulated["revenue"],
#             "facebook_S": dt_simulated["facebook_S"],
#             "competitor_sales_B": dt_simulated["competitor_sales_B"],
#             "events": dt_simulated["events"],
#             "newsletter": dt_simulated["newsletter"],
#         }
#     )

#     with patch("prophet.Prophet") as mock_prophet:
#         mock_prophet_instance = MagicMock()
#         mock_prophet_instance.fit.return_value = None
#         prophet_output = pd.DataFrame(
#             {
#                 "ds": dt_mod["ds"],
#                 "trend": np.ones(len(dt_mod)),
#                 "yearly": np.zeros(len(dt_mod)),
#                 "holidays": np.zeros(len(dt_mod)),
#                 "yhat": np.ones(len(dt_mod)),
#             }
#         )
#         mock_prophet_instance.predict.return_value = prophet_output
#         mock_prophet.return_value = mock_prophet_instance

#         result = feature_engineering._prophet_decomposition(dt_mod)

#         expected_columns = [var for var in prophet_vars if var != "season"] + (
#             ["season"] if "season" in prophet_vars else []
#         )
#         for col in expected_columns:
#             assert col in result.columns


# @pytest.mark.parametrize("interval_type", ["day", "month"])
# def test_set_holidays(feature_engineering_setup, interval_type):
#     """Test holiday setting for different interval types"""
#     feature_engineering, _, _, _, _, dt_holidays = feature_engineering_setup

#     if interval_type == "day":
#         dt_transform = pd.DataFrame({"ds": pd.date_range(start="2015-01-01", end="2015-01-10")})
#     else:
#         dt_transform = pd.DataFrame({"ds": pd.date_range(start="2015-01-01", end="2015-12-01", freq="MS")})

#     result = feature_engineering._set_holidays(dt_transform, dt_holidays, interval_type)

#     assert all(col in result.columns for col in ["holiday", "ds", "country"])


# def test_set_holidays_invalid_interval(feature_engineering_setup):
#     """Test invalid interval type handling"""
#     feature_engineering, _, _, _, _, dt_holidays = feature_engineering_setup
#     dt_transform = pd.DataFrame({"ds": pd.date_range(start="2015-01-01", end="2015-01-10")})

#     with pytest.raises(ValueError):
#         feature_engineering._set_holidays(dt_transform, dt_holidays, "invalid")
