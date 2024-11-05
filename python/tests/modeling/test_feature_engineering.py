# pyre-strict

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import AdstockType


class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        resources_path = Path("python/src/robyn/tutorials/resources")
        cls.dt_simulated = pd.read_csv(resources_path / "dt_simulated_weekly.csv", parse_dates=["DATE"])
        cls.dt_holidays = pd.read_csv(resources_path / "dt_prophet_holidays.csv", parse_dates=["ds"])

    def setUp(self):
        self.mmm_data = MagicMock(spec=MMMData)
        self.hyperparameters = MagicMock(spec=Hyperparameters)
        self.holidays_data = MagicMock(spec=HolidaysData)

        self.mmm_data.mmmdata_spec = MagicMock()
        self.mmm_data.mmmdata_spec.date_var = "DATE"
        self.mmm_data.mmmdata_spec.dep_var = "revenue"

        self.mmm_data.mmmdata_spec.paid_media_spends = ["facebook_S"]
        self.mmm_data.mmmdata_spec.paid_media_vars = ["facebook_I"]

        self.mmm_data.mmmdata_spec.window_start = pd.Timestamp("2015-11-23")
        self.mmm_data.mmmdata_spec.window_end = pd.Timestamp("2015-12-21")
        self.mmm_data.mmmdata_spec.interval_type = "week"
        self.mmm_data.mmmdata_spec.day_interval = 7

        self.mmm_data.mmmdata_spec.context_vars = ["competitor_sales_B"]
        self.mmm_data.mmmdata_spec.organic_vars = ["events", "newsletter"]
        self.mmm_data.mmmdata_spec.factor_vars = []

        self.mmm_data.data = self.dt_simulated.copy()
        self.mmm_data.mmmdata_spec.get_paid_media_var = MagicMock(return_value="facebook_I")

        self.holidays_data.prophet_vars = ["trend", "season", "holiday"]
        self.holidays_data.dt_holidays = self.dt_holidays.copy()
        self.holidays_data.prophet_country = "AD"

        self.hyperparameters.adstock = AdstockType.GEOMETRIC

        self.feature_engineering = FeatureEngineering(self.mmm_data, self.hyperparameters, self.holidays_data)

    def test_init(self):
        """Test initialization of FeatureEngineering class"""
        self.assertIsInstance(self.feature_engineering.mmm_data, MMMData)
        self.assertIsInstance(self.feature_engineering.hyperparameters, Hyperparameters)
        self.assertIsInstance(self.feature_engineering.holidays_data, HolidaysData)
        self.assertIsNotNone(self.feature_engineering.logger)

    def test_perform_feature_engineering(self):
        """Test the complete feature engineering pipeline"""
        result = self.feature_engineering.perform_feature_engineering(quiet=True)

        self.assertIsInstance(result, FeaturizedMMMData)
        self.assertIsInstance(result.dt_mod, pd.DataFrame)
        self.assertIsInstance(result.dt_modRollWind, pd.DataFrame)
        self.assertIsInstance(result.modNLS, dict)
        self.assertIn("results", result.modNLS)
        self.assertIn("yhat", result.modNLS)
        self.assertIn("plots", result.modNLS)

    def test_prepare_data(self):
        """Test data preparation method"""
        result = self.feature_engineering._prepare_data()

        self.assertTrue("ds" in result.columns)
        self.assertTrue("revenue" in result.columns)
        self.assertTrue("competitor_sales_B" in result.columns)
        self.assertEqual(len(result), len(self.dt_simulated))

        if not pd.api.types.is_datetime64_any_dtype(result["ds"]):
            result["ds"] = pd.to_datetime(result["ds"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["ds"]))

    def test_create_rolling_window_data_with_both_windows(self):
        """Test rolling window creation with both start and end dates"""
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(self.dt_simulated["DATE"]), "revenue": self.dt_simulated["revenue"]}
        )
        result = self.feature_engineering._create_rolling_window_data(dt_transform)
        self.assertTrue(all(result["ds"] >= self.mmm_data.mmmdata_spec.window_start))
        self.assertTrue(all(result["ds"] <= self.mmm_data.mmmdata_spec.window_end))

    def test_create_rolling_window_data_with_no_windows(self):
        """Test rolling window creation with no window constraints"""
        self.mmm_data.mmmdata_spec.window_start = None
        self.mmm_data.mmmdata_spec.window_end = None
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(self.dt_simulated["DATE"]), "revenue": self.dt_simulated["revenue"]}
        )
        result = self.feature_engineering._create_rolling_window_data(dt_transform)
        self.assertEqual(len(result), len(dt_transform))

    def test_create_rolling_window_invalid_dates(self):
        """Test rolling window creation with invalid date ranges"""
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(self.dt_simulated["DATE"]), "revenue": self.dt_simulated["revenue"]}
        )

        # Test start after end
        self.mmm_data.mmmdata_spec.window_start = pd.Timestamp("2015-12-31")
        self.mmm_data.mmmdata_spec.window_end = pd.Timestamp("2015-01-01")
        with self.assertRaises(ValueError):
            self.feature_engineering._create_rolling_window_data(dt_transform)

    def test_calculate_media_cost_factor(self):
        """Test media cost factor calculation"""
        dt_input = self.dt_simulated[["facebook_S"]].copy()
        result = self.feature_engineering._calculate_media_cost_factor(dt_input)

        self.assertEqual(len(result), 1)  # One media channel
        self.assertEqual(result.sum(), 1.0)  # Should sum to 1
        self.assertTrue(all(0 <= x <= 1 for x in result))  # Values between 0 and 1

    def test_run_models(self):
        """Test model running for media variables"""
        dt_modRollWind = self.dt_simulated[["facebook_S", "facebook_I"]].copy()
        result = self.feature_engineering._run_models(dt_modRollWind, 1.0)

        self.assertIn("results", result)
        self.assertIn("yhat", result)
        self.assertIn("plots", result)

    def test_hill_function(self):
        """Test Hill function transformation"""
        x = np.array([1, 2, 3, 4, 5])
        alpha = 2
        gamma = 3
        result = self.feature_engineering._hill_function(x, alpha, gamma)

        self.assertEqual(len(result), len(x))
        self.assertTrue(all(0 <= r <= 1 for r in result))

    def test_apply_adstock_geometric(self):
        """Test geometric adstock transformation"""
        x = pd.Series([1, 2, 3, 4, 5])
        params = MagicMock(spec=ChannelHyperparameters)
        params.thetas = [0.5]

        result = self.feature_engineering._apply_adstock(x, params)
        self.assertEqual(len(result), len(x))

    def test_apply_adstock_weibull(self):
        """Test Weibull adstock transformation"""
        x = pd.Series([1, 2, 3, 4, 5])
        params = MagicMock(spec=ChannelHyperparameters)
        params.shapes = [1.0]
        params.scales = [1.0]
        self.hyperparameters.adstock = AdstockType.WEIBULL_CDF

        result = self.feature_engineering._apply_adstock(x, params)
        self.assertEqual(len(result), len(x))

    def test_apply_adstock_invalid(self):
        """Test invalid adstock type handling"""
        x = pd.Series([1, 2, 3, 4, 5])
        params = MagicMock(spec=ChannelHyperparameters)
        self.hyperparameters.adstock = "INVALID"

        with self.assertRaises(ValueError):
            self.feature_engineering._apply_adstock(x, params)

    def test_prophet_decomposition(self):
        """Test Prophet decomposition"""
        dt_mod = pd.DataFrame(
            {
                "ds": pd.to_datetime(self.dt_simulated["DATE"]),
                "dep_var": self.dt_simulated["revenue"],
                "facebook_S": self.dt_simulated["facebook_S"],
                "competitor_sales_B": self.dt_simulated["competitor_sales_B"],
                "events": self.dt_simulated["events"],
                "newsletter": self.dt_simulated["newsletter"],
            }
        )

        with patch("prophet.Prophet") as mock_prophet:
            mock_prophet_instance = MagicMock()
            mock_prophet_instance.fit.return_value = None
            prophet_output = pd.DataFrame(
                {
                    "ds": dt_mod["ds"],
                    "trend": np.ones(len(dt_mod)),
                    "yearly": np.zeros(len(dt_mod)),
                    "holidays": np.zeros(len(dt_mod)),
                    "yhat": np.ones(len(dt_mod)),
                }
            )
            mock_prophet_instance.predict.return_value = prophet_output
            mock_prophet.return_value = mock_prophet_instance

            result = self.feature_engineering._prophet_decomposition(dt_mod)
            self.assertIn("trend", result.columns)
            self.assertIn("season", result.columns)
            self.assertIn("holiday", result.columns)

    def test_set_holidays_daily(self):
        """Test holiday setting for daily data"""
        dt_transform = pd.DataFrame({"ds": pd.date_range(start="2015-01-01", end="2015-01-10")})
        result = self.feature_engineering._set_holidays(dt_transform, self.holidays_data.dt_holidays, "day")
        self.assertTrue(all(col in result.columns for col in ["holiday", "ds", "country"]))

    def test_set_holidays_monthly(self):
        """Test holiday setting for monthly data"""
        dt_transform = pd.DataFrame({"ds": pd.date_range(start="2015-01-01", end="2015-12-01", freq="MS")})
        result = self.feature_engineering._set_holidays(dt_transform, self.holidays_data.dt_holidays, "month")
        self.assertTrue(all(col in result.columns for col in ["holiday", "ds", "country"]))

    def test_set_holidays_invalid_interval(self):
        """Test invalid interval type handling"""
        dt_transform = pd.DataFrame({"ds": pd.date_range(start="2015-01-01", end="2015-01-10")})
        with self.assertRaises(ValueError):
            self.feature_engineering._set_holidays(dt_transform, self.holidays_data.dt_holidays, "invalid")


if __name__ == "__main__":
    unittest.main()
