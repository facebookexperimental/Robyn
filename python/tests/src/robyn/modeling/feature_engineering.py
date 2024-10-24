import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import AdstockType


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.mmm_data = MagicMock(spec=MMMData)
        self.hyperparameters = MagicMock(spec=Hyperparameters)
        self.holidays_data = MagicMock(spec=HolidaysData)
        self.feature_engineering = FeatureEngineering(self.mmm_data, self.hyperparameters, self.holidays_data)

    @patch.object(FeatureEngineering, "_prepare_data")
    @patch.object(FeatureEngineering, "_prophet_decomposition")
    @patch.object(FeatureEngineering, "_create_rolling_window_data")
    @patch.object(FeatureEngineering, "_calculate_media_cost_factor")
    @patch.object(FeatureEngineering, "_run_models")
    def test_perform_feature_engineering(
        self,
        mock_run_models,
        mock_calculate_media_cost_factor,
        mock_create_rolling_window_data,
        mock_prophet_decomposition,
        mock_prepare_data,
    ):
        # Mocking
        mock_prepare_data.return_value = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "dep_var": [100, 200],
            }
        )
        mock_prophet_decomposition.return_value = mock_prepare_data.return_value
        mock_create_rolling_window_data.return_value = mock_prepare_data.return_value
        mock_calculate_media_cost_factor.return_value = pd.Series([0.5, 0.5])
        mock_run_models.return_value = {
            "results": {},
            "yhat": pd.DataFrame(),
            "plots": {},
        }

        # Execution
        result = self.feature_engineering.perform_feature_engineering()

        # Assertions
        self.assertIsInstance(result, FeaturizedMMMData)
        self.assertTrue(hasattr(result, "dt_mod"))
        self.assertTrue(hasattr(result, "dt_modRollWind"))
        self.assertTrue(hasattr(result, "modNLS"))

    def test_prepare_data(self):
        # Setup a mock MMMData
        self.mmm_data.data = pd.DataFrame(
            {
                "date_var": ["2021-01-01", "2021-01-02"],
                "dep_var": [100, 200],
                "competitor_sales_B": ["1", "2"],
            }
        )
        self.mmm_data.mmmdata_spec.date_var = "date_var"
        self.mmm_data.mmmdata_spec.dep_var = "dep_var"

        # Execution
        result = self.feature_engineering._prepare_data()

        # Assertions
        self.assertEqual(result["ds"].tolist(), ["2021-01-01", "2021-01-02"])
        self.assertEqual(result["dep_var"].tolist(), [100, 200])
        self.assertTrue(pd.api.types.is_integer_dtype(result["competitor_sales_B"]))

    def test_create_rolling_window_data(self):
        self.mmm_data.mmmdata_spec.window_start = "2021-01-01"
        self.mmm_data.mmmdata_spec.window_end = "2021-01-02"
        dt_transform = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
                "dep_var": [100, 200, 300],
            }
        )

        result = self.feature_engineering._create_rolling_window_data(dt_transform)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["ds"].min(), pd.Timestamp("2021-01-01"))
        self.assertEqual(result["ds"].max(), pd.Timestamp("2021-01-02"))

    def test_calculate_media_cost_factor(self):
        self.mmm_data.mmmdata_spec.paid_media_spends = ["spend1", "spend2"]
        dt_input_roll_wind = pd.DataFrame(
            {
                "spend1": [100, 200],
                "spend2": [300, 400],
            }
        )

        result = self.feature_engineering._calculate_media_cost_factor(dt_input_roll_wind)

        expected_result = pd.Series({"spend1": 0.16666666666666666, "spend2": 0.3333333333333333})
        pd.testing.assert_series_equal(result, expected_result)

    @patch.object(FeatureEngineering, "_fit_spend_exposure")
    def test_run_models(self, mock_fit_spend_exposure):
        self.mmm_data.mmmdata_spec.paid_media_spends = ["media1", "media2"]
        dt_modRollWind = pd.DataFrame({"media1": [1, 2], "media2": [3, 4]})
        media_cost_factor = 1.0

        mock_fit_spend_exposure.return_value = {
            "res": {"rsq": 0.9},
            "plot": pd.DataFrame({"x": [1], "y": [2]}),
            "yhat": [3],
        }

        result = self.feature_engineering._run_models(dt_modRollWind, media_cost_factor)

        self.assertIn("media1", result["results"])
        self.assertIn("media2", result["results"])

    def test_fit_spend_exposure(self):
        self.mmm_data.mmmdata_spec.paid_media_spends = ["spend_var"]
        self.mmm_data.mmmdata_spec.paid_media_vars = ["exposure_var"]

        dt_modRollWind = pd.DataFrame(
            {
                "spend_var": [10, 20, 30],
                "exposure_var": [15, 25, 35],
            }
        )

        result = self.feature_engineering._fit_spend_exposure(dt_modRollWind, "spend_var", 1.0)

        self.assertIn("channel", result["res"])
        self.assertIn("model_type", result["res"])
        self.assertIn("rsq", result["res"])
        self.assertIn("coef", result["res"])

    def test_hill_function(self):
        result = self.feature_engineering._hill_function(10, 2, 3)
        expected = (10**2) / (10**2 + 3**2)
        self.assertEqual(result, expected)

    @patch.object(FeatureEngineering, "_set_holidays")
    def test_prophet_decomposition(self, mock_set_holidays):
        mock_set_holidays.return_value = pd.DataFrame({"ds": pd.to_datetime(["2021-01-01"]), "holiday": ["New Year"]})
        dt_mod = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "dep_var": [100, 200],
            }
        )
        self.holidays_data.prophet_vars = ["trend", "season", "holiday"]

        result = self.feature_engineering._prophet_decomposition(dt_mod)

        self.assertIn("trend", result.columns)
        self.assertIn("season", result.columns)
        self.assertIn("holiday", result.columns)

    def test_set_holidays(self):
        dt_transform = pd.DataFrame({"ds": pd.to_datetime(["2021-01-01", "2021-01-02"])})
        dt_holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2021-01-01"]),
                "holiday": ["New Year"],
                "country": ["US"],
                "year": [2021],
            }
        )

        result = self.feature_engineering._set_holidays(dt_transform, dt_holidays, "day")

        self.assertEqual(len(result), 1)
        self.assertEqual(result["holiday"].iloc[0], "New Year")

    @patch.object(FeatureEngineering, "_geometric_adstock")
    @patch.object(FeatureEngineering, "_weibull_adstock")
    def test_apply_adstock(self, mock_weibull_adstock, mock_geometric_adstock):
        x = pd.Series([1, 2, 3])
        params = MagicMock(spec=ChannelHyperparameters)
        params.thetas = [0.5]
        params.shapes = [1.0]
        params.scales = [1.0]

        self.hyperparameters.adstock = "GEOMETRIC"
        mock_geometric_adstock.return_value = x
        result = self.feature_engineering._apply_adstock(x, params)
        mock_geometric_adstock.assert_called_once()

        self.hyperparameters.adstock = "WEIBULL_CDF"
        mock_weibull_adstock.return_value = x
        result = self.feature_engineering._apply_adstock(x, params)
        mock_weibull_adstock.assert_called_once()

    def test_weibull_adstock(self):
        x = pd.Series([1, 2, 3])
        shape = 1.0
        scale = 1.0
        result = self.feature_engineering._weibull_adstock(x, shape, scale)
        self.assertIsInstance(result, pd.Series)


if __name__ == "__main__":
    unittest.main()
