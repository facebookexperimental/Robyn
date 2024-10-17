import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData


class TestFeatureEngineering(unittest.TestCase):

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
        # Setup mocks
        mock_prepare_data.return_value = pd.DataFrame({"mocked": [1, 2, 3]})
        mock_prophet_decomposition.return_value = pd.DataFrame(
            {"prophet_decomposed": [1, 2, 3]}
        )
        mock_create_rolling_window_data.return_value = pd.DataFrame(
            {"rolling_window": [1, 2, 3]}
        )
        mock_calculate_media_cost_factor.return_value = "Media Cost Factor"
        mock_run_models.return_value = "Model Results"

        # Instantiate FeatureEngineering
        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        # Call perform_feature_engineering
        result = fe.perform_feature_engineering(quiet=False)

        # Assertions
        self.assertIsInstance(result, FeaturizedMMMData)
        self.assertEqual(result.dt_mod.columns.tolist(), ["mocked"])
        self.assertEqual(result.dt_modRollWind.columns.tolist(), ["rolling_window"])
        self.assertEqual(result.modNLS, "Model Results")

    @patch.object(FeatureEngineering, "_prepare_data")
    @patch.object(FeatureEngineering, "_create_rolling_window_data")
    @patch.object(FeatureEngineering, "_calculate_media_cost_factor")
    @patch.object(FeatureEngineering, "_run_models")
    def test_perform_feature_engineering_no_prophet_vars(
        self,
        mock_run_models,
        mock_calculate_media_cost_factor,
        mock_create_rolling_window_data,
        mock_prepare_data,
    ):
        # Setup mocks
        mock_prepare_data.return_value = pd.DataFrame({"mocked": [1, 2, 3]})
        mock_create_rolling_window_data.return_value = pd.DataFrame(
            {"rolling_window": [1, 2, 3]}
        )
        mock_calculate_media_cost_factor.return_value = "Media Cost Factor"
        mock_run_models.return_value = "Model Results"

        # Instantiate FeatureEngineering
        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        # Call perform_feature_engineering
        result = fe.perform_feature_engineering(quiet=True)

        # Assertions
        self.assertIsInstance(result, FeaturizedMMMData)
        self.assertEqual(result.dt_mod.columns.tolist(), ["mocked"])
        self.assertEqual(result.dt_modRollWind.columns.tolist(), ["rolling_window"])
        self.assertEqual(result.modNLS, "Model Results")

    @patch.object(FeatureEngineering, "_prepare_data")
    @patch.object(FeatureEngineering, "_create_rolling_window_data")
    @patch.object(FeatureEngineering, "_calculate_media_cost_factor")
    @patch.object(FeatureEngineering, "_run_models")
    def test_perform_feature_engineering_missing_columns(
        self,
        mock_run_models,
        mock_calculate_media_cost_factor,
        mock_create_rolling_window_data,
        mock_prepare_data,
    ):
        # Setup mocks
        mock_prepare_data.return_value = pd.DataFrame({"missing_column": [1, 2, 3]})
        mock_create_rolling_window_data.return_value = pd.DataFrame(
            {"rolling_window": [1, 2, 3]}
        )
        mock_calculate_media_cost_factor.return_value = "Media Cost Factor"
        mock_run_models.return_value = "Model Results"

        # Instantiate FeatureEngineering
        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        # Call perform_feature_engineering
        result = fe.perform_feature_engineering(quiet=True)

        # Assertions
        self.assertIsInstance(result, FeaturizedMMMData)
        self.assertEqual(result.dt_mod.columns.tolist(), ["missing_column"])
        self.assertEqual(result.dt_modRollWind.columns.tolist(), ["rolling_window"])
        self.assertEqual(result.modNLS, "Model Results")

    def test__prepare_data(self):
        # Simulate mmm_data with necessary attributes
        mmm_data = MagicMock()
        mmm_data.data = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "sales": [100, 150],
                "competitor_sales_B": [200, 250],
            }
        )
        mmm_data.mmmdata_spec.date_var = "date"
        mmm_data.mmmdata_spec.dep_var = "sales"

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._prepare_data()

        self.assertEqual(
            result.columns.tolist(),
            ["date", "sales", "competitor_sales_B", "ds", "dep_var"],
        )
        self.assertEqual(result["competitor_sales_B"].dtype, np.int64)
        self.assertEqual(result["ds"].dtype, np.object_)

    def test__prepare_data_missing_competitor_sales_B(self):
        # Simulate mmm_data with necessary attributes
        mmm_data = MagicMock()
        mmm_data.data = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "sales": [100, 150],
            }
        )
        mmm_data.mmmdata_spec.date_var = "date"
        mmm_data.mmmdata_spec.dep_var = "sales"

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        with self.assertRaises(KeyError) as context:
            fe._prepare_data()

        self.assertEqual(str(context.exception), "'competitor_sales_B'")

    def test__prepare_data_empty_input(self):
        # Simulate mmm_data with necessary attributes
        mmm_data = MagicMock()
        mmm_data.data = pd.DataFrame(columns=["date", "sales", "competitor_sales_B"])
        mmm_data.mmmdata_spec.date_var = "date"
        mmm_data.mmmdata_spec.dep_var = "sales"

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._prepare_data()

        self.assertEqual(result.shape, (0, 5))

    def test__prepare_data_null_values_in_competitor_sales_B(self):
        # Simulate mmm_data with necessary attributes
        mmm_data = MagicMock()
        mmm_data.data = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "sales": [100, 150],
                "competitor_sales_B": [np.nan, 250],
            }
        )
        mmm_data.mmmdata_spec.date_var = "date"
        mmm_data.mmmdata_spec.dep_var = "sales"

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._prepare_data()

        self.assertEqual(result["competitor_sales_B"].dtype, np.int64)

    def test__create_rolling_window_data(self):
        # Simulate dt_transform DataFrame
        dt_transform = pd.DataFrame(
            {"ds": ["2020-01-01", "2020-01-02"], "value": [100, 150]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.window_start = None
        mmm_data.mmmdata_spec.window_end = None

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._create_rolling_window_data(dt_transform)

        pd.testing.assert_frame_equal(result, dt_transform)

    def test__create_rolling_window_data_with_window_end(self):
        # Simulate dt_transform DataFrame
        dt_transform = pd.DataFrame(
            {"ds": ["2020-01-01", "2020-01-02"], "value": [100, 150]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.window_start = None
        mmm_data.mmmdata_spec.window_end = "2020-01-01"

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._create_rolling_window_data(dt_transform)

        expected_result = dt_transform[dt_transform["ds"] <= "2020-01-01"]

        pd.testing.assert_frame_equal(result, expected_result)

    def test__create_rolling_window_data_with_window_start(self):
        # Simulate dt_transform DataFrame
        dt_transform = pd.DataFrame(
            {"ds": ["2020-01-01", "2020-01-02"], "value": [100, 150]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.window_start = "2020-01-02"
        mmm_data.mmmdata_spec.window_end = None

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._create_rolling_window_data(dt_transform)

        expected_result = dt_transform[dt_transform["ds"] >= "2020-01-02"]

        pd.testing.assert_frame_equal(result, expected_result)

    def test__create_rolling_window_data_with_both_window_boundaries(self):
        # Simulate dt_transform DataFrame
        dt_transform = pd.DataFrame(
            {"ds": ["2020-01-01", "2020-01-02", "2020-01-03"], "value": [100, 150, 200]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.window_start = "2020-01-02"
        mmm_data.mmmdata_spec.window_end = "2020-01-03"

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._create_rolling_window_data(dt_transform)

        expected_result = dt_transform[
            (dt_transform["ds"] >= "2020-01-02") & (dt_transform["ds"] <= "2020-01-03")
        ]

        pd.testing.assert_frame_equal(result, expected_result)

    def test__create_rolling_window_data_with_invalid_window_boundaries(self):
        # Simulate dt_transform DataFrame
        dt_transform = pd.DataFrame(
            {"ds": ["2020-01-01", "2020-01-02", "2020-01-03"], "value": [100, 150, 200]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.window_start = "2020-01-03"
        mmm_data.mmmdata_spec.window_end = "2020-01-02"

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._create_rolling_window_data(dt_transform)

        self.assertTrue(result.empty)

    def test__calculate_media_cost_factor(self):
        # Simulate dt_input_roll_wind DataFrame
        dt_input_roll_wind = pd.DataFrame(
            {"paid_media_1": [100, 150], "paid_media_2": [200, 250]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1", "paid_media_2"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._calculate_media_cost_factor(dt_input_roll_wind)

        expected_result = pd.Series([0.1, 0.15], index=["paid_media_1", "paid_media_2"])

        pd.testing.assert_series_equal(result, expected_result)

    def test__calculate_media_cost_factor_with_zero_spend(self):
        # Simulate dt_input_roll_wind DataFrame
        dt_input_roll_wind = pd.DataFrame(
            {"paid_media_1": [0, 0], "paid_media_2": [0, 0]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1", "paid_media_2"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._calculate_media_cost_factor(dt_input_roll_wind)

        expected_result = pd.Series(
            [np.nan, np.nan], index=["paid_media_1", "paid_media_2"]
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test__calculate_media_cost_factor_with_large_numbers(self):
        # Simulate dt_input_roll_wind DataFrame
        dt_input_roll_wind = pd.DataFrame(
            {"paid_media_1": [1e10, 1.5e10], "paid_media_2": [2e10, 2.5e10]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1", "paid_media_2"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._calculate_media_cost_factor(dt_input_roll_wind)

        expected_result = pd.Series([0.1, 0.15], index=["paid_media_1", "paid_media_2"])

        pd.testing.assert_series_equal(result, expected_result)

    def test__calculate_media_cost_factor_with_missing_spends(self):
        # Simulate dt_input_roll_wind DataFrame
        dt_input_roll_wind = pd.DataFrame(
            {"paid_media_1": [100, np.nan], "paid_media_2": [200, 250]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1", "paid_media_2"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._calculate_media_cost_factor(dt_input_roll_wind)

        expected_result = pd.Series([0.1, 0.15], index=["paid_media_1", "paid_media_2"])

        pd.testing.assert_series_equal(result, expected_result)

    @patch.object(FeatureEngineering, "_fit_spend_exposure")
    def test__run_models(self, mock_fit_spend_exposure):
        # Setup mocks
        mock_fit_spend_exposure.side_effect = [
            {
                "res": "result1",
                "plot": pd.DataFrame({"data": [1, 2]}),
                "yhat": pd.Series([0.1, 0.2]),
            },
            None,
        ]

        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"media_var_1": [100, 150], "media_var_2": [200, 250]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["media_var_1", "media_var_2"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._run_models(dt_modRollWind, media_cost_factor=1.0)

        self.assertIn("media_var_1", result["results"])
        self.assertNotIn("media_var_2", result["results"])
        self.assertEqual(result["results"]["media_var_1"], "result1")
        self.assertEqual(result["yhat"].shape[0], 2)
        self.assertEqual(result["plots"]["media_var_1"].shape[0], 2)

    def test__run_models_with_no_media_spends(self):
        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"other_var_1": [100, 150], "other_var_2": [200, 250]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = []

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._run_models(dt_modRollWind, media_cost_factor=1.0)

        self.assertEqual(result["results"], {})
        self.assertTrue(result["yhat"].empty)
        self.assertEqual(result["plots"], {})

    @patch.object(FeatureEngineering, "_fit_spend_exposure")
    def test__run_models_with_multiple_media_spends(self, mock_fit_spend_exposure):
        # Setup mocks
        mock_fit_spend_exposure.side_effect = [
            {"res": "result1", "plot": pd.DataFrame({"data": [1, 2]})},
            {"res": "result2", "plot": pd.DataFrame({"data": [3, 4]})},
        ]

        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"media_var_1": [100, 150], "media_var_2": [200, 250]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["media_var_1", "media_var_2"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._run_models(dt_modRollWind, media_cost_factor=0.5)

        self.assertIn("media_var_1", result["results"])
        self.assertIn("media_var_2", result["results"])
        self.assertEqual(result["results"]["media_var_1"], "result1")
        self.assertEqual(result["results"]["media_var_2"], "result2")
        self.assertEqual(result["plots"]["media_var_1"].shape[0], 2)
        self.assertEqual(result["plots"]["media_var_2"].shape[0], 2)

    @patch.object(FeatureEngineering, "_fit_spend_exposure")
    def test__run_models_with_zero_media_cost_factor(self, mock_fit_spend_exposure):
        # Setup mocks
        mock_fit_spend_exposure.return_value = None

        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"media_var_1": [100, 150], "media_var_2": [200, 250]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["media_var_1", "media_var_2"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._run_models(dt_modRollWind, media_cost_factor=0.0)

        self.assertEqual(result["results"], {})
        self.assertTrue(result["yhat"].empty)
        self.assertEqual(result["plots"], {})

    def test__fit_spend_exposure(self):
        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"paid_media_1": [1, 2, 3, 4, 5], "exposure_1": [2, 4, 6, 8, 10]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1"]
        mmm_data.mmmdata_spec.paid_media_vars = ["exposure_1"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._fit_spend_exposure(
            dt_modRollWind, "paid_media_1", media_cost_factor=1.0
        )

        self.assertEqual(result["res"]["channel"], "paid_media_1")
        self.assertEqual(
            result["res"]["model_type"], "lm"
        )  # Assuming linear model is better in this mock
        self.assertEqual(result["res"]["rsq"], 1.0)
        self.assertEqual(result["plot"].shape[0], 5)

    def test__fit_spend_exposure_linear_model(self):
        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"paid_media_1": [1, 2, 3, 4, 5], "exposure_1": [2, 4, 6, 8, 10]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1"]
        mmm_data.mmmdata_spec.paid_media_vars = ["exposure_1"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._fit_spend_exposure(
            dt_modRollWind, "paid_media_1", media_cost_factor=1.0
        )

        self.assertEqual(result["res"]["channel"], "paid_media_1")
        self.assertEqual(result["res"]["model_type"], "lm")
        self.assertEqual(result["res"]["rsq"], 1.0)
        self.assertEqual(result["plot"].shape[0], 5)

    def test__fit_spend_exposure_zero_spend(self):
        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"paid_media_1": [0, 0, 0, 0, 0], "exposure_1": [2, 4, 6, 8, 10]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1"]
        mmm_data.mmmdata_spec.paid_media_vars = ["exposure_1"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._fit_spend_exposure(
            dt_modRollWind, "paid_media_1", media_cost_factor=1.0
        )

        self.assertEqual(result["res"]["channel"], "paid_media_1")
        self.assertEqual(result["res"]["model_type"], "lm")
        self.assertEqual(result["plot"].shape[0], 5)

    def test__fit_spend_exposure_negative_spend(self):
        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"paid_media_1": [-1, -2, -3, -4, -5], "exposure_1": [2, 4, 6, 8, 10]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1"]
        mmm_data.mmmdata_spec.paid_media_vars = ["exposure_1"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._fit_spend_exposure(
            dt_modRollWind, "paid_media_1", media_cost_factor=1.0
        )

        self.assertEqual(result["res"]["channel"], "paid_media_1")
        self.assertEqual(result["res"]["model_type"], "lm")
        self.assertEqual(result["plot"].shape[0], 5)

    def test__fit_spend_exposure_nan_exposure_data(self):
        # Simulate dt_modRollWind DataFrame
        dt_modRollWind = pd.DataFrame(
            {"paid_media_1": [1, 2, 3], "exposure_1": [np.nan, 4, 6]}
        )

        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["paid_media_1"]
        mmm_data.mmmdata_spec.paid_media_vars = ["exposure_1"]

        fe = FeatureEngineering(
            mmm_data=mmm_data, hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._fit_spend_exposure(
            dt_modRollWind, "paid_media_1", media_cost_factor=1.0
        )

        self.assertEqual(result["res"]["channel"], "paid_media_1")
        self.assertEqual(result["res"]["model_type"], "lm")
        self.assertEqual(result["plot"].shape[0], 3)

    def test__hill_function(self):
        result = FeatureEngineering._hill_function(x=2.0, alpha=1.0, gamma=1.0)
        self.assertAlmostEqual(result, 0.6666666666666666)

    def test__hill_function_with_zero_x(self):
        result = FeatureEngineering._hill_function(x=0.0, alpha=1.0, gamma=1.0)
        self.assertEqual(result, 0.0)

    def test__hill_function_with_zero_alpha(self):
        result = FeatureEngineering._hill_function(x=2.0, alpha=0.0, gamma=1.0)
        self.assertEqual(result, 0.5)

    def test__hill_function_with_zero_gamma(self):
        result = FeatureEngineering._hill_function(x=2.0, alpha=1.0, gamma=0.0)
        self.assertEqual(result, 1.0)

    def test__hill_function_with_large_alpha_gamma(self):
        result = FeatureEngineering._hill_function(x=2.0, alpha=100.0, gamma=100.0)
        self.assertEqual(result, 0.5)

    def test__hill_function_with_small_alpha(self):
        result = FeatureEngineering._hill_function(x=2.0, alpha=1e-10, gamma=1.0)
        self.assertEqual(result, 0.5)

    def test__hill_function_with_small_gamma(self):
        result = FeatureEngineering._hill_function(x=2.0, alpha=1.0, gamma=1e-10)
        self.assertEqual(result, 1.0)

    @patch("robyn.modeling.feature_engineering.Prophet")
    @patch.object(FeatureEngineering, "_set_holidays")
    def test__prophet_decomposition(self, mock_set_holidays, mock_prophet):
        # Setup mocks
        mock_set_holidays.return_value = pd.DataFrame(
            {"ds": ["2020-01-01"], "holiday": ["Holiday"]}
        )
        mock_prophet().fit().predict.return_value = pd.DataFrame(
            {
                "ds": ["2020-01-01"],
                "trend": [1.0],
                "season": [2.0],
                "monthly": [3.0],
                "weekly": [4.0],
                "yearly": [5.0],
                "holidays": [6.0],
            }
        )

        # Simulate dt_mod DataFrame
        dt_mod = pd.DataFrame({"ds": ["2020-01-01"], "dep_var": [100]})

        holidays_data = MagicMock()
        holidays_data.prophet_vars = [
            "trend",
            "holiday",
            "season",
            "monthly",
            "weekday",
        ]
        holidays_data.prophet_country = ["US"]

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=MagicMock(),
            holidays_data=holidays_data,
        )

        result = fe._prophet_decomposition(dt_mod)

        self.assertIn("trend", result.columns)
        self.assertIn("season", result.columns)
        self.assertIn("monthly", result.columns)
        self.assertIn("weekday", result.columns)
        self.assertIn("holiday", result.columns)

    @patch("robyn.modeling.feature_engineering.Prophet")
    @patch.object(FeatureEngineering, "_set_holidays")
    def test__prophet_decomposition_no_holidays(self, mock_set_holidays, mock_prophet):
        # Setup mocks
        mock_set_holidays.return_value = pd.DataFrame(
            {"ds": ["2020-01-01"], "holiday": ["Holiday"]}
        )
        mock_prophet().fit().predict.return_value = pd.DataFrame(
            {
                "ds": ["2020-01-01"],
                "trend": [1.0],
                "season": [2.0],
                "monthly": [3.0],
                "weekly": [4.0],
                "yearly": [5.0],
                "holidays": [6.0],
            }
        )

        # Simulate dt_mod DataFrame
        dt_mod = pd.DataFrame({"ds": ["2020-01-01"], "dep_var": [100]})

        holidays_data = MagicMock()
        holidays_data.prophet_vars = ["trend", "season", "monthly", "weekday"]
        holidays_data.prophet_country = []

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=MagicMock(),
            holidays_data=holidays_data,
        )

        result = fe._prophet_decomposition(dt_mod)

        self.assertIn("trend", result.columns)
        self.assertIn("season", result.columns)
        self.assertIn("monthly", result.columns)
        self.assertIn("weekday", result.columns)
        self.assertNotIn("holiday", result.columns)

    @patch("robyn.modeling.feature_engineering.Prophet")
    @patch.object(FeatureEngineering, "_set_holidays")
    def test__prophet_decomposition_only_trend_and_season(
        self, mock_set_holidays, mock_prophet
    ):
        # Setup mocks
        mock_set_holidays.return_value = pd.DataFrame(
            {"ds": ["2020-01-01"], "holiday": ["Holiday"]}
        )
        mock_prophet().fit().predict.return_value = pd.DataFrame(
            {
                "ds": ["2020-01-01"],
                "trend": [1.0],
                "season": [2.0],
                "monthly": [3.0],
                "weekly": [4.0],
                "yearly": [5.0],
                "holidays": [6.0],
            }
        )

        # Simulate dt_mod DataFrame
        dt_mod = pd.DataFrame({"ds": ["2020-01-01"], "dep_var": [100]})

        holidays_data = MagicMock()
        holidays_data.prophet_vars = ["trend", "season"]
        holidays_data.prophet_country = ["US"]

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=MagicMock(),
            holidays_data=holidays_data,
        )

        result = fe._prophet_decomposition(dt_mod)

        self.assertIn("trend", result.columns)
        self.assertIn("season", result.columns)
        self.assertNotIn("monthly", result.columns)
        self.assertNotIn("weekday", result.columns)
        self.assertNotIn("holiday", result.columns)

    @patch("robyn.modeling.feature_engineering.Prophet")
    @patch.object(FeatureEngineering, "_set_holidays")
    def test__prophet_decomposition_custom_seasonality(
        self, mock_set_holidays, mock_prophet
    ):
        # Setup mocks
        mock_set_holidays.return_value = pd.DataFrame(
            {"ds": ["2020-01-01"], "holiday": ["Holiday"]}
        )
        mock_prophet().fit().predict.return_value = pd.DataFrame(
            {
                "ds": ["2020-01-01"],
                "trend": [1.0],
                "season": [2.0],
                "monthly": [3.0],
                "weekly": [4.0],
                "yearly": [5.0],
                "holidays": [6.0],
            }
        )

        # Simulate dt_mod DataFrame
        dt_mod = pd.DataFrame({"ds": ["2020-01-01"], "dep_var": [100]})

        holidays_data = MagicMock()
        holidays_data.prophet_vars = [
            "trend",
            "holiday",
            "season",
            "monthly",
            "weekday",
        ]
        holidays_data.prophet_country = ["US"]

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=MagicMock(),
            holidays_data=holidays_data,
        )
        fe.custom_params = {"yearly.seasonality": 10, "weekly.seasonality": 5}

        result = fe._prophet_decomposition(dt_mod)

        self.assertIn("trend", result.columns)
        self.assertIn("season", result.columns)
        self.assertIn("monthly", result.columns)
        self.assertIn("weekday", result.columns)
        self.assertIn("holiday", result.columns)

    def test__set_holidays(self):
        # Simulate dt_transform and dt_holidays DataFrames
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(["2020-01-01", "2020-01-02"])}
        )
        dt_holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2020-01-01"]),
                "holiday": ["Holiday"],
                "country": ["US"],
                "year": [2020],
            }
        )

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._set_holidays(dt_transform, dt_holidays, interval_type="day")

        pd.testing.assert_frame_equal(result, dt_holidays)

    def test__set_holidays_weekly(self):
        # Simulate dt_transform and dt_holidays DataFrames
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(["2020-01-01", "2020-01-08"])}
        )
        dt_holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2020-01-01", "2020-01-08"]),
                "holiday": ["New Year", "Another Holiday"],
                "country": ["US", "US"],
                "year": [2020, 2020],
            }
        )

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._set_holidays(dt_transform, dt_holidays, interval_type="week")

        expected_result = (
            dt_holidays.groupby(["ds", "country", "year"])
            .agg(holiday=("holiday", ", ".join))
            .reset_index()
        )

        pd.testing.assert_frame_equal(result, expected_result)

    def test__set_holidays_monthly(self):
        # Simulate dt_transform and dt_holidays DataFrames
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(["2020-01-01", "2020-02-01"])}
        )
        dt_holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                "holiday": ["New Year", "Valentine's Day"],
                "country": ["US", "US"],
                "year": [2020, 2020],
            }
        )

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._set_holidays(dt_transform, dt_holidays, interval_type="month")

        expected_result = (
            dt_holidays.groupby(["ds", "country", "year"])
            .agg(holiday=("holiday", ", ".join))
            .reset_index()
        )

        pd.testing.assert_frame_equal(result, expected_result)

    def test__set_holidays_monthly_invalid(self):
        # Simulate dt_transform and dt_holidays DataFrames
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(["2020-01-02", "2020-02-02"])}
        )
        dt_holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                "holiday": ["New Year", "Valentine's Day"],
                "country": ["US", "US"],
                "year": [2020, 2020],
            }
        )

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        with self.assertRaises(ValueError) as context:
            fe._set_holidays(dt_transform, dt_holidays, interval_type="month")

        self.assertEqual(
            str(context.exception),
            "Monthly data should have first day of month as datestamp, e.g.'2020-01-01'",
        )

    def test__set_holidays_invalid_interval_type(self):
        # Simulate dt_transform and dt_holidays DataFrames
        dt_transform = pd.DataFrame(
            {"ds": pd.to_datetime(["2020-01-01", "2020-02-01"])}
        )
        dt_holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                "holiday": ["New Year", "Valentine's Day"],
                "country": ["US", "US"],
                "year": [2020, 2020],
            }
        )

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        with self.assertRaises(ValueError) as context:
            fe._set_holidays(dt_transform, dt_holidays, interval_type="year")

        self.assertEqual(
            str(context.exception),
            "Invalid interval_type. Must be 'day', 'week', or 'month'.",
        )

    @patch.object(FeatureEngineering, "_apply_adstock")
    @patch.object(FeatureEngineering, "_apply_saturation")
    def test__apply_transformations(self, mock_apply_saturation, mock_apply_adstock):
        # Set up mocks
        mock_apply_adstock.return_value = pd.Series([10, 20, 30])
        mock_apply_saturation.return_value = pd.Series([1, 2, 3])

        # Simulate x and params
        x = pd.Series([1, 2, 3])
        params = MagicMock()

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._apply_transformations(x, params)

        pd.testing.assert_series_equal(result, pd.Series([1, 2, 3]))

    @patch.object(FeatureEngineering, "_apply_adstock")
    @patch.object(FeatureEngineering, "_apply_saturation")
    def test__apply_transformations_with_empty_data(
        self, mock_apply_saturation, mock_apply_adstock
    ):
        # Set up mocks
        mock_apply_adstock.return_value = pd.Series(dtype="float64")
        mock_apply_saturation.return_value = pd.Series(dtype="float64")

        # Simulate x and params
        x = pd.Series(dtype="float64")
        params = MagicMock()

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._apply_transformations(x, params)

        pd.testing.assert_series_equal(result, pd.Series(dtype="float64"))

    def test__apply_transformations_with_null_params(self):
        # Simulate x
        x = pd.Series([1, 2, 3])

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        with self.assertRaises(TypeError):
            fe._apply_transformations(x, None)

    @patch.object(FeatureEngineering, "_apply_adstock")
    @patch.object(FeatureEngineering, "_apply_saturation")
    def test__apply_transformations_with_negative_values(
        self, mock_apply_saturation, mock_apply_adstock
    ):
        # Set up mocks
        mock_apply_adstock.return_value = pd.Series([-10, -20, -30])
        mock_apply_saturation.return_value = pd.Series([-1, -2, -3])

        # Simulate x and params
        x = pd.Series([-1, -2, -3])
        params = MagicMock()

        fe = FeatureEngineering(
            mmm_data=MagicMock(), hyperparameters=MagicMock(), holidays_data=MagicMock()
        )

        result = fe._apply_transformations(x, params)

        pd.testing.assert_series_equal(result, pd.Series([-1, -2, -3]))

    @patch.object(FeatureEngineering, "_geometric_adstock")
    def test__apply_adstock(self, mock_geometric_adstock):
        # Setup mock
        mock_geometric_adstock.return_value = pd.Series([1, 2, 3])

        # Simulate x and params
        x = pd.Series([1, 2, 3])
        params = MagicMock()
        params.thetas = [0.5]

        hyperparameters = MagicMock()
        hyperparameters.adstock = "GEOMETRIC"

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=hyperparameters,
            holidays_data=MagicMock(),
        )

        result = fe._apply_adstock(x, params)

        pd.testing.assert_series_equal(result, pd.Series([1, 2, 3]))

    @patch.object(FeatureEngineering, "_weibull_adstock")
    def test__apply_adstock_weibull_cdf(self, mock_weibull_adstock):
        # Setup mock
        mock_weibull_adstock.return_value = pd.Series([1, 2, 3])

        # Simulate x and params
        x = pd.Series([1, 2, 3])
        params = MagicMock()
        params.shapes = [1.5]
        params.scales = [1.0]

        hyperparameters = MagicMock()
        hyperparameters.adstock = "WEIBULL_CDF"

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=hyperparameters,
            holidays_data=MagicMock(),
        )

        result = fe._apply_adstock(x, params)

        pd.testing.assert_series_equal(result, pd.Series([1, 2, 3]))

    @patch.object(FeatureEngineering, "_weibull_adstock")
    def test__apply_adstock_weibull_pdf(self, mock_weibull_adstock):
        # Setup mock
        mock_weibull_adstock.return_value = pd.Series([1, 2, 3])

        # Simulate x and params
        x = pd.Series([1, 2, 3])
        params = MagicMock()
        params.shapes = [1.5]
        params.scales = [1.0]

        hyperparameters = MagicMock()
        hyperparameters.adstock = "WEIBULL_PDF"

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=hyperparameters,
            holidays_data=MagicMock(),
        )

        result = fe._apply_adstock(x, params)

        pd.testing.assert_series_equal(result, pd.Series([1, 2, 3]))

    def test__apply_adstock_unsupported_type(self):
        # Simulate x and params
        x = pd.Series([1, 2, 3])
        params = MagicMock()
        params.thetas = [0.5]

        hyperparameters = MagicMock()
        hyperparameters.adstock = "UNKNOWN"

        fe = FeatureEngineering(
            mmm_data=MagicMock(),
            hyperparameters=hyperparameters,
            holidays_data=MagicMock(),
        )

        with self.assertRaises(ValueError) as context:
            fe._apply_adstock(x, params)

        self.assertEqual(str(context.exception), "Unsupported adstock type: UNKNOWN")

    def test__geometric_adstock(self):
        # Simulate x
        x = pd.Series([1, 2, 3, 4, 5])

        result = FeatureEngineering._geometric_adstock(x, theta=0.5)

        expected_result = pd.Series([1.0, 1.5, 2.25, 3.125, 4.0625])

        pd.testing.assert_series_equal(result, expected_result)

    def test__geometric_adstock_with_theta_zero(self):
        # Simulate x
        x = pd.Series([1, 2, 3, 4, 5])

        result = FeatureEngineering._geometric_adstock(x, theta=0.0)

        expected_result = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        pd.testing.assert_series_equal(result, expected_result)

    def test__geometric_adstock_with_theta_one(self):
        # Simulate x
        x = pd.Series([1, 2, 3, 4, 5])

        result = FeatureEngineering._geometric_adstock(x, theta=1.0)

        expected_result = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])

        pd.testing.assert_series_equal(result, expected_result)

    def test__geometric_adstock_with_empty_series(self):
        # Simulate x
        x = pd.Series(dtype="float64")

        result = FeatureEngineering._geometric_adstock(x, theta=0.5)

        expected_result = pd.Series(dtype="float64")

        pd.testing.assert_series_equal(result, expected_result)

    def test__geometric_adstock_with_single_element_series(self):
        # Simulate x
        x = pd.Series([1])

        result = FeatureEngineering._geometric_adstock(x, theta=0.5)

        expected_result = pd.Series([1.0])

        pd.testing.assert_series_equal(result, expected_result)

    def test__geometric_adstock_with_decreasing_series(self):
        # Simulate x
        x = pd.Series([5, 4, 3, 2, 1])

        result = FeatureEngineering._geometric_adstock(x, theta=0.7)

        expected_result = pd.Series([5.0, 4.7, 4.21, 3.563, 2.6681])

        pd.testing.assert_series_equal(result, expected_result)

    def test__geometric_adstock_with_theta_slightly_above_zero(self):
        # Simulate x
        x = pd.Series([1, 2, 3, 4, 5])

        result = FeatureEngineering._geometric_adstock(x, theta=0.01)

        expected_result = pd.Series([1.0, 1.01, 1.0299, 1.059601, 1.09900499])

        pd.testing.assert_series_equal(result, expected_result)

    def test__weibull_adstock(self):
        # Simulate x
        x = pd.Series([1, 2, 3, 4, 5])

        result = FeatureEngineering._weibull_adstock(x, shape=1.5, scale=1.0)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)
        self.assertTrue(result.iloc[0] > 0)
        self.assertTrue(result.iloc[-1] > 0)

    def test__weibull_adstock_with_constant_series(self):
        # Simulate x
        x = pd.Series([3, 3, 3, 3, 3])

        result = FeatureEngineering._weibull_adstock(x, shape=1.5, scale=1.0)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)
        self.assertTrue(result.iloc[0] > 0)
        self.assertTrue(result.iloc[-1] > 0)

    def test__weibull_adstock_with_zero_series(self):
        # Simulate x
        x = pd.Series([0, 0, 0, 0, 0])

        result = FeatureEngineering._weibull_adstock(x, shape=1.5, scale=1.0)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(result == 0))

    def test__weibull_adstock_with_decreasing_series(self):
        # Simulate x
        x = pd.Series([5, 4, 3, 2, 1])

        result = FeatureEngineering._weibull_adstock(x, shape=1.5, scale=1.0)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)
        self.assertTrue(result.iloc[0] > 0)
        self.assertTrue(result.iloc[-1] > 0)

    def test__weibull_adstock_with_shape_zero(self):
        # Simulate x
        x = pd.Series([1, 2, 3, 4, 5])

        result = FeatureEngineering._weibull_adstock(x, shape=0.0, scale=1.0)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(np.isclose(result, 0)))

    def test__weibull_adstock_with_scale_zero(self):
        # Simulate x
        x = pd.Series([1, 2, 3, 4, 5])

        with self.assertRaises(ZeroDivisionError):
            FeatureEngineering._weibull_adstock(x, shape=1.5, scale=0.0)

    def test__weibull_adstock_with_empty_series(self):
        # Simulate x
        x = pd.Series(dtype="float64")

        result = FeatureEngineering._weibull_adstock(x, shape=1.5, scale=1.0)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 0)

    def test__apply_saturation(self):
        # Simulate x and params
        x = pd.Series([1, 2, 3, 4, 5])
        params = MagicMock()
        params.alphas = [0.5]
        params.gammas = [1.0]

        result = FeatureEngineering._apply_saturation(x, params)

        expected_result = pd.Series(
            [
                1.0 / (1.0 + 1.0),
                2.0 / (2.0 + 1.0),
                3.0 / (3.0 + 1.0),
                4.0 / (4.0 + 1.0),
                5.0 / (5.0 + 1.0),
            ]
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test__apply_saturation_with_zero_values(self):
        # Simulate x and params
        x = pd.Series([0, 0, 0, 0, 0])
        params = MagicMock()
        params.alphas = [0.5]
        params.gammas = [1.0]

        result = FeatureEngineering._apply_saturation(x, params)

        expected_result = pd.Series([0, 0, 0, 0, 0])

        pd.testing.assert_series_equal(result, expected_result)

    def test__apply_saturation_with_negative_values(self):
        # Simulate x and params
        x = pd.Series([-1, -2, -3, -4, -5])
        params = MagicMock()
        params.alphas = [0.5]
        params.gammas = [1.0]

        result = FeatureEngineering._apply_saturation(x, params)

        expected_result = pd.Series(
            [
                -1.0 / (1.0 + 1.0),
                -2.0 / (2.0 + 1.0),
                -3.0 / (3.0 + 1.0),
                -4.0 / (4.0 + 1.0),
                -5.0 / (5.0 + 1.0),
            ]
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test__apply_saturation_with_large_values(self):
        # Simulate x and params
        x = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
        params = MagicMock()
        params.alphas = [0.5]
        params.gammas = [1.0]

        result = FeatureEngineering._apply_saturation(x, params)

        expected_result = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])

        pd.testing.assert_series_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
