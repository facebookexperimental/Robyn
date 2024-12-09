import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.enums import AdstockType
from robyn.modeling.transformations.transformations import Transformation
from robyn.modeling.pareto.response_curve import (
    ResponseCurveCalculator,
    ResponseOutput,
    UseCase,
    MetricDateInfo,
    MetricValueInfo,
)


class TestResponseCurveCalculator(unittest.TestCase):

    def setUp(self):
        # Mocking the dependencies
        self.mock_mmm_data = MagicMock(spec=MMMData)
        self.mock_mmm_data.mmmdata_spec = MagicMock()
        self.mock_model_outputs = MagicMock(spec=ModelOutputs)
        self.mock_hyperparameter = MagicMock(spec=Hyperparameters)
        self.mock_transformation = MagicMock(spec=Transformation)

        # Instantiating the ResponseCurveCalculator
        self.calculator = ResponseCurveCalculator(
            self.mock_mmm_data, self.mock_model_outputs, self.mock_hyperparameter
        )

    def test_calculate_response(self):
        # Mock data setup
        self.mock_mmm_data.data = pd.DataFrame(
            {
                "spend_metric": [1, 2, 3, 4, 5],
                "date_var": pd.date_range("2020-01-01", periods=5),
            }
        )
        self.mock_mmm_data.mmmdata_spec.date_var = "date_var"
        self.mock_mmm_data.mmmdata_spec.paid_media_spends = ["spend_metric"]
        self.mock_mmm_data.mmmdata_spec.paid_media_vars = ["exposure_metric"]
        self.mock_mmm_data.mmmdata_spec.organic_vars = ["organic_metric"]
        self.mock_mmm_data.mmmdata_spec.rolling_window_start_which = 0
        self.mock_mmm_data.mmmdata_spec.rolling_window_end_which = 5
        select_model = "model1"
        metric_name = "spend_metric"
        metric_value = [100, 200, 300, 400, 500]
        date_range = "all"
        dt_hyppar = pd.DataFrame(
            {
                "sol_id": ["model1"],
                "spend_metric_alphas": [0.1],
                "spend_metric_gammas": [0.2],
            }
        )
        dt_coef = pd.DataFrame(
            {"sol_id": ["model1"], "rn": ["spend_metric"], "coef": [0.5]}
        )
        # Mock the transformation methods
        self.calculator.transformation = MagicMock()  # Ensure transformation is a mock
        self.calculator.transformation.transform_adstock.return_value = MagicMock(
            x_decayed=np.array([1, 2, 3]),
            x_imme=np.array([0.5, 1, 1.5]),
            x=np.array([1, 2, 3]),
        )

        response_output = self.calculator.calculate_response(
            select_model,
            metric_name,
            metric_value,
            date_range,
            False,
            dt_hyppar,
            dt_coef=dt_coef,
        )

        self.assertIsInstance(response_output, ResponseOutput)
        self.assertEqual(response_output.metric_name, metric_name)

    def test__which_usecase(self):
        usecase = self.calculator._which_usecase(None, None)
        self.assertEqual(usecase, UseCase.ALL_HISTORICAL_VEC)

        usecase = self.calculator._which_usecase(None, "selected")
        self.assertEqual(usecase, UseCase.SELECTED_HISTORICAL_VEC)

        usecase = self.calculator._which_usecase(10.0, None)
        self.assertEqual(usecase, UseCase.TOTAL_METRIC_DEFAULT_RANGE)

        usecase = self.calculator._which_usecase(10.0, "selected")
        self.assertEqual(usecase, UseCase.TOTAL_METRIC_SELECTED_RANGE)

        usecase = self.calculator._which_usecase([1.0, 2.0, 3.0], None)
        self.assertEqual(usecase, UseCase.UNIT_METRIC_DEFAULT_LAST_N)

        usecase = self.calculator._which_usecase([1.0, 2.0, 3.0], "selected")
        self.assertEqual(usecase, UseCase.UNIT_METRIC_SELECTED_DATES)

    def test__check_metric_type(self):
        self.mock_mmm_data.mmmdata_spec.paid_media_spends = ["spend_metric"]
        self.mock_mmm_data.mmmdata_spec.paid_media_vars = ["exposure_metric"]
        self.mock_mmm_data.mmmdata_spec.organic_vars = ["organic_metric"]

        metric_type = self.calculator._check_metric_type("spend_metric")
        self.assertEqual(metric_type, "spend")

        metric_type = self.calculator._check_metric_type("exposure_metric")
        self.assertEqual(metric_type, "exposure")

        metric_type = self.calculator._check_metric_type("organic_metric")
        self.assertEqual(metric_type, "organic")

        with self.assertRaises(ValueError):
            self.calculator._check_metric_type("unknown_metric")

    def test__check_metric_dates(self):

        all_dates = pd.Series(pd.date_range("2020-01-01", periods=5))
        self.mock_mmm_data.mmmdata_spec.rolling_window_start_which = 0
        self.mock_mmm_data.mmmdata_spec.rolling_window_end_which = 5

        metric_date_info = self.calculator._check_metric_dates(None, all_dates, False)
        self.assertIsInstance(metric_date_info, MetricDateInfo)
        self.assertEqual(len(metric_date_info.date_range_updated), 5)

        metric_date_info = self.calculator._check_metric_dates(
            "last_3", all_dates, False
        )
        self.assertEqual(len(metric_date_info.date_range_updated), 3)

        metric_date_info = self.calculator._check_metric_dates(
            ["2020-01-01", "2020-01-04"], all_dates, False
        )
        self.assertEqual(len(metric_date_info.date_range_updated), 4)

        with self.assertRaises(ValueError):
            self.calculator._check_metric_dates("invalid_range", all_dates, False)

    def test__check_metric_value(self):
        all_values = pd.Series([1, 2, 3, 4, 5])
        metric_loc = slice(0, 3)

        metric_value_info = self.calculator._check_metric_value(
            None, "metric_name", all_values, metric_loc
        )
        self.assertIsInstance(metric_value_info, MetricValueInfo)
        self.assertTrue((metric_value_info.metric_value_updated == [1, 2, 3]).all())

        metric_value_info = self.calculator._check_metric_value(
            10, "metric_name", all_values, metric_loc
        )
        self.assertTrue((metric_value_info.metric_value_updated == [10, 10, 10]).all())

        with self.assertRaises(ValueError):
            self.calculator._check_metric_value(
                [1, 2], "metric_name", all_values, metric_loc
            )

    def test__transform_exposure_to_spend(self):
        metric_name = "exposure_metric"
        metric_value_updated = np.array([100, 200, 300])
        all_values_updated = pd.Series([1, 2, 3, 4, 5])
        metric_loc = slice(0, 3)

        self.mock_mmm_data.mmmdata_spec.modNLS = {
            "results": pd.DataFrame(
                {
                    "channel": ["exposure_metric"],
                    "rsq_nls": [0.9],
                    "rsq_lm": [0.8],
                    "Vmax": [1000],
                    "Km": [10],
                    "coef_lm": [1],
                }
            )
        }

        updated_values = self.calculator._transform_exposure_to_spend(
            metric_name, metric_value_updated, all_values_updated, metric_loc
        )
        self.assertIsInstance(updated_values, pd.Series)

    def test__get_spend_name(self):
        self.mock_mmm_data.mmmdata_spec.paid_media_spends = ["spend_metric"]
        self.mock_mmm_data.mmmdata_spec.paid_media_vars = ["exposure_metric"]

        spend_name = self.calculator._get_spend_name("exposure_metric")
        self.assertEqual(spend_name, "spend_metric")

    def test__get_channel_hyperparams(self):
        dt_hyppar = pd.DataFrame(
            {
                "sol_id": ["model1"],
                "spend_metric_thetas": [[0.1]],
                "spend_metric_shapes": [[0.2]],
                "spend_metric_scales": [[0.3]],
            }
        )

        self.mock_hyperparameter.adstock = AdstockType.WEIBULL

        channel_hyperparams = self.calculator._get_channel_hyperparams(
            "model1", "spend_metric", dt_hyppar
        )
        self.assertIsInstance(channel_hyperparams, ChannelHyperparameters)

    def test__get_saturation_params(self):
        dt_hyppar = pd.DataFrame(
            {
                "sol_id": ["model1"],
                "spend_metric_alphas": [[0.1]],
                "spend_metric_gammas": [[0.2]],
            }
        )

        saturation_params = self.calculator._get_saturation_params(
            "model1", "spend_metric", dt_hyppar
        )
        self.assertIsInstance(saturation_params, ChannelHyperparameters)
