import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from robyn.modeling.model_executor import ModelExecutor
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.data.entities.enums import AdstockType


class TestModelIntegration(unittest.TestCase):
    def setUp(self):
        # Create mock data
        self.mock_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2020-01-01", periods=100),
                "sales": np.random.rand(100) * 1000,
                "tv_spend": np.random.rand(100) * 100,
                "radio_spend": np.random.rand(100) * 50,
            }
        )

        # Create a more detailed mock for MMMData
        self.mock_mmmdata = Mock(spec=MMMData)
        self.mock_mmmdata.data = self.mock_data
        self.mock_mmmdata.mmmdata_spec = Mock()
        self.mock_mmmdata.mmmdata_spec.dep_var = "sales"
        self.mock_mmmdata.mmmdata_spec.date_var = "date"
        self.mock_mmmdata.mmmdata_spec.paid_media_spends = ["tv_spend", "radio_spend"]
        self.mock_mmmdata.mmmdata_spec.paid_media_vars = ["tv_spend", "radio_spend"]
        self.mock_mmmdata.mmmdata_spec.organic_vars = []
        self.mock_mmmdata.mmmdata_spec.context_vars = []
        self.mock_mmmdata.mmmdata_spec.prophet_vars = []

        self.mock_holidays_data = Mock(spec=HolidaysData)
        self.mock_hyperparameters = Hyperparameters(
            hyperparameters={
                "tv_spend": {"alphas": [0.5, 1.5], "gammas": [0.3, 0.7], "thetas": [0.1, 0.9]},
                "radio_spend": {"alphas": [0.5, 1.5], "gammas": [0.3, 0.7], "thetas": [0.1, 0.9]},
            },
            adstock=AdstockType.GEOMETRIC,
            lambda_=[0.1, 10],
            train_size=[0.6, 0.8],
        )
        self.mock_calibration_input = Mock(spec=CalibrationInput)
        self.mock_featurized_mmm_data = Mock(spec=FeaturizedMMMData)
        self.mock_featurized_mmm_data.dt_mod = self.mock_data

    @patch("robyn.modeling.ridge_model_builder.RidgeModelBuilder.build_models")
    def test_end_to_end_model_run(self, mock_build_models):
        # Mock the build_models method to return a predefined output
        mock_build_models.return_value = Mock(
            trials=[Mock(nrmse=0.1, decomp_rssd=0.2, mape=0.05) for _ in range(5)],
            train_timestamp="2023-06-01 10:00:00",
            cores=4,
            iterations=100,
            intercept=True,
            intercept_sign="non_negative",
            nevergrad_algo="TwoPointsDE",
            ts_validation=True,
            add_penalty_factor=False,
            hyper_updated={"tv_spend_alpha": [0.5, 1.5], "radio_spend_alpha": [0.5, 1.5]},
            hyper_fixed=False,
            convergence={},
            ts_validation_plot=None,
            select_id="1_1_1",
            seed=123,
        )

        executor = ModelExecutor(
            self.mock_mmmdata,
            self.mock_holidays_data,
            self.mock_hyperparameters,
            self.mock_calibration_input,
            self.mock_featurized_mmm_data,
        )

        trials_config = TrialsConfig(trials=5, iterations=100)
        result = executor.model_run(
            trials_config=trials_config,
            model_name=Models.RIDGE,
            nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
            ts_validation=True,
            add_penalty_factor=False,
            intercept=True,
            intercept_sign="non_negative",
            seed=123,
        )

        # Assertions to check if the model run produced expected results
        self.assertEqual(len(result.trials), 5)
        self.assertEqual(result.iterations, 100)
        self.assertTrue(result.ts_validation)
        self.assertFalse(result.add_penalty_factor)
        self.assertEqual(result.nevergrad_algo, "TwoPointsDE")
        self.assertIsNotNone(result.hyper_updated)
        self.assertFalse(result.hyper_fixed)

        # Check if the build_models method was called with correct parameters
        mock_build_models.assert_called_once()
        call_args = mock_build_models.call_args[1]
        self.assertEqual(call_args["trials_config"], trials_config)
        self.assertTrue(call_args["ts_validation"])
        self.assertFalse(call_args["add_penalty_factor"])
        self.assertEqual(call_args["nevergrad_algo"], NevergradAlgorithm.TWO_POINTS_DE)


if __name__ == "__main__":
    unittest.main()
