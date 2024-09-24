import unittest
from unittest.mock import Mock, patch
from robyn.modeling.model_executor import ModelExecutor
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from unittest.mock import MagicMock


class TestModelExecutor(unittest.TestCase):
    def setUp(self):
        self.mock_holidays_data = Mock(spec=HolidaysData)
        self.mock_hyperparameters = Mock(spec=Hyperparameters)
        self.mock_hyperparameters.copy = MagicMock(return_value=self.mock_hyperparameters)
        self.mock_hyperparameters.hyper_bound_list_updated = MagicMock(return_value=[(0.1, 1.0)] * 10)
        self.mock_calibration_input = Mock(spec=CalibrationInput)
        self.mock_featurized_mmm_data = Mock(spec=FeaturizedMMMData)
        self.mock_mmmdata = Mock(spec=MMMData)
        self.mock_mmmdata.mmmdata_spec = Mock()
        self.mock_mmmdata.mmmdata_spec.paid_media_vars = []
        self.mock_mmmdata.mmmdata_spec.paid_media_spends = []
        self.mock_mmmdata.mmmdata_spec.organic_vars = []
        self.mock_mmmdata.mmmdata_spec.context_vars = []
        self.mock_mmmdata.mmmdata_spec.prophet_vars = []

        self.executor = ModelExecutor(
            self.mock_mmmdata,
            self.mock_holidays_data,
            self.mock_hyperparameters,
            self.mock_calibration_input,
            self.mock_featurized_mmm_data,
        )

    @patch("robyn.modeling.ridge_model_builder.RidgeModelBuilder")
    def test_model_run_ridge(self, mock_ridge_builder):
        mock_ridge_builder_instance = mock_ridge_builder.return_value
        mock_ridge_builder_instance.build_models.return_value = Mock()

        trials_config = TrialsConfig(trials=1, iterations=100)
        result = self.executor.model_run(trials_config=trials_config, model_name=Models.RIDGE)

        mock_ridge_builder.assert_called_once()
        mock_ridge_builder_instance.build_models.assert_called_once()
        self.assertIsNotNone(result)

    def test_model_run_unsupported_model(self):
        with self.assertRaises(NotImplementedError):
            self.executor.model_run(model_name="UnsupportedModel")

    def test_generate_additional_outputs(self):
        mock_model_outputs = Mock()
        mock_model_outputs.trials = 5
        mock_model_outputs.results = [Mock(performance=0.8) for _ in range(5)]

        additional_outputs = self.executor._generate_additional_outputs(mock_model_outputs)

        self.assertIn("average_performance", additional_outputs)
        self.assertAlmostEqual(additional_outputs["average_performance"], 0.8)


if __name__ == "__main__":
    unittest.main()
