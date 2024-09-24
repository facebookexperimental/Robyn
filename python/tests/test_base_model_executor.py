import unittest
from unittest.mock import Mock
from robyn.modeling.base_model_executor import BaseModelExecutor
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.modeling.feature_engineering import FeaturizedMMMData


class TestBaseModelExecutor(unittest.TestCase):
    def setUp(self):
        self.mock_holidays_data = Mock(spec=HolidaysData)
        self.mock_hyperparameters = Mock(spec=Hyperparameters)
        self.mock_calibration_input = Mock(spec=CalibrationInput)
        self.mock_featurized_mmm_data = Mock(spec=FeaturizedMMMData)
        self.mock_mmmdata = Mock(spec=MMMData)
        self.mock_mmmdata.mmmdata_spec = Mock()
        self.mock_mmmdata.mmmdata_spec.paid_media_vars = []
        self.mock_mmmdata.mmmdata_spec.paid_media_spends = []
        self.mock_mmmdata.mmmdata_spec.organic_vars = []
        self.mock_mmmdata.mmmdata_spec.context_vars = []
        self.mock_mmmdata.mmmdata_spec.prophet_vars = []

    def test_abstract_model_run(self):
        class ConcreteModelExecutor(BaseModelExecutor):
            def model_run(self, **kwargs):
                return "Model run completed"

        executor = ConcreteModelExecutor(
            self.mock_mmmdata,
            self.mock_holidays_data,
            self.mock_hyperparameters,
            self.mock_calibration_input,
            self.mock_featurized_mmm_data,
        )

        result = executor.model_run()
        self.assertEqual(result, "Model run completed")

    def test_validate_input(self):
        class ConcreteModelExecutor(BaseModelExecutor):
            def model_run(self, **kwargs):
                pass

        executor = ConcreteModelExecutor(
            self.mock_mmmdata,
            self.mock_holidays_data,
            self.mock_hyperparameters,
            self.mock_calibration_input,
            self.mock_featurized_mmm_data,
        )

        # Test with valid inputs
        executor._validate_input()  # Should not raise an exception

        # Test with invalid inputs
        executor.mmmdata = None
        with self.assertRaises(ValueError):
            executor._validate_input()


if __name__ == "__main__":
    unittest.main()
