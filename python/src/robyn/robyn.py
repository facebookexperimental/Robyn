# pyre-strict

# TODO This needs to be rewritten to match the new structure of the codebase
# TODO Add separate methods if state is loaded from robyn_object or json_file for each method

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.calibration_input_validation import CalibrationInputValidation
from robyn.data.validation.holidays_data_validation import HolidaysDataValidation
from robyn.data.validation.hyperparameter_validation import HyperparametersValidation
from robyn.data.validation.mmmdata_validation import MMMDataValidation



class Robyn:
    def __init__(self, working_dir: str):
        """
        Initializes the Robyn object with a working directory.

        Args:
            working_dir (str): The path to the working directory.
        """
        self.working_dir = working_dir

    # Load input data for the first time and validates
    def initialize(
        self,
        mmm_data: MMMData,
        holidays_data: HolidaysData,
        hyperparameters: Hyperparameters,
        calibration_input: CalibrationInput,
    ) -> None:
        """
        Loads input data for the first time and validates it.
        Calls validate from MMMDataValidation, HolidaysDataValidation, HyperparametersValidation, and CalibrationInputValidation.

        Args:
            mmm_data (MMMData): The MMM data object.
            holidays_data (HolidaysData): The holidays data object.
            hyperparameters (HyperParametersConfig): The hyperparameters configuration object.
            calibration_input (CalibrationInputConfig): The calibration input configuration object.
        """

        mmm_data_validation = MMMDataValidation(mmm_data)
        holidays_data_validation = HolidaysDataValidation(holidays_data)
        hyperparameters_validation = HyperparametersValidation(hyperparameters)
        calibration_input_validation = CalibrationInputValidation(mmm_data, calibration_input)

        mmm_data_validation_results = mmm_data_validation.validate()
        print(mmm_data_validation_results)
        holidays_data_validation.validate()
        hyperparameters_validation.validate()
        calibration_input_validation.validate()

        print("Validation complete")
