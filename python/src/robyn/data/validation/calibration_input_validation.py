from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
from typing import List

# from robyn.modeling.entities.modelrun_trials_config import TrialsConfig

class CalibrationInputValidation(Validation):
    def __init__(self, mmmdata: MMMData, calibration_input: CalibrationInput) -> None:
        self.mmmdata = mmmdata
        self.calibration_input = calibration_input

    def check_calibration(
        self, mmmdata: MMMData, calibration_input: CalibrationInput, window_start: int, window_end: int
    ) -> ValidationResult:
        """
        This function checks the calibration input data for consistency and correctness.
        It verifies that the input data contains the required columns, that the date range
        is within the modeling window, and that the spend values match the input data.
        """
        # method implementation goes here
        raise NotImplementedError("Not yet implemented")

    def check_obj_weight(
        self, calibration_input: CalibrationInput, objective_weights: List[float], refresh: bool
    ) -> ValidationResult:
        # method implementation goes here
        raise NotImplementedError("Not yet implemented")

    def check_iteration(
        self,
        calibration_input: CalibrationInput,
        # trials_config: TrialsConfig,
        hyps_fixed: bool,
        refresh: bool,
    ) -> ValidationResult:
        # method implementation goes here
        raise NotImplementedError("Not yet implemented")

    def validate(self) -> ValidationResult:
        pass
        #raise NotImplementedError("Not yet implemented")