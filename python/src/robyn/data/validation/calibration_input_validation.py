from calibration_input import CalibrationInput
from mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
from typing import List

from robyn.modeling.entities.modelrun_trials_config import TrialsConfig

class CalibrationInputValidation(Validation):
    def __init__(self, mmmdata: MMMData, calibration_input: CalibrationInput) -> None:
        self.mmmdata = mmmdata
        self.calibration_input = calibration_input

    def check_calibration(
        self, mmmdata: MMMData, calibration_input: CalibrationInput, window_start: int, window_end: int
    ) -> ValidationResult:
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
        trials_config: TrialsConfig,
        hyps_fixed: bool,
        refresh: bool,
    ) -> ValidationResult:
        # method implementation goes here
        raise NotImplementedError("Not yet implemented")

    def validate(self) -> ValidationResult:
        raise NotImplementedError("Not yet implemented")