# Following methods from checks module should go here.
# def check_calibration(dt_input, date_var, calibration_input, dayInterval, dep_var,window_start, window_end, paid_media_spends, organic_vars):
# def check_obj_weight(calibration_input, objective_weights, refresh):
# def check_iteration(calibration_input, iterations, trials, hyps_fixed, refresh):


from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.mmmdata_collection import TimeWindow
from robyn.data.validation.validation import Validation, ValidationResult


@dataclass
class CalibrationInputValidation:
    """
    CalibrationInputValidation class to validate calibration Input in mmmdata_collection.
    """

    def __init__(self, calibrationInput: CalibrationInput) -> None:
        self.calibrationInput: CalibrationInput = calibrationInput

    def check_calibration_input(
        self, mmm_data: MMMData, time_window: TimeWindow, day_interval: int
    ) -> ValidationResult:
        """
        Check if the calibration input is valid.
        """
        invalid_variables: List[str] = []
        errors: List[str] = []
        raise NotImplementedError("Not yet implemented")

    def check_objective_weights(
        self, objective_weights: Optional[List[float]], refresh: bool
    ) -> ValidationResult:
        """
        Check if the objective weights are valid.
        """
        raise NotImplementedError("Not yet implemented")

    def check_iteration(
        self, iterations: int, trials: int, hyps_fixed: bool, refresh: bool
    ) -> ValidationResult:
        """
        Check if the iteration and trials are valid.
        """
        raise NotImplementedError("Not yet implemented")
