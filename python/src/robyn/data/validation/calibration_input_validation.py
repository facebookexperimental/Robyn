# Following methods from checks module should go here.
# def check_calibration(dt_input, date_var, calibration_input, dayInterval, dep_var,window_start, window_end, paid_media_spends, organic_vars):
# def check_obj_weight(calibration_input, objective_weights, refresh):
# def check_iteration(calibration_input, iterations, trials, hyps_fixed, refresh):


from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.mmmdata_collection import TimeWindow


@dataclass
class CalibrationInputValidation:
    """
    CalibrationInputValidation class to validate calibration Input in mmmdata_collection.
    """

    def __init__(self, calibrationInput: CalibrationInput) -> None:
        self.calibrationInput: CalibrationInput = calibrationInput

    def check_calibration_input(
        self, mmm_data: MMMData, time_window: TimeWindow, day_interval: int
    ) -> Dict[str, List[str]]:
        """
        Check if the calibration input is valid.
        """
        invalid_variables: List[str] = []
        errors: List[str] = []

        return {"invalid_variables": invalid_variables, "errors": errors}

    def check_objective_weights(
        self, objective_weights: Optional[List[float]], refresh: bool
    ) -> bool:
        """
        Check if the objective weights are valid.
        """
        return True

    def check_iteration(
        self, iterations: int, trials: int, hyps_fixed: bool, refresh: bool
    ) -> Optional[str]:
        """
        Check if the iteration and trials are valid.
        """
        return None
