# pyre-strict

from typing import TypedDict, List
import pandas as pd

class CalibrationInput(TypedDict):
    channel: List[str]
    liftStartDate: pd.Series
    liftEndDate: pd.Series
    liftAbs: List[int]
    spend: List[int]
    confidence: List[float]
    metric: List[str]
    calibration_scope: List[str]

class CalibrationInputConfig:
    def __init__(self, calibration_input: CalibrationInput) -> None:
        self.calibration_input: CalibrationInput = calibration_input

    def __str__(self) -> str:
        return (
            f"CalibrationInputConfig:\n"
            f"channel: {self.calibration_input['channel']}\n"
            f"liftStartDate: {self.calibration_input['liftStartDate']}\n"
            f"liftEndDate: {self.calibration_input['liftEndDate']}\n"
            f"liftAbs: {self.calibration_input['liftAbs']}\n"
            f"spend: {self.calibration_input['spend']}\n"
            f"confidence: {self.calibration_input['confidence']}\n"
            f"metric: {self.calibration_input['metric']}\n"
            f"calibration_scope: {self.calibration_input['calibration_scope']}\n"
        )

    def update(self, **kwargs: object) -> None:
        """
        Update the attributes of the CalibrationInputConfig object.
        
        :param kwargs: Keyword arguments corresponding to the attributes to update.
        """
        for key, value in kwargs.items():
            if key in self.calibration_input:
                self.calibration_input[key] = value
            else:
                raise AttributeError(f"{key} is not a valid attribute of CalibrationInputConfig")

# Example usage:
if __name__ == "__main__":
    # Create a sample calibration_input DataFrame
    calibration_input: CalibrationInput = {
        "channel": ["facebook_S", "tv_S", "facebook_S+search_S", "newsletter"],
        "liftStartDate": pd.to_datetime(["2018-05-01", "2018-04-03", "2018-07-01", "2017-12-01"]),
        "liftEndDate": pd.to_datetime(["2018-06-10", "2018-06-03", "2018-07-20", "2017-12-31"]),
        "liftAbs": [400000, 300000, 700000, 200],
        "spend": [421000, 7100, 350000, 0],
        "confidence": [0.85, 0.8, 0.99, 0.95],
        "metric": ["revenue", "revenue", "revenue", "revenue"],
        "calibration_scope": ["immediate", "immediate", "immediate", "immediate"]
    }

    # Initialize CalibrationInputConfig
    calib_config: CalibrationInputConfig = CalibrationInputConfig(calibration_input)
    print(calib_config)

    # Update some attributes in CalibrationInputConfig
    calib_config.update(metric=["revenue", "revenue", "revenue", "profit"])

    # Print the updated CalibrationInputConfig object
    print(calib_config)
