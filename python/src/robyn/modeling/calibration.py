#pyre-strict

from typing import Optional, Dict
import pandas as pd

from mmmdata import MMMData
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.modeling.entities.calibration_result import CalibrationResult

class Calibration:

    #TODO review and update attributes
    def __init__(self,
                 calibration_input: CalibrationInput,
                 mmmdata: MMMData,
                 dayInterval: int,
                 xDecompVec: pd.DataFrame,
                 coefs: pd.DataFrame,
                 hypParamSam: Dict[str, float],
                 wind_start: int = 1,
                 wind_end: Optional[int] = None,
                 adstock: AdstockType = None) -> None:
        """
        Initializes a Calibration object.
        Args:
            calibration_input (CalibrationInput): The calibration input.
            mmmdata (MMMData): The MMMData object.
            dayInterval (int): The day interval.
            xDecompVec (pd.DataFrame): The decomposition vector.
            coefs (pd.DataFrame): The coefficients.
            hypParamSam (Dict[str, float]): The hyperparameter sample.
            wind_start (int, optional): The start of the window. Defaults to 1.
            wind_end (int, optional): The end of the window. Defaults to None.
            adstock (str, optional): The adstock. Defaults to None.
        """
        self.calibration_input = calibration_input
        self.mmmdata = mmmdata
        self.dayInterval = dayInterval
        self.xDecompVec = xDecompVec
        self.coefs = coefs
        self.hypParamSam = hypParamSam
        self.wind_start = wind_start
        self.wind_end = wind_end
        self.adstock = adstock


    def calibrate(self) -> CalibrationResult:
        """
        Performs calibration on the provided input data.

        Returns:
        -------
        CalibrationResult
            The calibrated data.
        """
    pass
