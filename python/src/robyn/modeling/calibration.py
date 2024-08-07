#pyre-strict

import pandas as pd
from typing import Optional, List, Dict
from data import CalibrationInput

class Calibration:
    """
    A class used to perform calibration on the provided input data.

    Attributes:
    ----------
    calibration_input : pd.DataFrame
        The input data for calibration.
    df_raw : pd.DataFrame
        The raw data.
    dayInterval : int
        The day interval.
    xDecompVec : pd.DataFrame
        The decomposed vector.
    coefs : pd.DataFrame
        The coefficients.
    hypParamSam : Dict[str, float]
        The hyperparameter samples.
    wind_start : int
        The start of the window (default is 1).
    wind_end : int
        The end of the window (default is the number of rows in df_raw).
    adstock : str
        The adstock type (default is None).

    Methods:
    -------
    calibrate()
        Performs calibration on the provided input data.
    """

    def __init__(self,
                 calibration_input: CalibrationInput,
                 df_raw: pd.DataFrame,
                 dayInterval: int,
                 xDecompVec: pd.DataFrame,
                 coefs: pd.DataFrame,
                 hypParamSam: Dict[str, float],
                 wind_start: int = 1,
                 wind_end: Optional[int] = None,
                 adstock: Optional[str] = None):
        """
        Initializes the Calibration class.

        Args:
        ----
        calibration_input : pd.DataFrame
            The input data for calibration.
        df_raw : pd.DataFrame
            The raw data.
        dayInterval : int
            The day interval.
        xDecompVec : pd.DataFrame
            The decomposed vector.
        coefs : pd.DataFrame
            The coefficients.
        hypParamSam : Dict[str, float]
            The hyperparameter samples.
        wind_start : int
            The start of the window (default is 1).
        wind_end : int
            The end of the window (default is the number of rows in df_raw).
        adstock : str
            The adstock type (default is None).
        """
        self.calibration_input = calibration_input
        self.df_raw = df_raw
        self.dayInterval = dayInterval
        self.xDecompVec = xDecompVec
        self.coefs = coefs
        self.hypParamSam = hypParamSam
        self.wind_start = wind_start
        self.wind_end = wind_end if wind_end else len(df_raw)
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
