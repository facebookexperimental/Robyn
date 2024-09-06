from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from robyn.data.entities.enums import (
    DependentVarType,
    AdstockType,
    SaturationType,
    ProphetVariableType,
    PaidMediaSigns,
    OrganicSigns,
    ContextSigns,
    ProphetSigns,
    CalibrationScope,
)

from robyn.data.entities.calibration_input import CalibrationInput, ChannelCalibrationData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.mmmdata import MMMData


@dataclass
class FeaturizedMMMData:
    """
    A dataclass to hold the output of the feature engineering process.

    Attributes:
        dt_mod (pd.DataFrame): The transformed data after feature engineering.
        dt_modRollWind (pd.DataFrame): The rolling window data used in the modeling process.
        modNLS (Dict[str, Any]): Results of the non-linear models, including fitted models, plots, and predictions.
    """

    dt_mod: pd.DataFrame
    dt_modRollWind: pd.DataFrame
    modNLS: Dict[str, Any]


class FeatureEngineering:
    """
    A class to perform feature engineering for Marketing Mix Modeling (MMM).

    This class handles various aspects of feature engineering, including data preparation,
    rolling window creation, media cost factor calculation, non-linear modeling,
    and prophet decomposition.

    Attributes:
        mmm_data (MMMData): The input data and specifications for the MMM.
        hyperparameters (Hyperparameters): The hyperparameters for the model.
    """

    def __init__(self, mmm_data: MMMData, hyperparameters: Hyperparameters):
        """
        Initialize the FeatureEngineering class.

        Args:
            mmm_data (MMMData): The input data and specifications for the MMM.
            hyperparameters (Hyperparameters): The hyperparameters for the model.
        """
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters

    def perform_feature_engineering(self, quiet: bool = False) -> FeaturizedMMMData:
        """
        Perform the feature engineering process.

        This method orchestrates the entire feature engineering pipeline, including
        data preparation, rolling window creation, media cost factor calculation,
        model running, and prophet decomposition (if applicable).

        Args:
            quiet (bool, optional): If True, suppresses print statements. Defaults to False.

        Returns:
            FeaturizedMMMData: The output of the feature engineering process.
        """
        pass

    def _prepare_data(self) -> pd.DataFrame:
        """
        Prepare the input data for feature engineering.

        Returns:
            pd.DataFrame: The prepared data.
        """
        pass

    def _create_rolling_window_data(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window data from the transformed data.

        Args:
            dt_transform (pd.DataFrame): The transformed data.

        Returns:
            pd.DataFrame: The rolling window data.
        """
        pass

    def _calculate_media_cost_factor(self, dt_input_roll_wind: pd.DataFrame) -> pd.Series:
        """
        Calculate the media cost factor.

        Args:
            dt_input_roll_wind (pd.DataFrame): The rolling window data.

        Returns:
            pd.Series: The media cost factor.
        """
        pass

    def _run_models(self, dt_input_roll_wind: pd.DataFrame, media_cost_factor: pd.Series) -> Dict[str, Any]:
        """
        Run models on the rolling window data.

        Args:
            dt_input_roll_wind (pd.DataFrame): The rolling window data.
            media_cost_factor (pd.Series): The media cost factor.

        Returns:
            Dict[str, Any]: The model results, including fitted models, plots, and predictions.
        """
        pass

    def _fit_spend_exposure(
        self, dt_spend_mod_input: pd.DataFrame, media_cost_factor: float, paid_media_var: str
    ) -> Dict[str, Any]:
        """
        Fit the spend exposure model.

        Args:
            dt_spend_mod_input (pd.DataFrame): The spend and exposure data.
            media_cost_factor (float): The media cost factor.
            paid_media_var (str): The name of the paid media variable.

        Returns:
            Dict[str, Any]: The model results, plot data, and predictions.
        """
        pass

    def _prophet_decomposition(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose the time series data using the Prophet model.

        Args:
            dt_transform (pd.DataFrame): The DataFrame containing the time series data to be decomposed.

        Returns:
            pd.DataFrame: The DataFrame with added columns for the decomposed components.
        """
        pass

    def _set_holidays(self) -> pd.DataFrame:
        """
        Set up the holidays for the Prophet model.

        Returns:
            pd.DataFrame: The DataFrame with holidays.
        """
        pass

    def _apply_transformations(self, x: pd.Series, params: ChannelHyperparameters) -> pd.Series:
        """
        Apply adstock and saturation transformations to the input series.

        Args:
            x (pd.Series): Input series to transform.
            params (ChannelHyperparameters): Hyperparameters for the channel.

        Returns:
            pd.Series: Transformed series.
        """
        pass
