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
        if not quiet:
            print(">> Running Robyn feature engineering...")

        dt_transform = self._prepare_data()
        dt_transform_roll_wind = self._create_rolling_window_data(dt_transform)
        media_cost_factor = self._calculate_media_cost_factor(dt_transform_roll_wind)
        model_results = self._run_models(dt_transform_roll_wind, media_cost_factor)

        if self.mmm_data.mmmdata_spec.prophet_vars:
            dt_transform = self._prophet_decomposition(dt_transform)

        self._check_no_variance(dt_transform)

        return FeaturizedMMMData(dt_mod=dt_transform, dt_modRollWind=dt_transform_roll_wind, modNLS=model_results)

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

    def _check_no_variance(self, dt_transform: pd.DataFrame) -> None:
        """
        Check for columns with no variance in the transformed data.

        Args:
            dt_transform (pd.DataFrame): The transformed data to check.
        """
        pass

    @staticmethod
    def hyper_names(adstock: str, all_media: List[str], all_vars: Optional[List[str]] = None) -> List[str]:
        """
        Get the names of all hyperparameters.

        Args:
            adstock (str): The type of adstock transformation.
            all_media (List[str]): List of all media variables.
            all_vars (Optional[List[str]], optional): List of all variables. Defaults to None.

        Returns:
            List[str]: The names of all hyperparameters.
        """
        pass

    @staticmethod
    def hyper_limits() -> pd.DataFrame:
        """
        Get the limits for hyperparameters.

        Returns:
            pd.DataFrame: A DataFrame containing the upper and lower bounds for each hyperparameter.
        """
        pass
