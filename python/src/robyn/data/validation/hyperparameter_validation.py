# Following methods from checks module should go here.
# def check_hyperparameters(hyperparameters=None, adstock=None, paid_media_spends=None, organic_vars=None, exposure_vars=None):
# def check_train_size(hyps):
# def check_hyper_limits(hyperparameters, hyper):

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

from robyn.data.entities.enums import AdstockType
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult


@dataclass
class HyperparameterValidation:
    """
    HyperparameterValidation class to validate model hyperparameters.
    """

    HYPS_NAMES = ["thetas", "shapes", "scales", "alphas", "gammas", "penalty"]
    HYPS_OTHERS = ["lambda", "train_size"]

    def __init__(self, hyperparameters: Hyperparameters) -> None:
        self.hyperparameters: Hyperparameters = hyperparameters

    def check_hyperparameters(
        self,
        adstock: AdstockType,
        mmm_data: MMMData,
        exposure_vars: Optional[List[str]],
    ) -> ValidationResult:
        """
        Check if the hyperparameters are valid.

        Args:
            adstock (AdstockType): The adstock type.
            mmm_data (MMMData): The organic variables and paid media spends.
            exposure_vars (Optional[List[str]]): The exposure variables.

        Returns:
            Optional[pd.DataFrame]: The validated hyperparameters.
        """
        raise NotImplementedError("Not yet implemented")

    def _check_train_size(self) -> ValidationResult:
        """
        Check if the train size is valid.

        Returns:
            bool: True if the train size is valid, False otherwise.
        """
        raise NotImplementedError("Not yet implemented")

    def _check_hyperparameter_limits(
        hyperparameters: pd.DataFrame, column: str
    ) -> ValidationResult:
        """
        Check if the hyperparameters dataframe are within the limits.

        Args:
            hyperparameters (pd.DataFrame): The dataframe to check.
            column (str): The column in hyperparameter to check.

        Returns:
            ValidationResult
        """
        raise NotImplementedError("Not yet implemented")

    def _combine_filtered_elements(
        filter_parameters: List[str],
        all_hyperparameters: List[str],
        all_media: List[str],
    ) -> ValidationResult:
        """
        Combine the filtered elements.

        Args:
            filter_parameters (List[str]): The filter parameters.
            all_hyperparameters (List[str]): all hyperparameters.
            all_media (List[str]): The all media.

        Returns:
            List[str]: The combined elements.
        """
        raise NotImplementedError("Not yet implemented")

    def hyper_names(
        self, adstock: AdstockType, all_media: List[str]
    ) -> ValidationResult:
        """
        Returns the sorted hyperparameter column names.

        Args:
            adstock (AdstockType): The adstock type.
            all_media (List[str]): The list of all media.

        Returns:
            List[str]: The hyperparameter column names.
        """
        raise NotImplementedError("Not yet implemented")

    def check_hyper_fixed(
        self,
        adstock: AdstockType,
        all_media: List[str],
        dt_mod: pd.DataFrame,
        dt_hyper_fixed: Optional[pd.DataFrame],
        add_penalty_factor: bool,
    ) -> ValidationResult:
        """
        Checks if hyperparameters are fixed and adjusts the list of hyperparameter names based on the input data and conditions.
        Args:
            adstock (AdstockType): Adstock data.
            all_media (List[str]): List of all media types.
            dt_mod (pd.DataFrame): DataFrame containing model data.
            dt_hyper_fixed (Optional[pd.DataFrame]): DataFrame containing fixed hyperparameters, can be None.
            add_penalty_factor (bool): Flag to determine if penalty factors should be added.
        Returns:
            ValidationResult: ValidationResult containing the status of hyperparameters being fixed and the list of hyperparameter names.
        """
        raise NotImplementedError("Not yet implemented")
