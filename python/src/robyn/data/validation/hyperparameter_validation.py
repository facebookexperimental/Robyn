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
    ) -> Optional[pd.DataFrame]:
        """
        Check if the hyperparameters are valid.

        Args:
            adstock (AdstockType): The adstock type.
            mmm_data (MMMData): The organic variables and paid media spends.
            exposure_vars (Optional[List[str]]): The exposure variables.

        Returns:
            Optional[pd.DataFrame]: The validated hyperparameters.
        """

        return None

    def _check_train_size(self) -> bool:
        """
        Check if the train size is valid.

        Returns:
            bool: True if the train size is valid, False otherwise.
        """
        if "train_size" in self.hyperparameters:
            if (
                not 1 <= len(self.hyperparameters["train_size"]) <= 2
                or any(self.hyperparameters["train_size"] <= 0.1)
                or any(self.hyperparameters["train_size"] > 1)
            ):
                return False
        return True

    def _check_hyperparameter_limits(
        hyperparameters: pd.DataFrame, column: str
    ) -> None:
        """
        Check if the hyperparameters dataframe are within the limits.

        Args:
            hyperparameters (pd.DataFrame): The dataframe to check.
            column (str): The column in hyperparameter to check.

        Returns:
            None
        """
        return

    def _combine_filtered_elements(
        filter_parameters: List[str],
        all_hyperparameters: List[str],
        all_media: List[str],
    ) -> List[str]:
        """
        Combine the filtered elements.

        Args:
            filter_parameters (List[str]): The filter parameters.
            all_hyperparameters (List[str]): all hyperparameters.
            all_media (List[str]): The all media.

        Returns:
            List[str]: The combined elements.
        """
        return all_media

    def hyper_names(self, adstock: AdstockType, all_media: List[str]) -> List[str]:
        """
        Returns the sorted hyperparameter column names.

        Args:
            adstock (AdstockType): The adstock type.
            all_media (List[str]): The list of all media.

        Returns:
            List[str]: The hyperparameter column names.
        """
        return self._combine_filtered_elements([], [], all_media)
