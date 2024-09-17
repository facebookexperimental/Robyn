# pyre-strict

import numpy as np
from typing import Dict, Union
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.mmmdata import MMMData


class Transformation:
    def __init__(self, input_data: np.ndarray):
        self.input_data = input_data

    def michaelis_menten(self, vmax: float, km: float, reverse: bool = False) -> np.ndarray:
        """
        Michaelis-Menten Transformation

        Args:
            vmax (float): Maximum rate achieved by the system
            km (float): Michaelis constant
            reverse (bool): If True, reverse the transformation

        Returns:
            np.ndarray: Transformed values
        """
        pass

    def adstock_geometric(self, theta: float) -> Dict[str, Union[np.ndarray, float]]:
        """
        Geometric Adstocking

        Args:
            theta (float): Decay rate

        Returns:
            Dict[str, Union[np.ndarray, float]]: Dictionary containing transformed values and metadata
        """
        pass

    def adstock_weibull(
        self, shape: float, scale: float, window_length: int = None, adstock_type: str = "cdf"
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Weibull Adstocking

        Args:
            shape (float): Shape parameter
            scale (float): Scale parameter
            window_length (int): Window length
            adstock_type (str): Type of Weibull function ("cdf" or "pdf")

        Returns:
            Dict[str, Union[np.ndarray, float]]: Dictionary containing transformed values and metadata
        """
        pass

    def transform_adstock(
        self,
        adstock: AdstockType,
        theta: float = None,
        shape: float = None,
        scale: float = None,
        window_length: int = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Transform using specified adstock method

        Args:
            adstock (AdstockType): Type of adstock transformation
            theta (float): Theta parameter for geometric adstock
            shape (float): Shape parameter for Weibull adstock
            scale (float): Scale parameter for Weibull adstock
            window_length (int): Window length for Weibull adstock

        Returns:
            Dict[str, Union[np.ndarray, float]]: Dictionary containing transformed values and metadata
        """
        pass

    def saturation_hill(self, alpha: float, gamma: float, marginal_input: np.ndarray = None) -> np.ndarray:
        """
        Hill Saturation Transformation

        Args:
            alpha (float): Alpha parameter
            gamma (float): Gamma parameter
            marginal_input (np.ndarray): Marginal input values

        Returns:
            np.ndarray: Transformed values
        """
        pass

    @staticmethod
    def weibull_cdf(input_data: np.ndarray, shape: float, scale: float) -> np.ndarray:
        """
        Weibull Cumulative Distribution Function

        Args:
            input_data (np.ndarray): Input values
            shape (float): Shape parameter
            scale (float): Scale parameter

        Returns:
            np.ndarray: CDF values
        """
        pass

    @staticmethod
    def weibull_pdf(input_data: np.ndarray, shape: float, scale: float) -> np.ndarray:
        """
        Weibull Probability Density Function

        Args:
            input_data (np.ndarray): Input values
            shape (float): Shape parameter
            scale (float): Scale parameter

        Returns:
            np.ndarray: PDF values
        """
        pass

    @staticmethod
    def normalize(input_data: np.ndarray) -> np.ndarray:
        """
        Normalize the input data

        Args:
            input_data (np.ndarray): Input values

        Returns:
            np.ndarray: Normalized values
        """
        pass


class MediaTransformer:
    def __init__(self, mmm_data: MMMData, hyperparameters: Hyperparameters):
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters

    def transform_media(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Transform media data using adstock and saturation methods

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
                - Dictionary of transformed media data
                - Dictionary of immediate effects
                - Dictionary of carryover effects
        """
        pass
