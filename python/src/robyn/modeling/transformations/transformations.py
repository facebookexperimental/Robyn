# pyre-strict

import numpy as np
from typing import Dict, Union
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.mmmdata import MMMData


class Transformation:
    def __init__(self, mmm_data: MMMData):
        """
        Initialize the Transformation class with MMMData.

        Args:
            mmm_data (MMMData): The MMMData object containing all input data.
        """
        self.mmm_data = mmm_data
        self.media_channels = self._get_media_channels()
        self.channel_data = self._get_channel_data()

    def _get_media_channels(self) -> list:
        """
        Get all media channels from MMMData.

        Returns:
            list: List of all media channel names.
        """
        return (self.mmm_data.mmmdata_spec.paid_media_vars or []) + (
            self.mmm_data.mmmdata_spec.paid_media_spends or []
        )

    def _get_channel_data(self) -> Dict[str, np.ndarray]:
        """
        Extract the data for all media channels.

        Returns:
            Dict[str, np.ndarray]: Dictionary with channel names as keys and their data as values.
        """
        return {channel: self.mmm_data.data[channel].values for channel in self.media_channels}

    def michaelis_menten(self, channel: str, vmax: float, km: float, reverse: bool = False) -> np.ndarray:
        """
        Michaelis-Menten Transformation

        Args:
            channel (str): Name of the media channel
            vmax (float): Maximum rate achieved by the system
            km (float): Michaelis constant
            reverse (bool): If True, reverse the transformation

        Returns:
            np.ndarray: Transformed values
        """
        pass

    def adstock_geometric(self, channel: str, theta: float) -> Dict[str, Union[np.ndarray, float]]:
        """
        Geometric Adstocking

        Args:
            channel (str): Name of the media channel
            theta (float): Decay rate

        Returns:
            Dict[str, Union[np.ndarray, float]]: Dictionary containing transformed values and metadata
        """
        pass

    def adstock_weibull(
        self, channel: str, shape: float, scale: float, window_length: int = None, adstock_type: str = "cdf"
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Weibull Adstocking

        Args:
            channel (str): Name of the media channel
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
        channel: str,
        adstock: AdstockType,
        theta: float = None,
        shape: float = None,
        scale: float = None,
        window_length: int = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Transform using specified adstock method

        Args:
            channel (str): Name of the media channel
            adstock (AdstockType): Type of adstock transformation
            theta (float): Theta parameter for geometric adstock
            shape (float): Shape parameter for Weibull adstock
            scale (float): Scale parameter for Weibull adstock
            window_length (int): Window length for Weibull adstock

        Returns:
            Dict[str, Union[np.ndarray, float]]: Dictionary containing transformed values and metadata
        """
        pass

    def saturation_hill(
        self, channel: str, alpha: float, gamma: float, marginal_input: np.ndarray = None
    ) -> np.ndarray:
        """
        Hill Saturation Transformation

        Args:
            channel (str): Name of the media channel
            alpha (float): Alpha parameter
            gamma (float): Gamma parameter
            marginal_input (np.ndarray): Marginal input values

        Returns:
            np.ndarray: Transformed values
        """
        pass

    @staticmethod
    def weibull_cdf(time_series_data: np.ndarray, shape: float, scale: float) -> np.ndarray:
        """
        Weibull Cumulative Distribution Function

        Args:
            time_series_data (np.ndarray): Time series data of a media channel
            shape (float): Shape parameter
            scale (float): Scale parameter

        Returns:
            np.ndarray: CDF values
        """
        pass

    @staticmethod
    def weibull_pdf(time_series_data: np.ndarray, shape: float, scale: float) -> np.ndarray:
        """
        Weibull Probability Density Function

        Args:
            time_series_data (np.ndarray): Time series data of a media channel
            shape (float): Shape parameter
            scale (float): Scale parameter

        Returns:
            np.ndarray: PDF values
        """
        pass

    @staticmethod
    def normalize(time_series_data: np.ndarray) -> np.ndarray:
        """
        Normalize the time series data

        Args:
            time_series_data (np.ndarray): Time series data of a media channel

        Returns:
            np.ndarray: Normalized time series data
        """
        pass