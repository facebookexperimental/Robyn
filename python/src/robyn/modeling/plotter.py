from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.enums import AdstockType, DependentVarType


class FeaturePlotter:
    """
    A class for creating various plots related to feature engineering in the Robyn framework.
    """

    def __init__(self, mmm_data: MMMData, hyperparameters: Hyperparameters):
        """
        Initialize the FeaturePlotter class.

        Args:
            mmm_data (MMMData): The input data and specifications for the MMM.
            hyperparameters (Hyperparameters): The hyperparameters for the model.
        """
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters

    def plot_adstock(self, channel: str) -> plt.Figure:
        """
        Plot the adstock transformation for a specific channel.

        Args:
            channel (str): The name of the channel to plot adstock for.

        Returns:
            plt.Figure: A matplotlib Figure object containing the adstock plot.
        """
        channel_params: ChannelHyperparameters = self.hyperparameters.get_hyperparameter(channel)
        adstock_type: AdstockType = self.hyperparameters.adstock

        x = np.arange(0, 10, 0.1)
        if adstock_type == AdstockType.GEOMETRIC:
            theta = channel_params.thetas[0] if channel_params.thetas else 0.5
            y = theta**x
        elif adstock_type in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            shape = channel_params.shapes[0] if channel_params.shapes else 1
            scale = channel_params.scales[0] if channel_params.scales else 1
            if adstock_type == AdstockType.WEIBULL_CDF:
                y = 1 - np.exp(-((x / scale) ** shape))
            else:  # WEIBULL_PDF
                y = (shape / scale) * (x / scale) ** (shape - 1) * np.exp(-((x / scale) ** shape))

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"Adstock Transformation for {channel}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Adstock Effect")

        return fig

    def plot_saturation(self, channel: str) -> plt.Figure:
        """
        Plot the saturation curve transformation for a specific channel.

        Args:
            channel (str): The name of the channel to plot saturation for.

        Returns:
            plt.Figure: A matplotlib Figure object containing the saturation curves plot.
        """
        channel_params: ChannelHyperparameters = self.hyperparameters.get_hyperparameter(channel)
        alpha = channel_params.alphas[0] if channel_params.alphas else 1
        gamma = channel_params.gammas[0] if channel_params.gammas else 1

        x = np.linspace(0, 1, 100)
        y = x**alpha / (x**alpha + gamma**alpha)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"Saturation Curve for {channel}")
        ax.set_xlabel("Normalized Media Spend")
        ax.set_ylabel("Saturation Effect")

        return fig

    def plot_spend_exposure(self, dt_mod: pd.DataFrame, channel: str) -> plt.Figure:
        """
        Plot the relationship between spend and exposure for a given channel.

        Args:
            dt_mod (pd.DataFrame): The modified data after feature engineering.
            channel (str): The name of the channel being plotted.

        Returns:
            plt.Figure: A matplotlib Figure object containing the spend-exposure plot.
        """
        spend = dt_mod[f"{channel}_S"].values
        exposure = dt_mod[channel].values

        fig, ax = plt.subplots()
        ax.scatter(spend, exposure, alpha=0.5)
        ax.set_title(f"Spend-Exposure Relationship for {channel}")
        ax.set_xlabel("Spend")
        ax.set_ylabel("Exposure")

        return fig

    def plot_feature_importance(self, feature_importance: Dict[str, float]) -> plt.Figure:
        """
        Plot the importance of different features in the model.

        Args:
            feature_importance (Dict[str, float]): Dictionary of feature importances.

        Returns:
            plt.Figure: A matplotlib Figure object containing the feature importance plot.
        """
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")

        return fig

    def plot_response_curves(self, dt_mod: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Plot response curves for different channels.

        Args:
            dt_mod (pd.DataFrame): The modified data after feature engineering.

        Returns:
            Dict[str, plt.Figure]: Dictionary mapping channel names to their respective response curve plots.
        """
        response_curves: Dict[str, plt.Figure] = {}

        for channel in self.mmm_data.mmmdata_spec.paid_media_vars:
            channel_data = dt_mod[["ds", channel]]
            channel_params = self.hyperparameters.get_hyperparameter(channel)

            x = np.linspace(channel_data[channel].min(), channel_data[channel].max(), 100)
            y = self._apply_transformations(x, channel_params)

            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title(f"Response Curve for {channel}")
            ax.set_xlabel("Media Spend")
            ax.set_ylabel("Response")

            response_curves[channel] = fig

        return response_curves

    def save_plots(self, plots: Dict[str, plt.Figure], output_dir: str) -> None:
        """
        Save all generated plots to the specified output directory.

        Args:
            plots (Dict[str, plt.Figure]): Dictionary mapping plot names to matplotlib Figure objects.
            output_dir (str): Directory path where the plots should be saved.

        Returns:
            None
        """
        for name, fig in plots.items():
            fig.savefig(f"{output_dir}/{name}.png")
        plt.close("all")

    def _apply_transformations(self, x: np.ndarray, params: ChannelHyperparameters) -> np.ndarray:
        """
        Apply adstock and saturation transformations to the input array.

        Args:
            x (np.ndarray): Input array to transform.
            params (ChannelHyperparameters): Hyperparameters for the channel.

        Returns:
            np.ndarray: Transformed array.
        """
        # Apply adstock
        if self.hyperparameters.adstock == AdstockType.GEOMETRIC:
            theta = params.thetas[0] if params.thetas else 0.5
            x_adstock = x * theta
        else:  # Weibull
            shape = params.shapes[0] if params.shapes else 1
            scale = params.scales[0] if params.scales else 1
            x_adstock = 1 - np.exp(-((x / scale) ** shape))

        # Apply saturation
        alpha = params.alphas[0] if params.alphas else 1
        gamma = params.gammas[0] if params.gammas else 1
        y = x_adstock**alpha / (x_adstock**alpha + gamma**alpha)

        return y
