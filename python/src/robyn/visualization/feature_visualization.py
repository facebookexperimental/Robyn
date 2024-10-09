from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.modeling.feature_engineering import FeaturizedMMMData  # New import


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
        pass

    def plot_saturation(self, channel: str) -> plt.Figure:
        """
        Plot the saturation curve transformation for a specific channel.

        Args:
            channel (str): The name of the channel to plot saturation for.

        Returns:
            plt.Figure: A matplotlib Figure object containing the saturation curves plot.
        """
        pass

    def plot_spend_exposure(self, featurized_data: FeaturizedMMMData, channel: str) -> plt.Figure:
        """
        Plot the relationship between spend and exposure for a given channel.

        Args:
            featurized_data (FeaturizedMMMData): The featurized data after feature engineering.
            channel (str): The name of the channel being plotted.

        Returns:
            plt.Figure: A matplotlib Figure object containing the spend-exposure plot.
        """
        if channel not in featurized_data.modNLS["results"]:
            raise ValueError(f"No spend-exposure data available for channel: {channel}")

        res = featurized_data.modNLS["results"][channel]
        plot_data = featurized_data.modNLS["plots"][channel]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot scatter of actual data
        sns.scatterplot(x="spend", y="exposure", data=plot_data, ax=ax, alpha=0.6, label="Actual")

        # Plot fitted line
        sns.lineplot(x="spend", y="yhat", data=plot_data, ax=ax, color="red", label="Fitted")

        ax.set_xlabel(f"Spend [{channel}]")
        ax.set_ylabel(f"Exposure [{channel}]")
        ax.set_title(f"Spend vs Exposure for {channel}")

        # Add model information to the plot
        model_type = res["model_type"]
        rsq = res["rsq"]
        if model_type == "nls":
            Vmax, Km = res["coef"]["Vmax"], res["coef"]["Km"]
            ax.text(
                0.05,
                0.95,
                f"Model: Michaelis-Menten\nR² = {rsq:.4f}\nVmax = {Vmax:.2f}\nKm = {Km:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
        else:
            coef = res["coef"]["coef"]
            ax.text(
                0.05,
                0.95,
                f"Model: Linear\nR² = {rsq:.4f}\nCoefficient = {coef:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        plt.legend()
        plt.tight_layout()

        return fig

    def plot_feature_importance(self, feature_importance: Dict[str, float]) -> plt.Figure:
        """
        Plot the importance of different features in the model.

        Args:
            feature_importance (Dict[str, float]): Dictionary of feature importances.

        Returns:
            plt.Figure: A matplotlib Figure object containing the feature importance plot.
        """
        pass

    def plot_response_curves(self, featurized_data: FeaturizedMMMData) -> Dict[str, plt.Figure]:
        """
        Plot response curves for different channels.

        Args:
            featurized_data (FeaturizedMMMData): The featurized data after feature engineering.

        Returns:
            Dict[str, plt.Figure]: Dictionary mapping channel names to their respective response curve plots.
        """
        dt_mod = featurized_data.dt_mod
        # Rest of the method implementation
        pass
