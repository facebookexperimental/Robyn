# pyre-strict
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import seaborn as sns
import matplotlib.pyplot as plt

from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.visualization.base_visualizer import BaseVisualizer


logger = logging.getLogger(__name__)


class FeaturePlotter(BaseVisualizer):
    """
    A class for creating various plots related to feature engineering in the Robyn framework.
    """

    def __init__(
        self,
        mmm_data: MMMData,
        hyperparameters: Hyperparameters,
        featurized_mmmdata: Optional[FeaturizedMMMData] = None,
    ):
        """
        Initialize the FeaturePlotter class.

        Args:
            mmm_data (MMMData): The input data and specifications for the MMM.
            hyperparameters (Hyperparameters): The hyperparameters for the model.
        """
        super().__init__()
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.featurized_mmmdata = featurized_mmmdata
        logger.info("Initializing FeaturePlotter")
        logger.debug("MMM Data: %s", mmm_data)
        logger.debug("Hyperparameters: %s", hyperparameters)

    def plot_spend_exposure(
        self, channel: str, display: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generates a spend-exposure plot for a specified channel.

        Parameters:
        -----------
        channel : str
            The name of the channel for which the spend-exposure plot is to be generated.

        Returns:
        --------
        plt.Figure
            The matplotlib Figure object containing the spend-exposure plot.

        Raises:
        -------
        ValueError
            If no spend-exposure data or plot data is available for the specified channel.
        Exception
            If any other error occurs during the plot generation process.

        Notes:
        ------
        The function retrieves the model results and plot data for the specified channel from the featurized_mmmdata attribute.
        It creates a scatter plot of the actual data and a fitted line plot. The plot includes model information such as
        model type, R-squared value, and model-specific parameters (e.g., Vmax and Km for Michaelis-Menten model or coefficient for linear model).
        """
        logger.info("Generating spend-exposure plot for channel: %s", channel)

        try:
            # Find the result for the current channel
            res = next(
                (
                    item
                    for item in self.featurized_mmmdata.modNLS["results"]
                    if item["channel"] == channel
                ),
                None,
            )
            logger.info("Found result for channel %s", channel)
            if res is None:
                logger.error("Channel %s not found in featurized data results", channel)
                raise ValueError(
                    f"No spend-exposure data available for channel: {channel}"
                )
            plot_data = self.featurized_mmmdata.modNLS["plots"].get(channel)
            if plot_data is None:
                logger.error("Plot data for channel %s not found", channel)
                raise ValueError(f"No plot data available for channel: {channel}")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot scatter of actual data
            sns.scatterplot(
                x="spend",
                y="exposure",
                data=plot_data,
                ax=ax,
                alpha=0.6,
                label="Actual",
            )
            logger.debug("Created scatter plot for actual data")
            # Plot fitted line
            sns.lineplot(
                x="spend", y="yhat", data=plot_data, ax=ax, color="red", label="Fitted"
            )
            logger.debug("Added fitted line to plot")
            ax.set_xlabel(f"Spend [{channel}]")
            ax.set_ylabel(f"Exposure [{channel}]")
            ax.set_title(f"Spend vs Exposure for {channel}")
            # Add model information to the plot
            model_type = res["model_type"]
            rsq = res["rsq"]
            logger.debug("Model type: %s, R-squared: %f", model_type, rsq)
            if model_type == "nls":
                Vmax, Km = res["Vmax"], res["Km"]
                ax.text(
                    0.05,
                    0.95,
                    f"Model: Michaelis-Menten\nR² = {rsq:.4f}\nVmax = {Vmax:.2f}\nKm = {Km:.2f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )
                logger.debug("Added NLS model parameters: Vmax=%f, Km=%f", Vmax, Km)
            else:
                coef = res["coef_lm"]
                ax.text(
                    0.05,
                    0.95,
                    f"Model: Linear\nR² = {rsq:.4f}\nCoefficient = {coef:.4f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )
                logger.debug("Added linear model parameters: coefficient=%f", coef)
            plt.legend()
            plt.tight_layout()
            plt.close()
            logger.info(
                "Successfully generated spend-exposure plot for channel %s", channel
            )
            return {"spend-exposure": fig}
        except Exception as e:
            logger.error(
                "Failed to generate spend-exposure plot for channel %s: %s",
                channel,
                str(e),
                exc_info=True,
            )
            raise

    def plot_feature_importance(
        self, feature_importance: Dict[str, float], display: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Plot the importance of different features in the model.

        Args:
            feature_importance (Dict[str, float]): Dictionary of feature importances.

        Returns:
            plt.Figure: A matplotlib Figure object containing the feature importance plot.
        """
        logger.info("Generating feature importance plot")
        logger.debug("Feature importance data: %s", feature_importance)
        try:
            # Implementation placeholder
            logger.warning("plot_feature_importance method not implemented yet")

        except Exception as e:
            logger.error("Failed to generate feature importance plot: %s", str(e))
            raise

    def plot_response_curves(self, display: bool = True) -> Dict[str, plt.Figure]:
        """
        Plot response curves for different channels.

        Args:
            self.featurized_mmmdata (FeaturizedMMMData): The featurized data after feature engineering.

        Returns:
            Dict[str, plt.Figure]: Dictionary mapping channel names to their respective response curve plots.
        """
        logger.info("Generating response curves")
        logger.debug("Processing featurized data: %s", self.featurized_mmmdata)
        try:
            dt_mod = self.featurized_mmmdata.dt_mod
            logger.debug("Modified data: %s", dt_mod)
            # Rest of the method implementation
            logger.warning("plot_response_curves method not fully implemented yet")

        except Exception as e:
            logger.error("Failed to generate response curves: %s", str(e))
            raise

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> Dict[str, plt.Figure]:
        """
        Override the abstract method plot_all from BaseVisualizer.
        """
        logger.info("Generating all plots")
        plot_collect: Dict[str, plt.Figure] = {}
        try:
            for item in self.featurized_mmmdata.modNLS["results"]:
                channel = item["channel"]
                # plot_collect.update(self.plot_adstock(channel, display))
                # plot_collect.update(self.plot_saturation(channel, display))
                plot_collect[channel] = self.plot_spend_exposure(
                    channel, display_plots
                )["spend-exposure"]

            # plot_collect.update(self.plot_feature_importance({}, display))

            super().display_plots(plot_collect)
        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise
