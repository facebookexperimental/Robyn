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

    def plot_adstock(self, channel: str, display: bool = True) -> Dict[str, plt.Figure]:
        """
        Plot the adstock transformation for a specific channel.

        Args:
            channel (str): The name of the channel to plot adstock for.

        Returns:
            plt.Figure: A matplotlib Figure object containing the adstock plot.
        """
        logger.info("Generating adstock plot for channel: %s", channel)
        logger.debug("Processing adstock transformation for channel %s", channel)
        try:
            # Implementation placeholder
            logger.warning("plot_adstock method not implemented yet")
        except Exception as e:
            logger.error(
                "Failed to generate adstock plot for channel %s: %s", channel, str(e)
            )
            raise

    def plot_saturation(
        self, channel: str, display: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Plot the saturation curve transformation for a specific channel.

        Args:
            channel (str): The name of the channel to plot saturation for.

        Returns:
            plt.Figure: A matplotlib Figure object containing the saturation curves plot.
        """
        logger.info("Generating saturation plot for channel: %s", channel)
        logger.debug(
            "Processing saturation curve transformation for channel %s", channel
        )
        try:
            # Implementation placeholder
            logger.warning("plot_saturation method not implemented yet")

        except Exception as e:
            logger.error(
                "Failed to generate saturation plot for channel %s: %s", channel, str(e)
            )
            raise

    def plot_spend_exposure(
        self, channel: str, display: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generates a spend-exposure plot for a specified channel.
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
            if res is None:
                raise ValueError(f"No spend-exposure data available for channel: {channel}")
            
            plot_data = self.featurized_mmmdata.modNLS["plots"].get(channel)
            if plot_data is None:
                raise ValueError(f"No plot data available for channel: {channel}")

            # Create figure using base visualizer methods
            fig, ax = self.create_figure(figsize=self.figure_sizes["medium"])

            # Plot scatter of actual data
            sns.scatterplot(
                x="spend",
                y="exposure",
                data=plot_data,
                ax=ax,
                alpha=self.alpha["primary"],
                label="Actual",
                color=self.colors["primary"]
            )

            # Plot fitted line
            sns.lineplot(
                x="spend",
                y="yhat",
                data=plot_data,
                ax=ax,
                color=self.colors["secondary"],
                label="Fitted"
            )

            # Set labels and title using base visualizer methods
            self._set_standardized_labels(
                ax,
                xlabel=f"Spend [{channel}]",
                ylabel=f"Exposure [{channel}]",
                title=f"Spend vs Exposure for {channel}"
            )

            # Add model information
            model_type = res["model_type"]
            rsq = res["rsq"]
            
            if model_type == "nls":
                Vmax, Km = res["Vmax"], res["Km"]
                text = f"Model: Michaelis-Menten\nR² = {rsq:.4f}\nVmax = {Vmax:.2f}\nKm = {Km:.2f}"
            else:
                coef = res["coef_lm"]
                text = f"Model: Linear\nR² = {rsq:.4f}\nCoefficient = {coef:.4f}"

            # Add text box with model information
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=self.alpha["annotation"],
                    edgecolor=self.colors["grid"]
                ),
                fontsize=self.fonts["sizes"]["annotation"]
            )

            # Add grid and style using base visualizer methods
            self._add_standardized_grid(ax)
            self._set_standardized_spines(ax)
            self._add_standardized_legend(ax, loc='lower right')
            
            # Finalize the figure
            self.finalize_figure(tight_layout=True)
            
            logger.info("Successfully generated spend-exposure plot for channel %s", channel)

            self.cleanup()
            return {"spend-exposure": fig}
        except Exception as e:
            logger.error(
                "Failed to generate spend-exposure plot for channel %s: %s",
                channel,
                str(e),
                exc_info=True,
            )
            raise

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate all plots available in the feature plotter.
        """
        logger.info("Generating all plots")
        plot_collect: Dict[str, plt.Figure] = {}
        
        try:
            # Create plots for each channel only once
            channels = {item["channel"] for item in self.featurized_mmmdata.modNLS["results"]}
            
            for channel in channels:
                spend_exposure_plot = self.plot_spend_exposure(channel, display=False)
                plot_collect[f"{channel}_spend_exposure"] = spend_exposure_plot["spend-exposure"]

            if display_plots:
                self.display_plots(plot_collect)

            if export_location:
                self.export_plots_fig(export_location, plot_collect)

            return plot_collect
        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise

    def __del__(self):
        """Cleanup when the plotter is destroyed."""
        self.cleanup()