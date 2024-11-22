# pyre-strict
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robyn.data.entities.enums import DependentVarType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class TransformationVisualizer(BaseVisualizer):
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        logger.debug(
            "Initializing TransformationVisualizer with pareto_result=%s, mmm_data=%s",
            pareto_result,
            mmm_data,
        )
        super().__init__()
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data

    def create_adstock_plots(self) -> None:
        """
        Generate adstock visualization plots and store them as instance variables.
        """
        logger.info("Starting creation of adstock plots")
        try:
            # Implementation placeholder
            logger.debug("Adstock plots creation completed successfully")
        except Exception as e:
            logger.error("Failed to create adstock plots: %s", str(e))
            raise

    def create_saturation_plots(self) -> None:
        """
        Generate saturation visualization plots and store them as instance variables.
        """
        logger.info("Starting creation of saturation plots")
        try:
            # Implementation placeholder
            logger.debug("Saturation plots creation completed successfully")
        except Exception as e:
            logger.error("Failed to create saturation plots: %s", str(e))
            raise

    def get_adstock_plots(self) -> Optional[Tuple[plt.Figure, plt.Figure]]:
        """
        Retrieve the adstock plots.

        Returns:
            Optional[Tuple[plt.Figure, plt.Figure]]: Tuple of matplotlib figures for adstock plots
        """
        logger.debug("Retrieving adstock plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully retrieved adstock plots")
            return None
        except Exception as e:
            logger.error("Failed to retrieve adstock plots: %s", str(e))
            raise

    def get_saturation_plots(self) -> Optional[Tuple[plt.Figure, plt.Figure]]:
        """
        Retrieve the saturation plots.

        Returns:
            Optional[Tuple[plt.Figure, plt.Figure]]: Tuple of matplotlib figures for saturation plots
        """
        logger.debug("Retrieving saturation plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully retrieved saturation plots")
            return None
        except Exception as e:
            logger.error("Failed to retrieve saturation plots: %s", str(e))
            raise

    def display_adstock_plots(self) -> None:
        """
        Display the adstock plots.
        """
        logger.info("Displaying adstock plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully displayed adstock plots")
        except Exception as e:
            logger.error("Failed to display adstock plots: %s", str(e))
            raise

    def display_saturation_plots(self) -> None:
        """
        Display the saturation plots.
        """
        logger.info("Displaying saturation plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully displayed saturation plots")
        except Exception as e:
            logger.error("Failed to display saturation plots: %s", str(e))
            raise

    def save_adstock_plots(self, filenames: List[str]) -> None:
        """
        Save the adstock plots to files.

        Args:
            filenames (List[str]): List of filenames to save the plots
        """
        logger.info("Saving adstock plots to files: %s", filenames)
        try:
            # Implementation placeholder
            logger.debug("Successfully saved adstock plots")
        except Exception as e:
            logger.error("Failed to save adstock plots: %s", str(e))
            raise

    def save_saturation_plots(self, filenames: List[str]) -> None:
        """
        Save the saturation plots to files.

        Args:
            filenames (List[str]): List of filenames to save the plots
        """
        logger.info("Saving saturation plots to files: %s", filenames)
        try:
            # Implementation placeholder
            logger.debug("Successfully saved saturation plots")
        except Exception as e:
            logger.error("Failed to save saturation plots: %s", str(e))
            raise

    def generate_spend_effect_comparison(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate comparison plot of spend share vs effect share."""

        logger.debug("Starting generation of spend effect comparison plot")
        try:
            # Get plot data safely
            logger.debug("Extracting plot data from pareto result")
            plot_data = self.pareto_result.plot_data_collect[solution_id]

            # Safely get bar and line data
            try:
                bar_data = plot_data["plot1data"]["plotMediaShareLoopBar"].copy()
                line_data = plot_data["plot1data"]["plotMediaShareLoopLine"].copy()
                y_sec_scale = plot_data["plot1data"]["ySecScale"]

                logger.debug(
                    "Processing plot data - bar_data shape: %s, line_data shape: %s",
                    bar_data.shape,
                    line_data.shape,
                )

                # Convert y_sec_scale to float safely
                if isinstance(y_sec_scale, pd.DataFrame):
                    y_sec_scale = float(
                        y_sec_scale.iat[0, 0]
                        if len(y_sec_scale.columns) > 0
                        else y_sec_scale.iloc[0]
                    )
                elif isinstance(y_sec_scale, pd.Series):
                    y_sec_scale = float(y_sec_scale.iloc[0])
                else:
                    y_sec_scale = float(y_sec_scale)

                logger.debug("Y-scale factor: %f", y_sec_scale)
            except (KeyError, AttributeError, IndexError) as e:
                logger.error(
                    "Error accessing plot data for solution %s: %s", solution_id, str(e)
                )
                return None

            # Transform variable names safely
            bar_data["variable"] = (
                bar_data["variable"].str.replace("_", " ").str.title()
            )

            # Create figure if no axes provided
            if ax is None:
                logger.debug("Creating new figure and axes")
                fig, ax = plt.subplots(figsize=(16, 10))
                plt.subplots_adjust(top=0.80, left=0.15, bottom=0.1, right=0.95)
            else:
                logger.debug("Using provided axes for plotting")
                fig = None

            # Set background color
            ax.set_facecolor("white")

            # Set up colors
            type_colour = "#03396C"  # Dark blue for line
            bar_colors = ["#A4C2F4", "#FFB7B2"]  # Light blue and light coral for bars
            bar_colors = bar_colors[::-1]  # Reverse colors

            # Set up dimensions
            channels = sorted(
                line_data["rn"].unique()
            )  # Use line_data for consistent ordering
            y_pos = np.arange(len(channels))

            logger.debug("Processing %d channels for visualization", len(channels))

            # Plot bars for each variable type
            bar_width = 0.35
            for i, (var, color) in enumerate(
                zip(reversed(bar_data["variable"].unique()), bar_colors)
            ):
                var_data = bar_data[bar_data["variable"] == var]
                # Ensure alignment with channels - safely get values
                values = []
                for ch in channels:
                    ch_data = var_data[var_data["rn"] == ch]
                    if not ch_data.empty:
                        values.append(ch_data["value"].iloc[0])
                    else:
                        values.append(0)

                logger.debug(
                    "Plotting bars for variable '%s' with %d values", var, len(values)
                )
                bars = ax.barh(
                    y=[y + (i - 0.5) * bar_width for y in y_pos],
                    width=values,
                    height=bar_width,
                    label=var,
                    color=color,
                    alpha=0.5,
                )

                # Add percentage labels to right of y-axis
                for idx, value in enumerate(values):
                    y_position = y_pos[idx] + (i - 0.5) * bar_width
                    percentage = f"{value * 100:.1f}%"

                    ax.text(
                        s=percentage,
                        x=0.02,
                        y=y_position,
                        ha="left",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                        transform=ax.get_yaxis_transform(),
                    )

            # Safely get line values
            line_values = []
            for ch in channels:
                ch_data = line_data[line_data["rn"] == ch]
                if not ch_data.empty:
                    line_values.append(ch_data["value"].iloc[0])
                else:
                    line_values.append(0)

            line_values = np.array(line_values)
            line_x = line_values / y_sec_scale

            logger.debug("Plotting line with %d points", len(line_x))
            # Plot line without label
            ax.plot(
                line_x, y_pos, color=type_colour, marker="o", markersize=8, zorder=3
            )

            # Add line value labels
            for i, value in enumerate(line_values):
                ax.text(
                    line_x[i] + 0.02,  # Added offset to move labels right of dots
                    y_pos[i],
                    f"{value:.2f}",
                    color=type_colour,
                    fontweight="bold",
                    ha="left",  # Left align since we're positioning to the right of dots
                    va="center",
                    zorder=4,
                )

            # Set channel labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(channels)
            ax.tick_params(axis="y", pad=5)

            # Format x-axis as percentage and show labels up to 70%
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%")
            )
            ax.set_xlim(0, 0.7)  # Set x-axis limit to 70%
            xticks = np.arange(
                0, 0.8, 0.1
            )  # Create ticks from 0 to 70% in 10% increments
            ax.set_xticks(xticks)
            ax.tick_params(axis="x", labelbottom=True)

            # Add grid
            ax.grid(True, axis="x", alpha=0.2, linestyle="-")
            ax.set_axisbelow(True)

            # Remove unnecessary spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set title at top position
            metric_type = (
                "ROAS"
                if (
                    self.mmm_data
                    and hasattr(self.mmm_data.mmmdata_spec, "dep_var_type")
                    and self.mmm_data.mmmdata_spec.dep_var_type
                    == DependentVarType.REVENUE
                )
                else "CPA"
            )

            logger.debug("Setting plot title with metric type: %s", metric_type)

            # Set title at top
            ax.set_title(
                f"Share of Total Spend, Effect & {metric_type} in Modeling Window*",
                pad=20,
                y=1.45,
            )

            # Create legend with single ROAS entry
            bars_legend = ax.get_legend_handles_labels()
            line_legend = [
                plt.Line2D(
                    [0],
                    [0],
                    color=type_colour,
                    marker="o",
                    linestyle="-",
                    markersize=8,
                    label="ROAS",
                )
            ]

            # Combine legend elements in reverse order
            handles = line_legend + list(reversed(bars_legend[0]))
            labels = ["ROAS"] + list(reversed(bars_legend[1]))

            # Create legend below title
            ax.legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(0, 1.05, 0.4, 0.1),
                loc="lower left",
                mode="expand",
                ncol=3,
                frameon=False,
                borderaxespad=0,
            )

            # Add axis labels
            ax.set_xlabel("Total Share by Channel")
            ax.set_ylabel(None)

            logger.debug("Successfully generated spend effect comparison plot")
            if fig:
                plt.tight_layout()
                plt.subplots_adjust(top=0.80, left=0.15, bottom=0.1, right=0.95)
                return fig
            return None

        except (KeyError, AttributeError, IndexError, ValueError) as error:
            logger.error(
                "Error generating spend effect plot for solution %s: %s",
                solution_id,
                str(error),
            )
            if ax:
                ax.text(
                    0.5,
                    0.5,
                    "Error generating spend effect plot",
                    ha="center",
                    va="center",
                )
            return None

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> None:
        pass
