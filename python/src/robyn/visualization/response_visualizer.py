from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import logging
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class ResponseVisualizer(BaseVisualizer):
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        super().__init__()
        logger.debug(
            "Initializing ResponseVisualizer with pareto_result=%s, mmm_data=%s",
            pareto_result,
            mmm_data,
        )
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data

    def plot_response(self) -> plt.Figure:
        """
        Plot response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        logger.info("Starting response curve plotting")
        pass

    def plot_marginal_response(self) -> plt.Figure:
        """
        Plot marginal response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        logger.info("Starting marginal response curve plotting")
        pass

    def generate_response_curves(
        self, solution_id: str, ax: Optional[plt.Axes] = None, trim_rate: float = 1.3
    ) -> Optional[plt.Figure]:
        """Generate response curves showing relationship between spend and response."""
        logger.debug("Generating response curves with trim_rate=%.2f", trim_rate)

        if solution_id not in self.pareto_result.plot_data_collect:
            raise ValueError(f"Invalid solution ID: {solution_id}")

        try:
            # Add debug logging to inspect plot4data structure
            plot_data = self.pareto_result.plot_data_collect[solution_id]

            curve_data = plot_data["plot4data"]["dt_scurvePlot"].copy()
            mean_data = plot_data["plot4data"]["dt_scurvePlotMean"].copy()

            # Add filtering for paid media channels and trim rate
            if trim_rate > 0:
                max_mean_spend = mean_data["mean_spend_adstocked"].max()
                max_mean_response = mean_data["mean_response"].max()
                curve_data = curve_data[
                    (curve_data["spend"] < max_mean_spend * trim_rate)
                    & (curve_data["response"] < max_mean_response * trim_rate)
                    & (
                        curve_data["channel"].isin(
                            self.mmm_data.mmmdata_spec.paid_media_spends
                        )
                    )
                ]
                # Add mean carryover information early
                curve_data = curve_data.merge(
                    mean_data[["channel", "mean_carryover"]], on="channel", how="left"
                )

            # Filter mean data to match curve data channels
            mean_data = mean_data[
                mean_data["channel"].isin(curve_data["channel"].unique())
            ]

            # Scale down the values to thousands
            curve_data["spend"] = curve_data["spend"] / 1000
            curve_data["response"] = curve_data["response"] / 1000
            mean_data["mean_spend_adstocked"] = mean_data["mean_spend_adstocked"] / 1000
            mean_data["mean_response"] = mean_data["mean_response"] / 1000

            # Add debug logging after scaling
            logger.debug("Scaled curve data head:\n%s", curve_data.head())
            logger.debug("Scaled mean data:\n%s", mean_data)

            # For each channel, verify the mean point exists on the curve
            for channel in curve_data["channel"].unique():
                channel_curve = curve_data[curve_data["channel"] == channel]
                channel_mean = mean_data[mean_data["channel"] == channel]

                if not channel_mean.empty:
                    mean_spend = channel_mean["mean_spend_adstocked"].iloc[0]
                    mean_response = channel_mean["mean_response"].iloc[0]

                    # Find closest point on curve
                    closest_point = channel_curve.iloc[
                        (channel_curve["spend"] - mean_spend).abs().argsort()[:1]
                    ]

                    logger.debug(
                        f"Channel {channel} - Mean point: ({mean_spend:.2f}, {mean_response:.2f}), "
                        f"Closest curve point: ({closest_point['spend'].iloc[0]:.2f}, "
                        f"{closest_point['response'].iloc[0]:.2f})"
                    )

            # Add mean carryover information
            curve_data = curve_data.merge(
                mean_data[["channel", "mean_carryover"]], on="channel", how="left"
            )

            if ax is None:
                logger.debug("Creating new figure with axes")
                fig, ax = plt.subplots(figsize=(16, 10))
            else:
                logger.debug("Using provided axes")
                fig = None

            # Define custom colors matching the R plot
            color_map = {
                "facebook_S": "#FF9D1C",  # Orange
                "ooh_S": "#69B3E7",  # Light blue
                "print_S": "#7B4EA3",  # Purple
                "search_S": "#E41A1C",  # Red
                "tv_S": "#4DAF4A",  # Green
            }

            channels = curve_data["channel"].unique()
            logger.debug("Processing %d unique channels: %s", len(channels), channels)

            for channel in channels:
                logger.debug("Plotting response curve for channel: %s", channel)
                channel_data = curve_data[curve_data["channel"] == channel].sort_values(
                    "spend"
                )

                color = color_map.get(
                    channel, "gray"
                )  # Default to gray if channel not in map

                ax.plot(
                    channel_data["spend"],
                    channel_data["response"],
                    color=color,
                    label=channel,
                    zorder=2,
                )

                if "mean_carryover" in channel_data.columns:
                    logger.debug("Adding carryover shading for channel: %s", channel)
                    carryover_data = channel_data[
                        channel_data["spend"] <= channel_data["mean_carryover"].iloc[0]
                    ]
                    ax.fill_between(
                        carryover_data["spend"],
                        np.zeros_like(carryover_data["spend"]),
                        carryover_data["response"],
                        color="grey",
                        alpha=0.4,
                        zorder=1,
                    )

            logger.debug("Adding mean points and labels")
            for idx, row in mean_data.iterrows():
                channel = row["channel"]
                color = color_map.get(channel, "gray")

                ax.scatter(
                    row["mean_spend_adstocked"],
                    row["mean_response"],
                    color=color,
                    s=100,
                    zorder=3,
                )

                # Format point labels
                formatted_spend = f"{row['mean_spend_adstocked']:.1f}K"

                ax.text(
                    row["mean_spend_adstocked"],
                    row["mean_response"],
                    formatted_spend,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color=color,
                )

            logger.debug("Formatting axis labels")

            # Custom locator for x and y axes
            def custom_tick_formatter(x, p):
                if x == 0:
                    return "0"
                return f"{int(x)}K"

            # Set axis limits matching the R plot
            ax.set_xlim(0, 120)
            ax.set_ylim(0, 150)

            # Set major ticks
            ax.set_xticks([0, 30, 60, 90, 120])
            ax.set_yticks([0, 25, 50, 75, 100, 125, 150])

            ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_tick_formatter))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_tick_formatter))

            # Grid styling to match R plot
            ax.grid(True, alpha=0.2, linestyle="-", color="gray")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_title("Response Curves and Mean Spends by Channel")
            ax.set_xlabel("Spend (carryover + immediate)")
            ax.set_ylabel("Response")

            # Adjust legend to match R plot
            ax.legend(
                loc="upper left",  # or any other position: 'lower right', 'center left', etc.
                frameon=True,
                framealpha=0.8,
                facecolor="white",
                edgecolor="none",
            )

            if fig:
                logger.debug("Adjusting layout")
                plt.tight_layout()
                logger.debug("Successfully generated response curves figure")
                return fig

            logger.debug("Successfully added response curves to existing axes")
            return None

        except Exception as e:
            logger.error("Error generating response curves: %s", str(e), exc_info=True)
            raise

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> None:
        pass
