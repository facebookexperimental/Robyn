from robyn.visualization.base_visualizer import BaseVisualizer
from robyn.allocator.entities.allocation_results import AllocationResult
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class AllocationPlotter(BaseVisualizer):
    """Plotter class for allocation results visualization."""

    def __init__(self, result: AllocationResult):
        """
        Initialize plotter with allocation results.

        Args:
            result: Allocation results to plot
        """
        logger.debug("Initializing AllocationPlotter")
        super().__init__(style="bmh")
        self.result = result
        if self.result is None:
            logger.error("AllocationResult cannot be None")
            raise ValueError("AllocationResult cannot be None")
        logger.info("AllocationPlotter initialized successfully with result: %s", self.result)

    def plot_all(self) -> Dict[str, plt.Figure]:
        """
        Generate all allocation plots.

        Returns:
            Dictionary of figures keyed by plot name
        """
        logger.info("Starting to generate all allocation plots")
        figures = {}
        try:
            logger.debug("Generating spend allocation plot")
            figures["spend_allocation"] = self.plot_spend_allocation()
            
            logger.debug("Generating response curves plot")
            figures["response_curves"] = self.plot_response_curves()
            
            logger.debug("Generating efficiency frontier plot")
            figures["efficiency_frontier"] = self.plot_efficiency_frontier()
            
            logger.debug("Generating spend vs response plot")
            figures["spend_vs_response"] = self.plot_spend_vs_response()
            
            logger.debug("Generating summary metrics plot")
            figures["summary_metrics"] = self.plot_summary_metrics()
            
            logger.info("Successfully generated all %d plots", len(figures))
        except Exception as e:
            logger.error("Failed to generate plots: %s", str(e))
            raise
        finally:
            logger.debug("Cleaning up plot resources")
            self.cleanup()
        return figures

    def plot_spend_allocation(self) -> plt.Figure:
        """Plot spend allocation comparison."""
        logger.debug("Starting spend allocation plot generation")
        
        # Create figure
        fig, ax = self.create_figure()
        optimal_allocations = self.result.optimal_allocations
        logger.debug("Processing optimal allocations data: %s", optimal_allocations)

        # Prepare data
        channels = optimal_allocations["channel"].values
        x = np.arange(len(channels))
        width = 0.35

        logger.debug("Plotting current spend bars for %d channels", len(channels))
        # Plot bars
        ax.bar(
            x - width / 2,
            optimal_allocations["current_spend"].values,
            width,
            label="Current",
            color=self.colors["current"],
            edgecolor="gray",
            alpha=self.alpha["primary"],
        )

        logger.debug("Plotting optimal spend bars")
        ax.bar(
            x + width / 2,
            optimal_allocations["optimal_spend"].values,
            width,
            label="Optimized",
            color=self.colors["optimal"],
            edgecolor="gray",
            alpha=self.alpha["primary"],
        )

        # Add annotations
        logger.debug("Adding percentage change annotations")
        for i, (curr, opt) in enumerate(
            zip(optimal_allocations["current_spend"].values, optimal_allocations["optimal_spend"].values)
        ):
            pct_change = ((opt / curr) - 1) * 100
            self.add_percentage_annotation(ax, x[i], max(curr, opt), pct_change)

        # Setup axis
        self.setup_axis(
            ax, title="Media Spend Allocation", ylabel="Spend", xticks=x, xticklabels=channels, rotation=45
        )

        self.add_legend(ax)
        self.finalize_figure()
        
        logger.info("Spend allocation plot generated successfully")
        return fig

    def plot_response_curves(self) -> plt.Figure:
        """Plot response curves for each channel."""
        logger.debug("Starting response curves plot generation")
        
        # Prepare data
        curves_df = self.result.response_curves
        channels = curves_df["channel"].unique()
        n_channels = len(channels)
        ncols = min(3, n_channels)
        nrows = (n_channels + ncols - 1) // ncols
        
        logger.debug("Processing %d channels for response curves", n_channels)

        # Create figure
        fig, axes = self.create_figure(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

        # Handle single subplot case
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each channel
        for idx, channel in enumerate(channels):
            logger.debug("Plotting response curve for channel: %s", channel)
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            channel_data = curves_df[curves_df["channel"] == channel]

            # Plot response curve
            ax.plot(
                channel_data["spend"],
                channel_data["response"],
                color=self.colors["optimal"],
                alpha=self.alpha["primary"],
            )

            # Plot current point
            current_data = channel_data[channel_data["is_current"]]
            if not current_data.empty:
                logger.debug("Plotting current point for channel %s", channel)
                ax.scatter(
                    current_data["spend"].iloc[0],
                    current_data["response"].iloc[0],
                    color=self.colors["negative"],
                    label="Current",
                    s=100,
                )

            # Plot optimal point
            optimal_data = channel_data[channel_data["is_optimal"]]
            if not optimal_data.empty:
                logger.debug("Plotting optimal point for channel %s", channel)
                ax.scatter(
                    optimal_data["spend"].iloc[0],
                    optimal_data["response"].iloc[0],
                    color=self.colors["positive"],
                    label="Optimal",
                    s=100,
                )

            self.setup_axis(ax, title=f"{channel} Response Curve")
            self.add_legend(ax)

        # Remove empty subplots
        for idx in range(n_channels, nrows * ncols):
            logger.debug("Removing empty subplot at index %d", idx)
            fig.delaxes(axes[idx // ncols, idx % ncols])

        self.finalize_figure()
        logger.info("Response curves plot generated successfully")
        return fig

    def plot_efficiency_frontier(self) -> plt.Figure:
        """Plot efficiency frontier."""
        logger.debug("Starting efficiency frontier plot generation")

        # Create figure
        fig, ax = self.create_figure()

        # Calculate totals
        optimal_allocations = self.result.optimal_allocations
        current_total_spend = optimal_allocations["current_spend"].sum()
        current_total_response = optimal_allocations["current_response"].sum()
        optimal_total_spend = optimal_allocations["optimal_spend"].sum()
        optimal_total_response = optimal_allocations["optimal_response"].sum()

        logger.debug("Calculated totals - Current spend: %f, Current response: %f, Optimal spend: %f, Optimal response: %f",
                    current_total_spend, current_total_response, optimal_total_spend, optimal_total_response)

        # Plot points and connect them
        ax.scatter(
            current_total_spend,
            current_total_response,
            color=self.colors["negative"],
            s=100,
            label="Current",
            zorder=2,
        )

        ax.scatter(
            optimal_total_spend,
            optimal_total_response,
            color=self.colors["positive"],
            s=100,
            label="Optimal",
            zorder=2,
        )

        ax.plot(
            [current_total_spend, optimal_total_spend],
            [current_total_response, optimal_total_response],
            "--",
            color=self.colors["neutral"],
            alpha=self.alpha["secondary"],
            zorder=1,
        )

        # Calculate and add percentage changes
        pct_spend_change = ((optimal_total_spend / current_total_spend) - 1) * 100
        pct_response_change = ((optimal_total_response / current_total_response) - 1) * 100
        
        logger.debug("Percentage changes - Spend: %f%%, Response: %f%%", pct_spend_change, pct_response_change)

        ax.annotate(
            f"Spend: {pct_spend_change:.1f}%\nResponse: {pct_response_change:.1f}%",
            xy=(optimal_total_spend, optimal_total_response),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=self.font_sizes["annotation"],
            bbox=dict(facecolor="white", edgecolor=self.colors["neutral"], alpha=self.alpha["annotation"]),
        )

        self.setup_axis(ax, title="Efficiency Frontier", xlabel="Total Spend", ylabel="Total Response")
        self.add_legend(ax)
        self.finalize_figure()

        logger.info("Efficiency frontier plot generated successfully")
        return fig

    def plot_spend_vs_response(self) -> plt.Figure:
        """Plot spend vs response changes."""
        logger.debug("Starting spend vs response plot generation")

        # Create figure
        fig, (ax1, ax2) = self.create_figure(nrows=2, ncols=1, figsize=(12, 10))

        # Get data
        df = self.result.optimal_allocations
        channels = df["channel"].values
        x = np.arange(len(channels))

        logger.debug("Processing spend changes for %d channels", len(channels))
        # Plot spend changes
        spend_pct = ((df["optimal_spend"] / df["current_spend"]) - 1) * 100
        colors = [self.colors["positive"] if pct >= 0 else self.colors["negative"] for pct in spend_pct]

        ax1.bar(x, spend_pct, color=colors, alpha=self.alpha["primary"])
        self._plot_change_axis(ax1, x, channels, spend_pct, "Spend Change %")

        logger.debug("Processing response changes")
        # Plot response changes
        response_pct = ((df["optimal_response"] / df["current_response"]) - 1) * 100
        colors = [self.colors["positive"] if pct >= 0 else self.colors["negative"] for pct in response_pct]

        ax2.bar(x, response_pct, color=colors, alpha=self.alpha["primary"])
        self._plot_change_axis(ax2, x, channels, response_pct, "Response Change %")

        self.finalize_figure(adjust_spacing=True)
        logger.info("Spend vs response plot generated successfully")
        return fig

    def _plot_change_axis(
        self, ax: plt.Axes, x: np.ndarray, channels: np.ndarray, pct_values: np.ndarray, ylabel: str
    ) -> None:
        """Helper method to setup change plot axes."""
        logger.debug("Setting up change plot axis for %s", ylabel)
        self.setup_axis(ax, ylabel=ylabel, xticks=x, xticklabels=channels, rotation=45)

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)

        for i, pct in enumerate(pct_values):
            self.add_percentage_annotation(
                ax, i, pct + (2 if pct >= 0 else -5), pct, va="bottom" if pct >= 0 else "top"
            )

    def plot_summary_metrics(self) -> plt.Figure:
        """Plot summary metrics."""
        logger.debug("Starting summary metrics plot generation")

        # Create figure
        fig, ax = self.create_figure()

        # Get data
        optimal_allocations = self.result.optimal_allocations
        channels = optimal_allocations["channel"].values
        dep_var_type = self.result.metrics.get("dep_var_type")
        
        logger.debug("Processing metrics for dependency variable type: %s", dep_var_type)

        # Calculate metrics
        if dep_var_type == "revenue":
            current_metric = optimal_allocations["current_response"] / optimal_allocations["current_spend"]
            optimal_metric = optimal_allocations["optimal_response"] / optimal_allocations["optimal_spend"]
            metric_name = "ROI"
        else:
            current_metric = optimal_allocations["current_spend"] / optimal_allocations["current_response"]
            optimal_metric = optimal_allocations["optimal_spend"] / optimal_allocations["optimal_response"]
            metric_name = "CPA"

        logger.debug("Calculated %s metrics for %d channels", metric_name, len(channels))

        # Plot bars
        x = np.arange(len(channels))
        width = 0.35

        ax.bar(
            x - width / 2,
            current_metric,
            width,
            label=f"Current {metric_name}",
            color=self.colors["current"],
            alpha=self.alpha["primary"],
        )

        ax.bar(
            x + width / 2,
            optimal_metric,
            width,
            label=f"Optimal {metric_name}",
            color=self.colors["optimal"],
            alpha=self.alpha["primary"],
        )

        # Add annotations
        for i, (curr, opt) in enumerate(zip(current_metric, optimal_metric)):
            pct_change = ((opt / curr) - 1) * 100
            self.add_percentage_annotation(ax, i, max(curr, opt), pct_change)

        self.setup_axis(
            ax,
            title=f"Channel {metric_name} Comparison",
            ylabel=metric_name,
            xticks=x,
            xticklabels=channels,
            rotation=45,
        )

        self.add_legend(ax)
        self.finalize_figure()

        logger.info("Summary metrics plot generated successfully")
        return fig

    def cleanup(self) -> None:
        """Clean up all plots."""
        logger.debug("Starting cleanup of plot resources")
        super().cleanup()
        plt.close("all")
        logger.debug("Cleanup completed")