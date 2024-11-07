from robyn.visualization.base_visualizer import BaseVisualizer
from robyn.allocator.entities.allocation_results import AllocationResult
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict


class AllocationPlotter(BaseVisualizer):
    """Plotter class for allocation results visualization."""

    def __init__(self, result: AllocationResult):
        """
        Initialize plotter with allocation results.

        Args:
            result: Allocation results to plot
        """
        super().__init__(style="bmh")
        self.result = result
        if self.result is None:
            raise ValueError("AllocationResult cannot be None")

    def plot_all(self) -> Dict[str, plt.Figure]:
        """
        Generate all allocation plots.

        Returns:
            Dictionary of figures keyed by plot name
        """
        figures = {}
        try:
            figures["spend_allocation"] = self.plot_spend_allocation()
            figures["response_curves"] = self.plot_response_curves()
            figures["efficiency_frontier"] = self.plot_efficiency_frontier()
            figures["spend_vs_response"] = self.plot_spend_vs_response()
            figures["summary_metrics"] = self.plot_summary_metrics()
        finally:
            self.cleanup()
        return figures

    def plot_spend_allocation(self) -> plt.Figure:
        """Plot spend allocation comparison."""
        # Create figure
        fig, ax = self.create_figure()
        optimal_allocations = self.result.optimal_allocations

        # Prepare data
        channels = optimal_allocations["channel"].values
        x = np.arange(len(channels))
        width = 0.35

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
        for i, (curr, opt) in enumerate(
            zip(optimal_allocations["current_spend"].values, optimal_allocations["optimal_spend"].values)
        ):
            pct_change = ((opt / curr) - 1) * 100
            self.add_percentage_annotation(ax, x[i], max(curr, opt), pct_change)

        # Setup axis
        self.setup_axis(
            ax, title="Media Spend Allocation", ylabel="Spend", xticks=x, xticklabels=channels, rotation=45
        )

        # Add legend and finalize
        self.add_legend(ax)
        self.finalize_figure()

        return fig

    def plot_response_curves(self) -> plt.Figure:
        """Plot response curves for each channel."""
        # Prepare data
        curves_df = self.result.response_curves
        channels = curves_df["channel"].unique()
        n_channels = len(channels)
        ncols = min(3, n_channels)
        nrows = (n_channels + ncols - 1) // ncols

        # Create figure
        fig, axes = self.create_figure(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

        # Handle single subplot case
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each channel
        for idx, channel in enumerate(channels):
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
                ax.scatter(
                    optimal_data["spend"].iloc[0],
                    optimal_data["response"].iloc[0],
                    color=self.colors["positive"],
                    label="Optimal",
                    s=100,
                )

            # Setup subplot
            self.setup_axis(ax, title=f"{channel} Response Curve")
            self.add_legend(ax)

        # Remove empty subplots and finalize
        for idx in range(n_channels, nrows * ncols):
            fig.delaxes(axes[idx // ncols, idx % ncols])

        self.finalize_figure()
        return fig

    def plot_efficiency_frontier(self) -> plt.Figure:
        """Plot efficiency frontier."""
        # Create figure
        fig, ax = self.create_figure()

        # Calculate totals
        optimal_allocations = self.result.optimal_allocations
        current_total_spend = optimal_allocations["current_spend"].sum()
        current_total_response = optimal_allocations["current_response"].sum()
        optimal_total_spend = optimal_allocations["optimal_spend"].sum()
        optimal_total_response = optimal_allocations["optimal_response"].sum()

        # Plot points
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

        # Connect points
        ax.plot(
            [current_total_spend, optimal_total_spend],
            [current_total_response, optimal_total_response],
            "--",
            color=self.colors["neutral"],
            alpha=self.alpha["secondary"],
            zorder=1,
        )

        # Add percentage changes annotation
        pct_spend_change = ((optimal_total_spend / current_total_spend) - 1) * 100
        pct_response_change = ((optimal_total_response / current_total_response) - 1) * 100

        ax.annotate(
            f"Spend: {pct_spend_change:.1f}%\nResponse: {pct_response_change:.1f}%",
            xy=(optimal_total_spend, optimal_total_response),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=self.font_sizes["annotation"],
            bbox=dict(facecolor="white", edgecolor=self.colors["neutral"], alpha=self.alpha["annotation"]),
        )

        # Setup axis
        self.setup_axis(ax, title="Efficiency Frontier", xlabel="Total Spend", ylabel="Total Response")

        self.add_legend(ax)
        self.finalize_figure()

        return fig

    def plot_spend_vs_response(self) -> plt.Figure:
        """Plot spend vs response changes."""
        # Create figure
        fig, (ax1, ax2) = self.create_figure(nrows=2, ncols=1, figsize=(12, 10))

        # Get data
        df = self.result.optimal_allocations
        channels = df["channel"].values
        x = np.arange(len(channels))

        # Plot spend changes
        spend_pct = ((df["optimal_spend"] / df["current_spend"]) - 1) * 100
        colors = [self.colors["positive"] if pct >= 0 else self.colors["negative"] for pct in spend_pct]

        ax1.bar(x, spend_pct, color=colors, alpha=self.alpha["primary"])
        self._plot_change_axis(ax1, x, channels, spend_pct, "Spend Change %")

        # Plot response changes
        response_pct = ((df["optimal_response"] / df["current_response"]) - 1) * 100
        colors = [self.colors["positive"] if pct >= 0 else self.colors["negative"] for pct in response_pct]

        ax2.bar(x, response_pct, color=colors, alpha=self.alpha["primary"])
        self._plot_change_axis(ax2, x, channels, response_pct, "Response Change %")

        self.finalize_figure(adjust_spacing=True)
        return fig

    def _plot_change_axis(
        self, ax: plt.Axes, x: np.ndarray, channels: np.ndarray, pct_values: np.ndarray, ylabel: str
    ) -> None:
        """Helper method to setup change plot axes."""
        self.setup_axis(ax, ylabel=ylabel, xticks=x, xticklabels=channels, rotation=45)

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)

        for i, pct in enumerate(pct_values):
            self.add_percentage_annotation(
                ax, i, pct + (2 if pct >= 0 else -5), pct, va="bottom" if pct >= 0 else "top"
            )

    def plot_summary_metrics(self) -> plt.Figure:
        """Plot summary metrics."""
        # Create figure
        fig, ax = self.create_figure()

        # Get data
        optimal_allocations = self.result.optimal_allocations
        channels = optimal_allocations["channel"].values
        dep_var_type = self.result.metrics.get("dep_var_type")

        # Calculate metrics
        if dep_var_type == "revenue":
            current_metric = optimal_allocations["current_response"] / optimal_allocations["current_spend"]
            optimal_metric = optimal_allocations["optimal_response"] / optimal_allocations["optimal_spend"]
            metric_name = "ROI"
        else:
            current_metric = optimal_allocations["current_spend"] / optimal_allocations["current_response"]
            optimal_metric = optimal_allocations["optimal_spend"] / optimal_allocations["optimal_response"]
            metric_name = "CPA"

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

        # Setup axis
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

        return fig

    def cleanup(self) -> None:
        """Clean up all plots."""
        super().cleanup()
        plt.close("all")
