from robyn.visualization.base_visualizer import BaseVisualizer
from robyn.allocator.entities.allocation_results import AllocationResult
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict


class AllocationPlotter(BaseVisualizer):
    """Plotter class for allocation results visualization."""

    def __init__(self, result: AllocationResult):
        """Initialize plotter with allocation results."""
        super().__init__(style="bmh")
        self.result = result
        if self.result is None:
            raise ValueError("AllocationResult cannot be None")

    def plot_all(self) -> Dict[str, plt.Figure]:
        """Generate all allocation plots."""
        figures = {}
        figures["spend_allocation"] = self.plot_spend_allocation()
        figures["response_curves"] = self.plot_response_curves()
        figures["efficiency_frontier"] = self.plot_efficiency_frontier()
        figures["spend_vs_response"] = self.plot_spend_vs_response()
        figures["summary_metrics"] = self.plot_summary_metrics()
        return figures

    def plot_spend_allocation(self) -> plt.Figure:
        """Plot spend allocation comparison."""
        fig, ax = self.create_figure()
        optimal_allocations = self.result.optimal_allocations

        channels = optimal_allocations["channel"].values
        x = np.arange(len(channels))
        width = 0.35

        # Plot current spend
        ax.bar(
            x - width / 2,
            optimal_allocations["current_spend"].values,
            width,
            label="Current",
            color=self.colors["current"],
            edgecolor="gray",
            alpha=self.alpha["primary"],
        )

        # Plot optimal spend
        ax.bar(
            x + width / 2,
            optimal_allocations["optimal_spend"].values,
            width,
            label="Optimized",
            color=self.colors["optimal"],
            edgecolor="gray",
            alpha=self.alpha["primary"],
        )

        # Add percentage change annotations
        for i, (curr, opt) in enumerate(
            zip(optimal_allocations["current_spend"].values, optimal_allocations["optimal_spend"].values)
        ):
            pct_change = ((opt / curr) - 1) * 100
            color = self.colors["positive"] if pct_change >= 0 else self.colors["negative"]
            ax.text(
                x[i],
                max(curr, opt),
                f"{pct_change:.1f}%",
                ha="center",
                va="bottom",
                color=color,
                fontsize=self.font_sizes["annotation"],
            )

        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=self.font_sizes["tick"])
        ax.set_ylabel("Spend", fontsize=self.font_sizes["label"])
        ax.set_title("Media Spend Allocation", fontsize=self.font_sizes["title"])
        ax.legend(fontsize=self.font_sizes["annotation"])
        ax.grid(True, alpha=self.alpha["grid"])

        plt.tight_layout()
        return fig

    def plot_response_curves(self) -> plt.Figure:
        """Plot response curves for each channel."""
        curves_df = self.result.response_curves
        channels = curves_df["channel"].unique()
        n_channels = len(channels)
        ncols = min(3, n_channels)
        nrows = (n_channels + ncols - 1) // ncols

        fig, axes = self.create_figure(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx, channel in enumerate(channels):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            channel_data = curves_df[curves_df["channel"] == channel]
            ax.plot(
                channel_data["spend"],
                channel_data["response"],
                color=self.colors["optimal"],
                alpha=self.alpha["primary"],
            )

            current_data = channel_data[channel_data["is_current"]]
            if not current_data.empty:
                ax.scatter(
                    current_data["spend"].iloc[0],
                    current_data["response"].iloc[0],
                    color=self.colors["negative"],
                    label="Current",
                    s=100,
                )

            optimal_data = channel_data[channel_data["is_optimal"]]
            if not optimal_data.empty:
                ax.scatter(
                    optimal_data["spend"].iloc[0],
                    optimal_data["response"].iloc[0],
                    color=self.colors["positive"],
                    label="Optimal",
                    s=100,
                )

            ax.set_title(f"{channel} Response Curve", fontsize=self.font_sizes["title"])
            ax.legend(fontsize=self.font_sizes["annotation"])
            ax.grid(True, alpha=self.alpha["grid"])

        for idx in range(n_channels, nrows * ncols):
            fig.delaxes(axes[idx // ncols, idx % ncols])

        plt.tight_layout()
        return fig

    def plot_efficiency_frontier(self) -> plt.Figure:
        """Plot efficiency frontier."""
        fig, ax = self.create_figure()

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

        # Add annotations
        pct_spend_change = ((optimal_total_spend / current_total_spend) - 1) * 100
        pct_response_change = ((optimal_total_response / current_total_response) - 1) * 100
        ax.annotate(
            f"Spend: {pct_spend_change:.1f}%\nResponse: {pct_response_change:.1f}%",
            xy=(optimal_total_spend, optimal_total_response),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=self.font_sizes["annotation"],
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7),
        )

        # Customize plot
        ax.set_xlabel("Total Spend", fontsize=self.font_sizes["label"])
        ax.set_ylabel("Total Response", fontsize=self.font_sizes["label"])
        ax.set_title("Efficiency Frontier", fontsize=self.font_sizes["title"])
        ax.legend(fontsize=self.font_sizes["annotation"])
        ax.grid(True, alpha=self.alpha["grid"])

        plt.tight_layout()
        return fig

    def plot_spend_vs_response(self) -> plt.Figure:
        """Plot spend vs response changes."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.current_figure = fig  # Track the figure

        df = self.result.optimal_allocations
        channels = df["channel"].values
        x = np.arange(len(channels))

        # Spend changes
        spend_pct = ((df["optimal_spend"] / df["current_spend"]) - 1) * 100
        colors = [self.colors["positive"] if pct >= 0 else self.colors["negative"] for pct in spend_pct]
        ax1.bar(x, spend_pct, color=colors, alpha=self.alpha["primary"])
        self._customize_change_axis(ax1, x, channels, spend_pct, "Spend Change %")

        # Response changes
        response_pct = ((df["optimal_response"] / df["current_response"]) - 1) * 100
        colors = [self.colors["positive"] if pct >= 0 else self.colors["negative"] for pct in response_pct]
        ax2.bar(x, response_pct, color=colors, alpha=self.alpha["primary"])
        self._customize_change_axis(ax2, x, channels, response_pct, "Response Change %")

        plt.tight_layout()
        return fig

    def _customize_change_axis(self, ax, x, channels, pct_values, ylabel):
        """Helper method to customize change plot axes."""
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=self.font_sizes["tick"])
        ax.set_ylabel(ylabel, fontsize=self.font_sizes["label"])
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)
        ax.grid(True, alpha=self.alpha["grid"])

        for i, pct in enumerate(pct_values):
            ax.text(
                i,
                pct + (2 if pct >= 0 else -5),
                f"{pct:.1f}%",
                ha="center",
                va="bottom" if pct >= 0 else "top",
                fontsize=self.font_sizes["annotation"],
            )

    def plot_summary_metrics(self) -> plt.Figure:
        """Plot summary metrics."""
        fig, ax = self.create_figure()
        optimal_allocations = self.result.optimal_allocations
        channels = optimal_allocations["channel"].values

        # Calculate metrics based on type
        dep_var_type = self.result.metrics.get("dep_var_type")
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

        # Add change annotations
        for i, (curr, opt) in enumerate(zip(current_metric, optimal_metric)):
            change = ((opt / curr) - 1) * 100
            color = self.colors["positive"] if change >= 0 else self.colors["negative"]
            ax.text(
                i,
                max(curr, opt),
                f"{change:.1f}%",
                ha="center",
                va="bottom",
                color=color,
                fontsize=self.font_sizes["annotation"],
            )

        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=self.font_sizes["tick"])
        ax.set_ylabel(metric_name, fontsize=self.font_sizes["label"])
        ax.set_title(f"Channel {metric_name} Comparison", fontsize=self.font_sizes["title"])
        ax.legend(fontsize=self.font_sizes["annotation"])
        ax.grid(True, alpha=self.alpha["grid"])

        plt.tight_layout()
        return fig

    def cleanup(self) -> None:
        """Clean up all plots."""
        super().cleanup()
        plt.close("all")  # Ensure all figures are closed
