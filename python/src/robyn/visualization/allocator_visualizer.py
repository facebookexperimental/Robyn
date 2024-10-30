from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from robyn.allocator.entities.allocation_results import AllocationResult


class AllocationPlotter:
    """Creates visualizations for allocation results matching R version."""

    def __init__(self):
        """Initialize plotter with default settings."""
        # Use matplotlib's built-in clean style
        plt.style.use("bmh")

        # Set default plot settings
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False

        # Store standard figure size and colors
        self.fig_size = (12, 8)
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))

        # Set color scheme
        self.current_color = "lightgray"
        self.optimal_color = "#4688C7"  # Steel blue
        self.positive_color = "#2ECC71"  # Green
        self.negative_color = "#E74C3C"  # Red

    def plot_all(self, result: AllocationResult) -> Dict[str, plt.Figure]:
        """Generate all one-pager plots for allocation results."""
        return {
            "spend_allocation": self.plot_spend_allocation(result),
            "response_curves": self.plot_response_curves(result),
            "efficiency_frontier": self.plot_efficiency_frontier(result),
            "spend_vs_response": self.plot_spend_vs_response(result),
            "summary_metrics": self.plot_summary_metrics(result),
        }

    def plot_spend_allocation(self, result: AllocationResult) -> plt.Figure:
        """Plot spend allocation comparison between current and optimized."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        df = result.optimal_allocations
        channels = df["channel"].values
        x = np.arange(len(channels))
        width = 0.35

        # Plot bars
        current_spend = df["current_spend"].values
        optimal_spend = df["optimal_spend"].values

        ax.bar(
            x - width / 2, current_spend, width, label="Current", color=self.current_color, edgecolor="gray", alpha=0.7
        )
        ax.bar(
            x + width / 2,
            optimal_spend,
            width,
            label="Optimized",
            color=self.optimal_color,
            edgecolor="gray",
            alpha=0.7,
        )

        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right")
        ax.set_ylabel("Spend")
        ax.set_title("Media Spend Allocation")
        ax.legend()

        # Add spend change percentage labels
        for i, (curr, opt) in enumerate(zip(current_spend, optimal_spend)):
            pct_change = ((opt / curr) - 1) * 100
            color = self.positive_color if pct_change >= 0 else self.negative_color
            ax.text(i, max(curr, opt), f"{pct_change:+.1f}%", ha="center", va="bottom", color=color)

        plt.tight_layout()
        return fig

    def plot_response_curves(self, result: AllocationResult) -> plt.Figure:
        """Plot response curves with current and optimal points."""
        curves_df = result.response_curves
        channels = curves_df["channel"].unique()
        n_channels = len(channels)
        ncols = min(3, n_channels)
        nrows = (n_channels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx, channel in enumerate(channels):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            channel_data = curves_df[curves_df["channel"] == channel]

            # Plot response curve
            ax.plot(channel_data["spend"], channel_data["response"], color=self.optimal_color, alpha=0.6)

            # Add current and optimal points
            current_data = channel_data[channel_data["is_current"]]
            optimal_data = channel_data[channel_data["is_optimal"]]

            if not current_data.empty:
                ax.scatter(
                    current_data["spend"].iloc[0],
                    current_data["response"].iloc[0],
                    color=self.negative_color,
                    label="Current",
                    s=100,
                )
            if not optimal_data.empty:
                ax.scatter(
                    optimal_data["spend"].iloc[0],
                    optimal_data["response"].iloc[0],
                    color=self.positive_color,
                    label="Optimal",
                    s=100,
                )

            ax.set_title(f"{channel} Response Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(n_channels, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        return fig

    def plot_efficiency_frontier(self, result: AllocationResult) -> plt.Figure:
        """Plot efficiency frontier showing spend vs response relationship."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        df = result.optimal_allocations

        # Calculate totals
        current_total_spend = df["current_spend"].sum()
        current_total_response = df["current_response"].sum()
        optimal_total_spend = df["optimal_spend"].sum()
        optimal_total_response = df["optimal_response"].sum()

        # Plot points and line
        ax.scatter(
            current_total_spend, current_total_response, color=self.negative_color, s=100, label="Current", zorder=2
        )
        ax.scatter(
            optimal_total_spend, optimal_total_response, color=self.positive_color, s=100, label="Optimal", zorder=2
        )

        ax.plot(
            [current_total_spend, optimal_total_spend],
            [current_total_response, optimal_total_response],
            "--",
            color="gray",
            alpha=0.5,
            zorder=1,
        )

        # Add labels
        pct_spend_change = ((optimal_total_spend / current_total_spend) - 1) * 100
        pct_response_change = ((optimal_total_response / current_total_response) - 1) * 100

        ax.annotate(
            f"Spend: {pct_spend_change:+.1f}%\nResponse: {pct_response_change:+.1f}%",
            xy=(optimal_total_spend, optimal_total_response),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7),
        )

        ax.set_xlabel("Total Spend")
        ax.set_ylabel("Total Response")
        ax.set_title("Efficiency Frontier")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_spend_vs_response(self, result: AllocationResult) -> plt.Figure:
        """Plot channel-level spend vs response changes."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        df = result.optimal_allocations
        channels = df["channel"].values
        x = np.arange(len(channels))

        # Plot spend changes
        spend_pct = ((df["optimal_spend"] / df["current_spend"]) - 1) * 100
        colors = [self.negative_color if x < 0 else self.positive_color for x in spend_pct]
        ax1.bar(x, spend_pct, color=colors, alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(channels, rotation=45, ha="right")
        ax1.set_ylabel("Spend Change %")
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.2)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(spend_pct):
            ax1.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top")

        # Plot response changes
        response_pct = ((df["optimal_response"] / df["current_response"]) - 1) * 100
        colors = [self.negative_color if x < 0 else self.positive_color for x in response_pct]
        ax2.bar(x, response_pct, color=colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(channels, rotation=45, ha="right")
        ax2.set_ylabel("Response Change %")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.2)
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(response_pct):
            ax2.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top")

        plt.tight_layout()
        return fig

    def plot_summary_metrics(self, result: AllocationResult) -> plt.Figure:
        """Plot summary metrics including ROI/CPA changes."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        df = result.optimal_allocations
        channels = df["channel"].values

        # Calculate ROI or CPA metrics
        if result.metrics.get("dep_var_type") == "revenue":
            current_metric = df["current_response"] / df["current_spend"]
            optimal_metric = df["optimal_response"] / df["optimal_spend"]
            metric_name = "ROI"
        else:
            current_metric = df["current_spend"] / df["current_response"]
            optimal_metric = df["optimal_spend"] / df["optimal_response"]
            metric_name = "CPA"

        x = np.arange(len(channels))
        width = 0.35

        ax.bar(
            x - width / 2, current_metric, width, label=f"Current {metric_name}", color=self.current_color, alpha=0.7
        )
        ax.bar(
            x + width / 2, optimal_metric, width, label=f"Optimal {metric_name}", color=self.optimal_color, alpha=0.7
        )

        # Add value labels
        for i, (curr, opt) in enumerate(zip(current_metric, optimal_metric)):
            pct_change = ((opt / curr) - 1) * 100
            color = self.positive_color if pct_change >= 0 else self.negative_color
            ax.text(i, max(curr, opt), f"{pct_change:+.1f}%", ha="center", va="bottom", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Channel {metric_name} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_plots(self, plots: Dict[str, plt.Figure], directory: str) -> None:
        """Save all plots to specified directory."""
        for name, fig in plots.items():
            fig.savefig(f"{directory}/allocation_{name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
