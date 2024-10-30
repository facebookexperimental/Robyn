import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from dataclasses import dataclass
from robyn.allocator.entities.allocation_results import AllocationResult


class AllocationPlotter:
    """Creates visualizations for allocation results."""

    def __init__(self):
        """Initialize plotter with basic settings."""
        # Set basic plot style
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False

        self.colors = plt.cm.Set2.colors
        self.fig_size = (12, 8)

    def plot_all(self, result: AllocationResult) -> Dict[str, plt.Figure]:
        """Create all allocation plots."""
        plots = {
            "spend_share": self.plot_spend_share_comparison(result),
            "response_curves": self.plot_response_curves(result),
            "spend_response_scatter": self.plot_spend_response_scatter(result),
            "roi_comparison": self.plot_roi_comparison(result),
            "spend_response_bars": self.plot_spend_response_bars(result),
        }
        return plots

    def plot_spend_share_comparison(self, result: AllocationResult) -> plt.Figure:
        """Plot channel spend share comparison between current and optimal."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        df = result.optimal_allocations
        current_share = df["current_spend"] / df["current_spend"].sum()
        optimal_share = df["optimal_spend"] / df["optimal_spend"].sum()

        x = np.arange(len(df["channel"]))
        width = 0.35

        ax.bar(x - width / 2, current_share, width, label="Current", color="lightgray")
        ax.bar(x + width / 2, optimal_share, width, label="Optimal", color="skyblue")

        ax.set_ylabel("Share of Total Spend")
        ax.set_title("Channel Spend Share: Current vs Optimal")
        ax.set_xticks(x)
        ax.set_xticklabels(df["channel"], rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_response_curves(self, result: AllocationResult) -> plt.Figure:
        """Plot response curves for each channel with current and optimal points."""
        df = result.response_curves
        channels = df["channel"].unique()
        n_channels = len(channels)

        # Calculate grid dimensions
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, channel in enumerate(channels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            channel_data = df[df["channel"] == channel]

            # Plot response curve
            ax.plot(channel_data["spend"], channel_data["response"], "b-", alpha=0.6)

            # Plot current point
            current = channel_data[channel_data["is_current"]].iloc[0]
            ax.scatter(current["spend"], current["response"], color="red", label="Current", s=100)

            # Plot optimal point
            optimal = channel_data[channel_data["is_optimal"]].iloc[0]
            ax.scatter(optimal["spend"], optimal["response"], color="green", label="Optimal", s=100)

            ax.set_title(f"{channel} Response Curve")
            ax.set_xlabel("Spend")
            ax.set_ylabel("Response")
            ax.legend()

        # Remove empty subplots
        for idx in range(n_channels, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        return fig

    def plot_spend_response_scatter(self, result: AllocationResult) -> plt.Figure:
        """Create scatter plot showing spend vs response relationship."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        df = result.optimal_allocations

        current = ax.scatter(df["current_spend"], df["current_response"], label="Current", color="red", s=100)
        optimal = ax.scatter(df["optimal_spend"], df["optimal_response"], label="Optimal", color="green", s=100)

        # Add arrows showing the change
        for _, row in df.iterrows():
            ax.annotate(
                "",
                xy=(row["optimal_spend"], row["optimal_response"]),
                xytext=(row["current_spend"], row["current_response"]),
                arrowprops=dict(arrowstyle="->"),
            )

        ax.set_xlabel("Spend")
        ax.set_ylabel("Response")
        ax.set_title("Channel Spend vs Response")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_roi_comparison(self, result: AllocationResult) -> plt.Figure:
        """Plot ROI comparison between current and optimal allocations."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        df = result.optimal_allocations

        current_roi = df["current_response"] / df["current_spend"]
        optimal_roi = df["optimal_response"] / df["optimal_spend"]

        x = np.arange(len(df["channel"]))
        width = 0.35

        ax.bar(x - width / 2, current_roi, width, label="Current ROI", color="lightgray")
        ax.bar(x + width / 2, optimal_roi, width, label="Optimal ROI", color="skyblue")

        ax.set_ylabel("ROI")
        ax.set_title("Channel ROI: Current vs Optimal")
        ax.set_xticks(x)
        ax.set_xticklabels(df["channel"], rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_spend_response_bars(self, result: AllocationResult) -> plt.Figure:
        """Create bar plot showing spend and response changes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        df = result.optimal_allocations

        # Spend changes
        spend_pct_change = (df["optimal_spend"] - df["current_spend"]) / df["current_spend"] * 100
        colors = ["red" if x < 0 else "green" for x in spend_pct_change]
        ax1.bar(df["channel"], spend_pct_change, color=colors)
        ax1.set_title("Spend Change %")
        ax1.set_xticklabels(df["channel"], rotation=45, ha="right")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Response changes
        response_pct_change = (df["optimal_response"] - df["current_response"]) / df["current_response"] * 100
        colors = ["red" if x < 0 else "green" for x in response_pct_change]
        ax2.bar(df["channel"], response_pct_change, color=colors)
        ax2.set_title("Response Change %")
        ax2.set_xticklabels(df["channel"], rotation=45, ha="right")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        return fig

    def save_plots(self, plots: Dict[str, plt.Figure], path: str) -> None:
        """Save all plots to specified directory."""
        for name, fig in plots.items():
            fig.savefig(f"{path}/allocation_{name}.png")
            plt.close(fig)
