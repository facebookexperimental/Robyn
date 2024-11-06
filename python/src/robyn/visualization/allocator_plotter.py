from robyn.visualization.base_visualizer import BaseVisualizer
from robyn.allocator.entities.allocation_results import AllocationResult
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict


class AllocationPlotter(BaseVisualizer):
    def __init__(self, result: AllocationResult):
        super().__init__(style="bmh")
        self.result = result
        plt.style.use("bmh")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "axes.grid": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        self.fig_size = (12, 8)
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))
        self.current_color = "lightgray"
        self.optimal_color = "#4688C7"
        self.positive_color = "#2ECC71"
        self.negative_color = "#E74C3C"

    def plot_all(self) -> Dict[str, plt.Figure]:
        figures = {}
        figures["spend_allocation"] = self.plot_spend_allocation()
        figures["response_curves"] = self.plot_response_curves()
        figures["efficiency_frontier"] = self.plot_efficiency_frontier()
        figures["spend_vs_response"] = self.plot_spend_vs_response()
        figures["summary_metrics"] = self.plot_summary_metrics()
        return figures

    def plot_spend_allocation(self) -> plt.Figure:
        if self.result is None:
            raise ValueError("No allocation results available. Call plot_all() first.")

        fig, ax = plt.subplots(figsize=self.fig_size)
        optimal_allocations = self.result.optimal_allocations

        channels = optimal_allocations["channel"].values
        x = np.arange(len(channels))
        width = 0.35

        current_spend = optimal_allocations["current_spend"].values
        ax.bar(
            x - width / 2,
            current_spend,
            width,
            label="Current",
            color=self.current_color,
            edgecolor="gray",
            alpha=0.7,
        )

        optimal_spend = optimal_allocations["optimal_spend"].values
        ax.bar(
            x + width / 2,
            optimal_spend,
            width,
            label="Optimized",
            color=self.optimal_color,
            edgecolor="gray",
            alpha=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right")
        ax.set_ylabel("Spend")
        ax.set_title("Media Spend Allocation")
        ax.legend()

        for i, (curr, opt) in enumerate(zip(current_spend, optimal_spend)):
            pct_change = ((opt / curr) - 1) * 100
            color = self.positive_color if pct_change >= 0 else self.negative_color
            ax.text(
                x[i],
                max(curr, opt),
                f"{pct_change:.1f}%",
                ha="center",
                va="bottom",
                color=color,
            )

        plt.tight_layout()
        return fig

    def plot_response_curves(self) -> plt.Figure:
        if self.result is None:
            raise ValueError("No allocation results available. Call plot_all() first.")

        curves_df = self.result.response_curves
        channels = curves_df["channel"].unique()
        n_channels = len(channels)
        ncols = min(3, n_channels)
        nrows = (n_channels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

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
                color=self.optimal_color,
                alpha=0.6,
            )

            current_data = channel_data[channel_data["is_current"]]
            if not current_data.empty:
                ax.scatter(
                    current_data["spend"].iloc[0],
                    current_data["response"].iloc[0],
                    color=self.negative_color,
                    label="Current",
                    s=100,
                )

            optimal_data = channel_data[channel_data["is_optimal"]]
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

        for idx in range(n_channels, nrows * ncols):
            fig.delaxes(axes[idx // ncols, idx % ncols])

        plt.tight_layout()
        return fig

    def plot_efficiency_frontier(self) -> plt.Figure:
        if self.result is None:
            raise ValueError("No allocation results available. Call plot_all() first.")

        fig, ax = plt.subplots(figsize=self.fig_size)

        optimal_allocations = self.result.optimal_allocations
        current_total_spend = optimal_allocations["current_spend"].sum()
        current_total_response = optimal_allocations["current_response"].sum()
        optimal_total_spend = optimal_allocations["optimal_spend"].sum()
        optimal_total_response = optimal_allocations["optimal_response"].sum()

        ax.scatter(
            current_total_spend,
            current_total_response,
            color=self.negative_color,
            s=100,
            label="Current",
            zorder=2,
        )
        ax.scatter(
            optimal_total_spend,
            optimal_total_response,
            color=self.positive_color,
            s=100,
            label="Optimal",
            zorder=2,
        )

        ax.plot(
            [current_total_spend, optimal_total_spend],
            [current_total_response, optimal_total_response],
            "--",
            color="gray",
            alpha=0.5,
            zorder=1,
        )

        pct_spend_change = ((optimal_total_spend / current_total_spend) - 1) * 100
        pct_response_change = (
            (optimal_total_response / current_total_response) - 1
        ) * 100
        ax.annotate(
            f"Spend: {pct_spend_change:.1f}%\nResponse: {pct_response_change:.1f}%",
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

    def plot_spend_vs_response(self) -> plt.Figure:
        if self.result is None:
            raise ValueError("No allocation results available. Call plot_all() first.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        df = self.result.optimal_allocations
        channels = df["channel"].values
        x = np.arange(len(channels))

        spend_pct = ((df["optimal_spend"] / df["current_spend"]) - 1) * 100
        colors = [
            self.positive_color if pct >= 0 else self.negative_color
            for pct in spend_pct
        ]
        ax1.bar(x, spend_pct, color=colors, alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(channels, rotation=45, ha="right"))
        ax1.set_ylabel("Spend Change %")
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.2)
        ax1.grid(True, alpha=0.3)
        for i, pct in enumerate(spend_pct):
            ax1.text(
                i,
                pct + (2 if pct >= 0 else -5),
                f"{pct:.1f}%",
                ha="center",
                va="bottom" if pct >= 0 else "top",
            )

        response_pct = ((df["optimal_response"] / df["current_response"]) - 1) * 100
        colors = [
            self.positive_color if pct >= 0 else self.negative_color
            for pct in response_pct
        ]
        ax2.bar(x, response_pct, color=colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(channels, rotation=45, ha="right"))
        ax2.set_ylabel("Response Change %")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.2)
        ax2.grid(True, alpha=0.3)
        for i, pct in enumerate(response_pct):
            ax2.text(
                i,
                pct + (2 if pct >= 0 else -5),
                f"{pct:.1f}%",
                ha="center",
                va="bottom" if pct >= 0 else "top",
            )

        plt.tight_layout()
        return fig

    def plot_summary_metrics(self) -> plt.Figure:
        if self.result is None:
            raise ValueError("No allocation results available. Call plot_all() first.")
        fig, ax = plt.subplots(figsize=self.fig_size)
        optimal_allocations = self.result.optimal_allocations
        channels = optimal_allocations["channel"].values
        dep_var_type = self.result.metrics.get("dep_var_type")
        if dep_var_type == "revenue":
            current_metric = (
                optimal_allocations["current_response"]
                / optimal_allocations["current_spend"]
            )
            optimal_metric = (
                optimal_allocations["optimal_response"]
                / optimal_allocations["optimal_spend"]
            )
            metric_name = "ROI"
        else:
            current_metric = (
                optimal_allocations["current_spend"]
                / optimal_allocations["current_response"]
            )
            optimal_metric = (
                optimal_allocations["optimal_spend"]
                / optimal_allocations["optimal_response"]
            )
            metric_name = "CPA"
        x = np.arange(len(channels))
        width = 0.35
        ax.bar(
            x - width / 2,
            current_metric,
            width,
            label=f"Current {metric_name}",
            color=self.current_color,
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            optimal_metric,
            width,
            label=f"Optimal {metric_name}",
            color=self.optimal_color,
            alpha=0.7,
        )
        for i, (curr, opt) in enumerate(zip(current_metric, optimal_metric)):
            change = ((opt / curr) - 1) * 100
            color = self.positive_color if change >= 0 else self.negative_color
            ax.text(
                i,
                max(curr, opt),
                f"{change:.1f}%",
                ha="center",
                va="bottom",
                color=color,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Channel {metric_name} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
