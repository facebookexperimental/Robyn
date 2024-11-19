import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional
from robyn.allocator.entities.allocation_results import AllocationResult
import pandas as pd

logger = logging.getLogger(__name__)


class AllocationPlotter:
    """Creates allocation result visualizations matching R's Robyn output."""

    def __init__(self, result: AllocationResult):
        """
        Initialize plotter with allocation results.

        Args:
            result: AllocationResult instance containing optimization results
        """
        logger.debug("Initializing AllocationPlotter")

        if not isinstance(result, AllocationResult):
            logger.error("Input must be an AllocationResult instance")
            raise TypeError("Input must be an AllocationResult instance")

        self.result = result
        # Color scheme matching R's palette
        self.colors = {
            "initial": "#A0A0A0",  # Gray
            "bounded": "#4682B4",  # Steel blue
            "bounded_x3": "#8B4513",  # Saddle brown
            "gray_area": "#808080",  # Gray for carryover areas
            "text": "#03396C",  # Dark blue for text
        }
        logger.info("AllocationPlotter initialized successfully")

    def create_onepager(self) -> plt.Figure:
        """Create combined one-pager plot matching R's layout."""
        logger.debug("Creating onepager plot")

        # Create figure with 3 row layout
        fig = plt.figure(figsize=(17, 19))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 2])

        # 1. Total Budget Optimization Result
        ax_budget = fig.add_subplot(gs[0])
        self._plot_total_budget(ax_budget)

        # 2. Budget Allocation Table
        ax_alloc = fig.add_subplot(gs[1])
        self._plot_allocation_table(ax_alloc)

        # 3. Response Curves
        ax_curves = fig.add_subplot(gs[2])
        self._plot_response_curves(ax_curves)

        # Add title and metadata
        title = f"Budget Allocation Onepager for Model ID {self.result.metrics['model_id']}"
        sim_date_txt = (
            f"Simulation date range: {self.result.metrics['date_range_start']} to "
            f"{self.result.metrics['date_range_end']} "
            f"({self.result.metrics['n_periods']} {self.result.metrics['interval_type']}s) | "
            f"Scenario: {self.result.metrics['scenario']}"
        )

        plt.suptitle(title, fontsize=14, y=0.98)
        plt.figtext(0.02, 0.96, self.result.summary.split("\n")[1], fontsize=10)  # Error metrics
        plt.figtext(0.02, 0.94, sim_date_txt, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        logger.info("Onepager plot created successfully")
        return fig

    def _plot_total_budget(self, ax: plt.Axes) -> None:
        """Plot total budget optimization results."""
        logger.debug("Plotting total budget optimization results")

        df = self.result.optimal_allocations

        # Calculate totals
        initial_spend = df["current_spend"].sum()
        initial_response = df["current_response"].sum()
        optimal_spend = df["optimal_spend"].sum()
        optimal_response = df["optimal_response"].sum()
        unbounded_spend = df["optimal_spend_unbound"].sum() if "optimal_spend_unbound" in df else optimal_spend
        unbounded_response = (
            df["optimal_response_unbound"].sum() if "optimal_response_unbound" in df else optimal_response
        )

        # Prepare bar data
        categories = ["Initial", "Bounded", "Bounded x3"]
        spend_values = [initial_spend, optimal_spend, unbounded_spend]
        response_values = [initial_response, optimal_response, unbounded_response]

        # Calculate metrics for each scenario
        is_revenue = self.result.metrics["dep_var_type"] == "revenue"
        metric_name = "ROAS" if is_revenue else "CPA"

        metric_values = []
        for s, r in zip(spend_values, response_values):
            if is_revenue:
                metric_values.append(r / s)  # ROAS
            else:
                metric_values.append(s / r)  # CPA

        # Create bar positions
        x = np.arange(len(categories))
        width = 0.35

        # Plot bars
        ax.bar(x - width / 2, spend_values, width, label="Spend", color=self.colors["initial"])
        ax.bar(x + width / 2, response_values, width, label="Response", color=self.colors["bounded"])

        # Add labels
        for i in range(len(categories)):
            # Spend value and change
            spend_change = ((spend_values[i] / initial_spend) - 1) * 100
            ax.text(
                x[i] - width / 2,
                spend_values[i],
                f"{spend_values[i]:,.0f}\n{spend_change:+.1f}%",
                ha="center",
                va="bottom",
            )

            # Response value and change
            resp_change = ((response_values[i] / initial_response) - 1) * 100
            ax.text(
                x[i] + width / 2,
                response_values[i],
                f"{response_values[i]:,.0f}\n{resp_change:+.1f}%",
                ha="center",
                va="bottom",
            )

            # Metric value
            ax.text(
                x[i],
                max(spend_values[i], response_values[i]) * 1.1,
                f"{metric_name}: {metric_values[i]:.2f}",
                ha="center",
                va="bottom",
                color=self.colors["text"],
            )

        ax.set_title("Total Budget Optimization Result (scaled up)", pad=40)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        logger.debug("Total budget plot completed")

    def _plot_allocation_table(self, ax: plt.Axes) -> None:
        """Plot media allocation table with shares and metrics."""
        logger.debug("Plotting allocation table")

        df = self.result.optimal_allocations
        channels = df["channel"].values
        n_channels = len(channels)

        # Create bar positions
        y_pos = np.arange(n_channels)
        bar_height = 0.35

        # Plot spend share bars
        spend_share_current = df["current_spend_share"] * 100
        spend_share_optimal = df["optimal_spend_share"] * 100

        ax.barh(
            y_pos - bar_height / 2,
            spend_share_current,
            bar_height,
            label="Current Spend Share",
            color=self.colors["initial"],
        )
        ax.barh(
            y_pos + bar_height / 2,
            spend_share_optimal,
            bar_height,
            label="Optimal Spend Share",
            color=self.colors["bounded"],
        )

        # Add labels
        is_revenue = self.result.metrics["dep_var_type"] == "revenue"
        metric_name = "ROAS" if is_revenue else "CPA"

        for i, (curr, opt) in enumerate(zip(spend_share_current, spend_share_optimal)):
            # Share percentages
            ax.text(curr + 0.5, i - bar_height / 2, f"{curr:.1f}%", va="center")
            ax.text(opt + 0.5, i + bar_height / 2, f"{opt:.1f}%", va="center")

            # Metric values
            if is_revenue:
                curr_metric = df.iloc[i]["current_response"] / df.iloc[i]["current_spend"]
                opt_metric = df.iloc[i]["optimal_response"] / df.iloc[i]["optimal_spend"]
            else:
                curr_metric = df.iloc[i]["current_spend"] / df.iloc[i]["current_response"]
                opt_metric = df.iloc[i]["optimal_spend"] / df.iloc[i]["optimal_response"]

            ax.text(
                max(curr, opt) + 5,
                i,
                f"{metric_name}: {curr_metric:.2f} → {opt_metric:.2f}",
                va="center",
                color=self.colors["text"],
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels)
        ax.set_title("Budget Allocation per Paid Media Variable per Week")
        ax.legend()
        logger.debug("Allocation table plot completed")

    def _plot_response_curves(self, ax: plt.Axes) -> None:
        """Plot response curves with carryover areas."""
        logger.debug("Plotting response curves")

        curves_df = self.result.response_curves
        channels = curves_df["channel"].unique()
        n_channels = len(channels)
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, channel in enumerate(channels):
            logger.debug("Plotting response curve for channel: %s", channel)
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            channel_data = curves_df[curves_df["channel"] == channel]

            # Plot main curve
            ax.plot(channel_data["spend"], channel_data["response"], color=self.colors["bounded"], linewidth=2)

            # Plot carryover area
            if "carryover" in channel_data.columns:
                carryover_mask = channel_data["spend"] <= channel_data["carryover"].iloc[0]
                ax.fill_between(
                    channel_data[carryover_mask]["spend"],
                    0,
                    channel_data[carryover_mask]["response"],
                    color=self.colors["gray_area"],
                    alpha=0.3,
                )

            # Add points and labels
            for point_type in ["current", "optimal"]:
                point = channel_data[channel_data[f"is_{point_type}"]].iloc[0]
                color = self.colors["initial"] if point_type == "current" else self.colors["bounded"]

                ax.scatter(point["spend"], point["response"], color=color, s=100, zorder=5)
                ax.text(
                    point["spend"],
                    point["response"],
                    f"{point['spend']:,.0f}",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            ax.set_title(channel)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Simulated Response Curve per Week", y=1.02)
        plt.tight_layout()

        # Remove empty subplots
        for idx in range(n_channels, n_rows * n_cols):
            fig.delaxes(axes[idx // n_cols, idx % n_cols])

        logger.debug("Response curves plot completed")

    def save(self, filepath: str) -> None:
        """Save current figure to file."""
        logger.info("Saving plot to: %s", filepath)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
