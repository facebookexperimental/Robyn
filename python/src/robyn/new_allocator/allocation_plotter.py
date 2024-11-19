# allocation_plotter.py

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from robyn.new_allocator.optimization.objective_function import ObjectiveFunction
from matplotlib.gridspec import GridSpec


class AllocationPlotter:
    """Creates visualization plots for budget allocation results."""

    def __init__(self, dark_mode: bool = False):
        """Initialize plotter with style settings."""
        self.dark_mode = dark_mode
        self._set_style()

    def _set_style(self):
        """Sets plotting style."""
        # plt.style.use("seaborn")
        # if self.dark_mode:
        #     plt.style.use("dark_background")

        # Custom color palette
        self.colors = {
            "primary": "#59B3D2",
            "secondary": "#E5586E",
            "tertiary": "#38618C",
            "initial": "grey",
            "bounded": "steelblue",
            "unbounded": "darkgoldenrod4",
        }

    def create_onepager(
        self,
        dt_optim_out: pd.DataFrame,
        plot_data: Dict[str, Dict[str, np.ndarray]],
        scenario: str,
        date_range: Tuple[str, str],
        interval_type: str = "Week",
        figsize: Tuple[int, int] = (15, 20),
    ) -> plt.Figure:
        """Creates allocation one-pager plot.

        Args:
            dt_optim_out: Allocation results DataFrame
            plot_data: Response curves data from _prepare_response_curves_data
            scenario: Optimization scenario used
            date_range: Tuple of (start_date, end_date)
            interval_type: Time interval type (e.g. "Week")
            figsize: Figure size tuple

        Returns:
            Matplotlib figure containing the one-pager plots
        """
        # Create figure with grid
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, height_ratios=[1, 2, 2], hspace=0.3)

        # Create subplots
        ax1 = fig.add_subplot(gs[0])  # Total Budget Optimization
        ax2 = fig.add_subplot(gs[1])  # Budget Allocation per Media
        ax3 = fig.add_subplot(gs[2])  # Response Curves

        # Plot total budget optimization
        self._plot_total_budget_optimization(ax1, dt_optim_out, scenario)

        # Plot budget allocation
        self._plot_budget_allocation(ax2, dt_optim_out, interval_type)

        # Plot response curves
        self._plot_response_curves(ax3, plot_data, dt_optim_out)

        # Add title and metadata
        fig.suptitle(
            f"Budget Allocation One-pager\n" f"Date Range: {date_range[0]} to {date_range[1]}", fontsize=14, y=0.95
        )

        return fig

    def _plot_total_budget_optimization(self, ax: plt.Axes, dt_optim_out: pd.DataFrame, scenario: str) -> None:
        """Plots total budget optimization results."""
        # Calculate totals
        init_spend_total = dt_optim_out["initSpendUnit"].sum()
        init_response_total = dt_optim_out["initResponseUnit"].sum()
        optm_spend_total = dt_optim_out["optmSpendUnit"].sum()
        optm_response_total = dt_optim_out["optmResponseUnit"].sum()

        # Prepare data
        metrics = pd.DataFrame(
            {
                "Metric": ["Total Spend", "Total Response"],
                "Initial": [init_spend_total, init_response_total],
                "Optimized": [optm_spend_total, optm_response_total],
            }
        )

        # Create grouped bar plot
        x = np.arange(len(metrics["Metric"]))
        width = 0.35

        ax.bar(x - width / 2, metrics["Initial"], width, label="Initial", color=self.colors["initial"])
        ax.bar(x + width / 2, metrics["Optimized"], width, label="Optimized", color=self.colors["bounded"])

        # Add value labels
        for i in x:
            ax.text(i - width / 2, metrics["Initial"][i], f'{metrics["Initial"][i]:,.0f}', ha="center", va="bottom")
            ax.text(
                i + width / 2, metrics["Optimized"][i], f'{metrics["Optimized"][i]:,.0f}', ha="center", va="bottom"
            )

        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(metrics["Metric"])
        ax.set_title("Total Budget Optimization Result")
        ax.legend()

    def _plot_budget_allocation(self, ax: plt.Axes, dt_optim_out: pd.DataFrame, interval_type: str) -> None:
        """Plots budget allocation per media variable."""
        # Prepare data for plotting
        plot_data = pd.DataFrame(
            {
                "Channel": dt_optim_out["channels"],
                "Initial Share": dt_optim_out["initSpendShare"] * 100,
                "Optimized Share": dt_optim_out["optmSpendShareUnit"] * 100,
                "Initial Spend": dt_optim_out["initSpendUnit"],
                "Optimized Spend": dt_optim_out["optmSpendUnit"],
            }
        )

        # Create horizontal bar plot
        y_pos = np.arange(len(plot_data["Channel"]))

        # Plot share bars
        ax.barh(
            y_pos - 0.2,
            plot_data["Initial Share"],
            height=0.3,
            color=self.colors["initial"],
            alpha=0.7,
            label="Initial Share (%)",
        )
        ax.barh(
            y_pos + 0.2,
            plot_data["Optimized Share"],
            height=0.3,
            color=self.colors["bounded"],
            alpha=0.7,
            label="Optimized Share (%)",
        )

        # Add spend values as text
        for i, row in plot_data.iterrows():
            ax.text(row["Initial Share"], i - 0.2, f'${row["Initial Spend"]:,.0f}', va="center", ha="left")
            ax.text(row["Optimized Share"], i + 0.2, f'${row["Optimized Spend"]:,.0f}', va="center", ha="left")

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data["Channel"])
        ax.set_xlabel("Share of Budget (%)")
        ax.set_title(f"Budget Allocation per Paid Media Variable per {interval_type}")
        ax.legend(loc="upper right")

    def _plot_response_curves(self, ax: plt.Axes, plot_data: Dict, dt_optim_out: pd.DataFrame) -> None:
        """Plots response curves for each channel."""
        # Plot response curves for each channel
        for channel, data in plot_data.items():
            # Plot response curve
            ax.plot(data["spend"], data["response"], label=channel, alpha=0.7)

            # Plot current and optimized points
            current = dt_optim_out[dt_optim_out["channels"] == channel].iloc[0]
            ax.scatter(
                current["initSpendUnit"], current["initResponseUnit"], marker="o", s=100, label=f"{channel} (Current)"
            )
            ax.scatter(
                current["optmSpendUnit"],
                current["optmResponseUnit"],
                marker="*",
                s=100,
                label=f"{channel} (Optimized)",
            )

        # Customize plot
        ax.set_xlabel("Spend")
        ax.set_ylabel("Response")
        ax.set_title("Simulated Response Curves per Channel")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """Saves plot to file."""
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _prepare_response_curves_data(
        self, channels: List[str], allocation_df: pd.DataFrame, objective_function: ObjectiveFunction
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepares data for response curves plotting.

        Args:
            channels: List of media channel names
            allocation_df: DataFrame containing allocation results
            objective_function: Configured ObjectiveFunction instance

        Returns:
            Dictionary containing spend and response data for each channel
        """
        plot_data = {}

        for channel in channels:
            # Get channel's current and optimal spend
            channel_data = allocation_df[allocation_df["channels"] == channel].iloc[0]
            current_spend = channel_data["initSpendUnit"]
            optimal_spend = channel_data["optmSpendUnit"]

            # Create spend range for curve
            max_spend = max(current_spend, optimal_spend) * 1.5
            spend_range = np.linspace(0, max_spend, 100)

            # Calculate responses
            responses = [
                objective_function.calculate_response(x=np.array([spend]), channel_name=channel, get_sum=True)
                for spend in spend_range
            ]

            plot_data[channel] = {"spend": spend_range, "response": np.array(responses)}

        return plot_data
