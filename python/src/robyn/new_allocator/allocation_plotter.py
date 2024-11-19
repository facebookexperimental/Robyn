# allocation_plotter.py

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from robyn.new_allocator.optimization.objective_function import ObjectiveFunction


class AllocationPlotter:
    """Creates visualization plots for budget allocation results matching R implementation."""

    def __init__(self):
        """Initialize plotter with style settings matching R."""
        self.colors = {"initial": "grey", "bounded": "steelblue"}
        self.figure_size = (17, 19)  # R's default size

    def prepare_plot_data(
        self, channels: List[str], dt_optimout: pd.DataFrame, objective_function: ObjectiveFunction
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepares data for response curves plotting."""
        plot_data = {}

        print("\nPreparing plot data with available columns:", dt_optimout.columns)

        for channel in channels:
            print(f"\nProcessing channel: {channel}")
            # Get channel data
            channel_data = dt_optimout[dt_optimout["channels"] == channel].iloc[0]

            # Get current and optimal spend
            current_spend = channel_data["initSpendUnit"]
            optimal_spend = channel_data["optmSpendUnit"]

            print(f"Current spend: {current_spend}")
            print(f"Optimal spend: {optimal_spend}")

            # Create spend range from 0 to 1.5x max spend
            max_spend = max(current_spend, optimal_spend) * 1.5
            spend_range = np.linspace(0, max_spend, 100)

            # Calculate responses
            responses = []
            for spend in spend_range:
                response = objective_function.calculate_response(
                    x=np.array([spend]), channel_name=channel, get_sum=True
                )
                responses.append(response)

            plot_data[channel] = {"spend": spend_range, "response": np.array(responses)}

        return plot_data

    def create_onepager(
        self,
        dt_optimout: pd.DataFrame,
        plot_data: Dict[str, Dict[str, np.ndarray]],
        scenario: str,
        model_id: str,
        errors: Optional[str] = None,
    ) -> Figure:
        """Creates one-pager plot matching R implementation."""
        # Create figure and layout
        fig = plt.figure(figsize=self.figure_size)
        gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.3)

        # Create subplots
        ax_top = fig.add_subplot(gs[0, :])  # Budget optimization
        ax_mid = fig.add_subplot(gs[1:3, :])  # Allocation heatmap
        ax_bot = fig.add_subplot(gs[3:, :])  # Response curves

        # Plot each section
        self._plot_budget_optimization(ax_top, dt_optimout)
        self._plot_allocation_heatmap(ax_mid, dt_optimout)
        self._plot_response_curves(ax_bot, plot_data, dt_optimout)

        # Add title and metadata
        title = f"Budget Allocation Onepager for Model ID {model_id}"
        if errors:
            title += f"\n{errors}"
        fig.suptitle(title, fontsize=14, y=0.95)

        return fig

    def _plot_budget_optimization(self, ax: Axes, dt_optimout: pd.DataFrame) -> None:
        """Plots budget optimization showing Initial and Bounded scenarios."""
        # Prepare data
        scenarios = ["Initial", "Bounded"]
        metrics = ["Total Spend", "Total Response"]

        data = pd.DataFrame(
            {
                "Scenario": scenarios * 2,
                "Metric": ["Total Spend"] * 2 + ["Total Response"] * 2,
                "Value": [
                    dt_optimout["initSpendUnit"].sum(),
                    dt_optimout["optmSpendUnit"].sum(),
                    dt_optimout["initResponseUnit"].sum(),
                    dt_optimout["optmResponseUnit"].sum(),
                ],
            }
        )

        # Create grouped bar plot
        sns.barplot(data=data, x="Metric", y="Value", hue="Scenario", palette=self.colors, ax=ax)

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="{:,.0f}")

        ax.set_title("Total Budget Optimization Result")

    def _plot_allocation_heatmap(self, ax: Axes, dt_optimout: pd.DataFrame) -> None:
        """Plots allocation heatmap with all metrics."""
        channels = dt_optimout["channels"].unique()
        metrics = ["abs.mean\nspend", "mean\nspend %", "mean\nresponse %", "mean\nROAS"]
        scenarios = ["Initial", "Bounded"]

        # Prepare data matrix
        data = []
        for channel in channels:
            row = []
            channel_data = dt_optimout[dt_optimout["channels"] == channel].iloc[0]

            # Get metrics for each scenario
            for scenario in scenarios:
                if scenario == "Initial":
                    row.extend(
                        [
                            channel_data["initSpendUnit"],
                            channel_data["initSpendShare"] * 100,
                            channel_data["initResponseUnit"] / channel_data["initResponseUnitTotal"] * 100,
                            channel_data["initRoiUnit"],
                        ]
                    )
                else:
                    row.extend(
                        [
                            channel_data["optmSpendUnit"],
                            channel_data["optmSpendShareUnit"] * 100,
                            channel_data["optmResponseUnit"] / channel_data["optmResponseUnitTotal"] * 100,
                            channel_data["optmRoiUnit"],
                        ]
                    )
            data.append(row)

        # Create heatmap
        hm = sns.heatmap(
            data=np.array(data),
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
            ax=ax,
            xticklabels=metrics * 2,
            yticklabels=channels,
        )

        # Add scenario dividers and labels
        n_metrics = len(metrics)
        for i, scenario in enumerate(scenarios):
            ax.axvline(x=n_metrics * (i + 1), color="white", lw=2)
            ax.text(n_metrics * i + n_metrics / 2, -0.5, scenario, ha="center", va="center")

        ax.set_title("Budget Allocation per Paid Media Variable per Week")

    def _plot_response_curves(self, ax: Axes, plot_data: Dict, dt_optimout: pd.DataFrame) -> None:
        """Plots response curves with points for Initial and Bounded scenarios."""
        for channel, data in plot_data.items():
            # Plot response curve
            ax.plot(data["spend"], data["response"], label=channel)

            # Get channel data
            channel_data = dt_optimout[dt_optimout["channels"] == channel].iloc[0]

            # Plot points for each scenario
            scenarios = [
                ("Initial", "initSpendUnit", "initResponseUnit", "o", "grey"),
                ("Bounded", "optmSpendUnit", "optmResponseUnit", "s", "steelblue"),
            ]

            for label, spend_col, response_col, marker, color in scenarios:
                spend = channel_data[spend_col]
                response = channel_data[response_col]

                ax.scatter(spend, response, marker=marker, s=100, c=color, label=f"{channel} ({label})")

                # Add spend label
                ax.annotate(f"${spend:,.0f}", (spend, response), xytext=(5, 5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("Spend")
        ax.set_ylabel("Response")
        ax.set_title("Simulated Response Curves per Week")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    def save_plot(self, fig: Figure, filename: str, dpi: int = 300) -> None:
        """Saves plot to file."""
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
