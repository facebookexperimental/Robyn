# allocation_plotter.py

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from robyn.new_allocator.optimization.objective_function import ObjectiveFunction


class AllocationPlotter:
    """Creates visualization plots for budget allocation results."""

    def __init__(self, dark_mode: bool = False):
        self.dark_mode = dark_mode
        self._set_style()

    def _set_style(self):
        """Sets plotting style."""
        self.colors = {
            "initial": "grey",
            "bounded": "steelblue",
            "boundedx3": "darkgoldenrod4",
            "positive": "#59B3D2",
            "negative": "#E5586E",
        }

    def create_onepager(
        self,
        dt_optim_out: pd.DataFrame,
        plot_data: Dict[str, Dict[str, np.ndarray]],
        scenario: str,
        date_range: Tuple[str, str],
        interval_type: str = "Week",
        figsize: Tuple[int, int] = (17, 19),
    ) -> plt.Figure:
        """Creates allocation one-pager plot matching R implementation."""
        # Get response type (ROAS/CPA)
        metric_type = "ROAS" if dt_optim_out["dep_var_type"].iloc[0] == "revenue" else "CPA"

        # Calculate periods
        periods = dt_optim_out["periods"].iloc[0]

        # Create figure with layout matching R
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(7, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1], hspace=0.4)

        # 1. Budget Optimization (Top)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_budget_optimization(ax1, dt_optim_out)

        # 2. Budget Allocation Heatmap (Middle)
        ax2 = fig.add_subplot(gs[1:3, :])
        self._plot_allocation_heatmap(ax2, dt_optim_out, interval_type, metric_type)

        # 3. Response Curves (Bottom)
        ax3 = fig.add_subplot(gs[3:, :])
        self._plot_response_curves(ax3, plot_data, dt_optim_out, metric_type)

        # Add title and metadata
        title = f"Budget Allocation Onepager for Model ID {dt_optim_out['solID'].iloc[0]}"
        subtitle = f"Simulation date range: {date_range[0]} to {date_range[1]} " f"({periods}) | Scenario: {scenario}"
        fig.suptitle(f"{title}\n{subtitle}", fontsize=14, y=0.95)

        return fig

    def _plot_budget_optimization(self, ax: plt.Axes, dt_optim_out: pd.DataFrame) -> None:
        """Plot budget optimization results with three scenarios."""
        # Prepare data for the three scenarios
        scenarios = ["Initial", "Bounded", "Bounded x3"]
        spend_vals = [
            dt_optim_out["initSpendTotal"].sum(),
            dt_optim_out["optmSpendTotal"].sum(),
            dt_optim_out["optmSpendTotalUnbound"].sum(),
        ]
        response_vals = [
            dt_optim_out["initResponseTotal"].sum(),
            dt_optim_out["optmResponseTotal"].sum(),
            dt_optim_out["optmResponseTotalUnbound"].sum(),
        ]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Scenario": scenarios * 2,
                "Metric": ["Spend"] * 3 + ["Response"] * 3,
                "Value": spend_vals + response_vals,
            }
        )

        # Create grouped bar plot
        sns.barplot(
            data=df,
            x="Metric",
            y="Value",
            hue="Scenario",
            palette=[self.colors["initial"], self.colors["bounded"], self.colors["boundedx3"]],
            ax=ax,
        )

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=3)

        ax.set_title("Total Budget Optimization Result")
        ax.legend(title=None, loc="upper right")

    def _plot_allocation_heatmap(
        self, ax: plt.Axes, dt_optim_out: pd.DataFrame, interval_type: str, metric_type: str
    ) -> None:
        """Plot allocation heatmap with metrics."""
        # Prepare metrics for heatmap
        metrics = ["abs.mean\nspend", "mean\nspend %", "mean\nresponse %", f"mean\n{metric_type}", f"m{metric_type}"]

        channels = dt_optim_out["channels"].unique()
        scenarios = ["Initial", "Bounded", "Bounded x3"]

        # Create data matrix
        data = []
        for channel in channels:
            channel_data = dt_optim_out[dt_optim_out["channels"] == channel].iloc[0]
            row = []
            for scenario in scenarios:
                if scenario == "Initial":
                    row.extend(
                        [
                            channel_data["initSpendUnit"],
                            channel_data["initSpendShare"] * 100,
                            channel_data["initResponseUnit"] / channel_data["initResponseUnitTotal"] * 100,
                            channel_data["initRoiUnit"] if metric_type == "ROAS" else channel_data["initCpaUnit"],
                            channel_data["initResponseMargUnit"] if "initResponseMargUnit" in channel_data else 0,
                        ]
                    )
                else:
                    suffix = "Unbound" if scenario == "Bounded x3" else ""
                    row.extend(
                        [
                            channel_data[f"optmSpendUnit{suffix}"],
                            channel_data[f"optmSpendShareUnit{suffix}"] * 100,
                            channel_data[f"optmResponseUnit{suffix}"]
                            / channel_data[f"optmResponseUnitTotal{suffix}"]
                            * 100,
                            (
                                channel_data[f"optmRoiUnit{suffix}"]
                                if metric_type == "ROAS"
                                else channel_data[f"optmCpaUnit{suffix}"]
                            ),
                            (
                                channel_data[f"optmResponseMargUnit{suffix}"]
                                if f"optmResponseMargUnit{suffix}" in channel_data
                                else 0
                            ),
                        ]
                    )
            data.append(row)

        # Create heatmap
        sns.heatmap(
            data=np.array(data),
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax,
            xticklabels=metrics * 3,
            yticklabels=channels,
        )

        # Add scenario labels
        for i, scenario in enumerate(scenarios):
            ax.text(
                i * len(metrics) + len(metrics) / 2,
                -0.5,
                scenario,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(f"Budget Allocation per Paid Media Variable per {interval_type}")

    def _plot_response_curves(
        self, ax: plt.Axes, plot_data: Dict, dt_optim_out: pd.DataFrame, metric_type: str
    ) -> None:
        """Plot response curves with points."""
        for channel, data in plot_data.items():
            # Plot response curve
            ax.plot(data["spend"], data["response"], label=channel)

            # Get points for current and optimized spends
            channel_data = dt_optim_out[dt_optim_out["channels"] == channel].iloc[0]

            # Plot current point
            ax.scatter(
                channel_data["initSpendUnit"],
                channel_data["initResponseUnit"],
                marker="o",
                s=100,
                color=self.colors["initial"],
            )

            # Plot bounded optimized point
            ax.scatter(
                channel_data["optmSpendUnit"],
                channel_data["optmResponseUnit"],
                marker="s",
                s=100,
                color=self.colors["bounded"],
            )

            # Plot unbounded optimized point
            ax.scatter(
                channel_data["optmSpendUnitUnbound"],
                channel_data["optmResponseUnitUnbound"],
                marker="^",
                s=100,
                color=self.colors["boundedx3"],
            )

            # Add spend labels
            for spend, response, marker in [
                (channel_data["initSpendUnit"], channel_data["initResponseUnit"], "Initial"),
                (channel_data["optmSpendUnit"], channel_data["optmResponseUnit"], "Bounded"),
                (channel_data["optmSpendUnitUnbound"], channel_data["optmResponseUnitUnbound"], "Bounded x3"),
            ]:
                ax.annotate(f"${spend:,.0f}", (spend, response), xytext=(5, 5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("Spend")
        ax.set_ylabel("Response")
        ax.set_title(f"Simulated Response Curves per {interval_type}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """Saves plot to file."""
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
