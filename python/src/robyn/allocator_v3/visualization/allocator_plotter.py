"""Standalone plotting module for Robyn allocator results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class PlotConfig:
    """Configuration for allocator plots."""

    title: str
    metric: str  # 'ROAS' or 'CPA'
    interval_type: str
    model_id: str
    errors: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    scenario: Optional[str] = None


class AllocatorPlotter:
    """Generates plots for Robyn allocator results."""

    def __init__(self):
        """Initialize with default styling."""
        self.colors = {
            "dark_blue": "#03396c",
            "steel_blue": "#59B3D2",
            "grey": "#808080",
            "coral": "#E5586E",
        }
        self._set_style()

    def _set_style(self):
        """Set matplotlib style for consistent plotting."""
        # plt.style.use("seaborn-whitegrid")
        plt.rcParams.update(
            {
                "figure.figsize": [12, 8],
                "axes.grid": True,
                "axes.grid.axis": "y",
                "grid.alpha": 0.3,
                "grid.color": "#cccccc",
            }
        )

    def create_plots(
        self,
        allocation_result: Any,
        config: PlotConfig,
        export: bool = True,
        plot_folder: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Create complete set of allocator plots.

        Args:
            allocation_result: Allocation optimization results
            config: Plot configuration
            export: Whether to save plots
            plot_folder: Directory for saving plots

        Returns:
            Dictionary of plot figures
        """
        plots = {}

        # 1. Budget optimization comparison
        plots["budget_opt"] = self._plot_budget_comparison(
            allocation_result.dt_optimOut, config
        )

        # 2. Channel allocation matrix
        plots["allocation"] = self._plot_allocation_matrix(
            allocation_result.dt_optimOut, config
        )

        # 3. Response curves
        plots["response"] = self._plot_response_curves(
            allocation_result.dt_optimOut, allocation_result.mainPoints, config
        )

        # 4. Combined onepager
        plots["onepager"] = self._create_onepager(plots, config, allocation_result)

        if export and plot_folder:
            for name, fig in plots.items():
                fig.savefig(
                    f"{plot_folder}/{config.model_id}_{name}.png",
                    dpi=400,
                    bbox_inches="tight",
                )

        return plots

    def _plot_budget_comparison(
        self, dt_optimOut: "OptimOutData", config: PlotConfig
    ) -> plt.Figure:
        """Plot budget comparison with raw values and summary stats."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate totals and metrics for each scenario
        scenarios = {
            "Initial": {
                "spend": dt_optimOut.init_spend_unit.sum(),
                "response": dt_optimOut.init_response_unit.sum(),
            },
            "Bounded": {
                "spend": dt_optimOut.optm_spend_unit.sum(),
                "response": dt_optimOut.optm_response_unit.sum(),
            },
            "Bounded x3": {
                "spend": dt_optimOut.optm_spend_unit_unbound.sum(),
                "response": dt_optimOut.optm_response_unit_unbound.sum(),
            },
        }

        # Calculate summary metrics
        for scenario in scenarios:
            scenarios[scenario].update(
                {
                    "spend_pct_change": (
                        scenarios[scenario]["spend"] / scenarios["Initial"]["spend"] - 1
                    )
                    * 100,
                    "response_pct_change": (
                        scenarios[scenario]["response"]
                        / scenarios["Initial"]["response"]
                        - 1
                    )
                    * 100,
                    "roas": scenarios[scenario]["response"]
                    / scenarios[scenario]["spend"],
                }
            )

        # Create bar plot
        x = np.arange(len(scenarios))
        width = 0.35

        # Plot raw values
        spend_bars = ax.bar(
            x - width / 2,
            [d["spend"] for d in scenarios.values()],
            width,
            label="Total Spend",
            color=self.colors["steel_blue"],
        )

        response_bars = ax.bar(
            x + width / 2,
            [d["response"] for d in scenarios.values()],
            width,
            label="Total Response",
            color=self.colors["dark_blue"],
        )

        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    f"{height:,.0f}",
                    ha="center",
                    va="bottom",
                )

        autolabel(spend_bars)
        autolabel(response_bars)

        # Add summary text below each scenario
        for i, (scenario, data) in enumerate(scenarios.items()):
            summary_text = (
                f"Spend: {data['spend_pct_change']:+.1f}%\n"
                f"Resp: {data['response_pct_change']:+.1f}%\n"
                f"ROAS: {data['roas']:.2f}"
            )
            ax.text(
                i,
                -0.15 * ax.get_ylim()[1],  # Position text below bars
                summary_text,
                ha="center",
                va="top",
                multialignment="left",
            )

        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios.keys())
        ax.legend()

        plt.title(
            f"Total Budget Optimization Result (scaled up to {dt_optimOut.periods})\n"
            f"{config.metric} by Channel"
        )

        # Adjust layout to make room for summary text
        plt.subplots_adjust(bottom=0.2)

        return fig

    def _plot_allocation_matrix(
        self, dt_optimOut: "OptimOutData", config: PlotConfig
    ) -> plt.Figure:
        """Plot allocation matrix matching R format."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))

        scenarios = ["Initial", "Bounded", "Bounded x3"]
        metrics = [
            "abs.mean\nspend",
            "mean\nspend%",
            "mean\nresponse%",
            f"mean\n{config.metric}",
            f"m{config.metric}",
        ]

        for i, scenario in enumerate(scenarios):
            data = self._get_allocation_data(dt_optimOut, scenario)

            # Format data for heatmap
            plot_data = data.copy()
            plot_data["abs.mean\nspend"] = plot_data["abs.mean\nspend"].apply(
                lambda x: f"{x:,.0f}"
            )
            for col in metrics[1:]:  # Format percentages and ratios
                plot_data[col] = plot_data[col].apply(
                    lambda x: f"{x*100:.1f}%" if col.endswith("%") else f"{x:.2f}"
                )

            # Create heatmap
            sns.heatmap(
                data[metrics],
                ax=axes[i],
                cmap="Blues",
                annot=plot_data[metrics].values,
                fmt="",
                cbar=i == 2,
            )

            axes[i].set_title(scenario)
            axes[i].set_xlabel("Metric")
            axes[i].set_ylabel("Channel" if i == 0 else "")

        plt.suptitle(
            f"Budget Allocation per Paid Media Variable per {config.interval_type}\n"
            f"Period: {dt_optimOut.date_min} to {dt_optimOut.date_max}"
        )
        plt.tight_layout()

        return fig

    def _plot_response_curves(
        self, dt_optimOut: Any, mainPoints: Any, config: PlotConfig
    ) -> plt.Figure:
        """Plot response curves with optimization points."""
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, channel in enumerate(dt_optimOut.channels):
            # Generate response curve
            spend_range = np.linspace(0, dt_optimOut.init_spend_unit[i] * 1.5, 100)
            response = self._calculate_response_curve(spend_range, channel, dt_optimOut)

            # Plot curve
            ax.plot(spend_range, response, label=channel, alpha=0.7)

            # Add points
            points = mainPoints[mainPoints.channel == channel]
            ax.scatter(points.spend_point, points.response_point, marker="o", s=100)

            # Add constraint lines
            lower = dt_optimOut.init_spend_unit[i] * dt_optimOut.constr_low[i]
            upper = dt_optimOut.init_spend_unit[i] * dt_optimOut.constr_up[i]
            ax.axvline(lower, color="gray", linestyle="--", alpha=0.3)
            ax.axvline(upper, color="gray", linestyle="--", alpha=0.3)

        ax.set_xlabel("Spend")
        ax.set_ylabel("Response")
        ax.set_title(f"Simulated Response Curve per {config.interval_type}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        return fig

    def _create_onepager(
        self, plots: Dict[str, plt.Figure], config: PlotConfig, allocation_result: Any
    ) -> plt.Figure:
        """Create combined onepager plot."""
        fig = plt.figure(figsize=(17, 19))
        gs = plt.GridSpec(4, 2, figure=fig)

        # Add plots to grid
        for i, (name, source_fig) in enumerate(
            [
                ("budget_opt", plots["budget_opt"]),
                ("allocation", plots["allocation"]),
                ("response", plots["response"]),
            ]
        ):
            ax = fig.add_subplot(gs[i, :])
            source_fig.canvas.draw()
            ax.imshow(source_fig.canvas.renderer.buffer_rgba())
            ax.axis("off")

        # Add title and metadata
        fig.suptitle(f"One-pager for Model ID: {config.model_id}", fontsize=14, y=0.95)

        # Add performance metrics
        performance_text = (
            f"Window: {config.window_start} to {config.window_end}\n"
            f"Scenario: {config.scenario}\n"
            f"{config.errors if config.errors else ''}"
        )
        fig.text(0.1, 0.92, performance_text, fontsize=10)

        plt.tight_layout()
        return fig

    def _get_allocation_data(
        self, dt_optimOut: "OptimOutData", scenario: str
    ) -> pd.DataFrame:
        """Prepare data for allocation matrix plot matching R format."""
        # Initialize base spend and response values
        if scenario == "Initial":
            data = {
                "abs.mean\nspend": dt_optimOut.init_spend_unit,
                "mean\nspend%": dt_optimOut.init_spend_unit
                / dt_optimOut.init_spend_unit.sum(),
                "mean\nresponse%": dt_optimOut.init_response_unit
                / dt_optimOut.init_response_unit.sum(),
                f"mean\n{dt_optimOut.metric}": dt_optimOut.init_response_unit
                / dt_optimOut.init_spend_unit,
            }
        elif scenario == "Bounded":
            data = {
                "abs.mean\nspend": dt_optimOut.optm_spend_unit,
                "mean\nspend%": dt_optimOut.optm_spend_unit
                / dt_optimOut.optm_spend_unit.sum(),
                "mean\nresponse%": dt_optimOut.optm_response_unit
                / dt_optimOut.optm_response_unit.sum(),
                f"mean\n{dt_optimOut.metric}": dt_optimOut.optm_response_unit
                / dt_optimOut.optm_spend_unit,
            }
        else:  # "Bounded x3"
            data = {
                "abs.mean\nspend": dt_optimOut.optm_spend_unit_unbound,
                "mean\nspend%": dt_optimOut.optm_spend_unit_unbound
                / dt_optimOut.optm_spend_unit_unbound.sum(),
                "mean\nresponse%": dt_optimOut.optm_response_unit_unbound
                / dt_optimOut.optm_response_unit_unbound.sum(),
                f"mean\n{dt_optimOut.metric}": dt_optimOut.optm_response_unit_unbound
                / dt_optimOut.optm_spend_unit_unbound,
            }

        # Calculate marginal metrics
        spend = data["abs.mean\nspend"]
        response = data[f"mean\n{dt_optimOut.metric}"] * spend
        marginal_roi = np.gradient(response) / np.gradient(spend)
        data[f"m{dt_optimOut.metric}"] = marginal_roi

        # Create DataFrame with data
        return pd.DataFrame(data, index=dt_optimOut.channels)

    @staticmethod
    def _calculate_response_curve(
        spend_range: np.ndarray, channel: str, dt_optimOut: Any
    ) -> np.ndarray:
        """Calculate response values for a spend range."""
        # This is a simplified response calculation - replace with actual response function
        return (
            spend_range
            * dt_optimOut.init_response_unit[dt_optimOut.channels == channel][0]
            / dt_optimOut.init_spend_unit[dt_optimOut.channels == channel][0]
        )
