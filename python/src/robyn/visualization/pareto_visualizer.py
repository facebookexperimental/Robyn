from pathlib import Path
from typing import Dict, List, Optional, Union
from matplotlib import ticker, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from robyn.data.entities.enums import ProphetVariableType
from robyn.data.entities.holidays_data import HolidaysData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.data.entities.hyperparameters import AdstockType
from robyn.data.entities.mmmdata import MMMData
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class ParetoVisualizer(BaseVisualizer):
    def __init__(
        self,
        pareto_result: ParetoResult,
        adstock: AdstockType,
        mmm_data: MMMData,
        holiday_data: Optional[HolidaysData] = None,
    ):
        super().__init__()
        self.pareto_result = pareto_result
        self.adstock = adstock
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data

    def _baseline_vars(
        self, baseline_level, prophet_vars: List[ProphetVariableType] = []
    ) -> list:
        """
        Returns a list of baseline variables based on the provided level.
        Args:
            InputCollect (dict): A dictionary containing various input data.
            baseline_level (int): The level of baseline variables to include.
        Returns:
            list: A list of baseline variable names.
        """
        # Check if baseline_level is valid
        if baseline_level < 0 or baseline_level > 5:
            raise ValueError("baseline_level must be an integer between 0 and 5")
        baseline_variables = []
        # Level 1: Include intercept variables
        if baseline_level >= 1:
            baseline_variables.extend(["(Intercept)", "intercept"])
        # Level 2: Include trend variables
        if baseline_level >= 2:
            baseline_variables.append("trend")
        # Level 3: Include prophet variables
        if baseline_level >= 3:
            baseline_variables.extend(list(set(baseline_variables + prophet_vars)))
        # Level 4: Include context variables
        if baseline_level >= 4:
            baseline_variables.extend(self.mmm_data.mmmdata_spec.context_vars)
        # Level 5: Include organic variables
        if baseline_level >= 5:
            baseline_variables.extend(self.mmm_data.mmmdata_spec.organic_vars)
        return list(set(baseline_variables))

    def format_number(self, x: float, pos=None) -> str:
        """Format large numbers with K/M/B abbreviations.

        Args:
            x: Number to format
            pos: Position (required by matplotlib FuncFormatter but not used)

        Returns:
            Formatted string
        """
        if abs(x) >= 1e9:
            return f"{x/1e9:.1f}B"
        elif abs(x) >= 1e6:
            return f"{x/1e6:.1f}M"
        elif abs(x) >= 1e3:
            return f"{x/1e3:.1f}K"
        else:
            return f"{x:.1f}"

    def generate_waterfall(
        self, solution_id: str, ax: Optional[plt.Axes] = None, baseline_level: int = 0
    ) -> Optional[plt.Figure]:
        """Generate waterfall chart for specific solution."""

        logger.debug("Starting generation of waterfall plot")
        if solution_id not in self.pareto_result.plot_data_collect:
            raise ValueError(f"Invalid solution ID: {solution_id}")

        # Get data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        waterfall_data = plot_data["plot2data"]["plotWaterfallLoop"].copy()

        # Get baseline variables
        prophet_vars = self.holiday_data.prophet_vars if self.holiday_data else []
        baseline_vars = self._baseline_vars(baseline_level, prophet_vars)

        # Transform baseline variables
        waterfall_data["rn"] = np.where(
            waterfall_data["rn"].isin(baseline_vars),
            f"Baseline_L{baseline_level}",
            waterfall_data["rn"],
        )

        # Group and summarize
        waterfall_data = (
            waterfall_data.groupby("rn", as_index=False)
            .agg({"xDecompAgg": "sum", "xDecompPerc": "sum"})
            .reset_index()
        )

        # Sort by percentage contribution
        waterfall_data = waterfall_data.sort_values("xDecompPerc", ascending=True)

        # Calculate waterfall positions
        waterfall_data["end"] = 1 - waterfall_data["xDecompPerc"].cumsum()
        waterfall_data["start"] = waterfall_data["end"].shift(1)
        waterfall_data["start"] = waterfall_data["start"].fillna(1)
        waterfall_data["sign"] = np.where(
            waterfall_data["xDecompPerc"] >= 0, "Positive", "Negative"
        )

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = None

        # Define colors
        colors = {"Positive": "#59B3D2", "Negative": "#E5586E"}

        # Create categorical y-axis
        y_pos = range(len(waterfall_data))

        # Create horizontal bars
        bars = ax.barh(
            y=y_pos,
            width=waterfall_data["start"] - waterfall_data["end"],
            left=waterfall_data["end"],
            color=[colors[sign] for sign in waterfall_data["sign"]],
            height=0.6,
        )

        # Add text labels
        for i, row in waterfall_data.iterrows():
            # Format label text
            if abs(row["xDecompAgg"]) >= 1e9:
                formatted_num = f"{row['xDecompAgg']/1e9:.1f}B"
            elif abs(row["xDecompAgg"]) >= 1e6:
                formatted_num = f"{row['xDecompAgg']/1e6:.1f}M"
            elif abs(row["xDecompAgg"]) >= 1e3:
                formatted_num = f"{row['xDecompAgg']/1e3:.1f}K"
            else:
                formatted_num = f"{row['xDecompAgg']:.1f}"

            # Calculate x-position as the right edge of each positive bar
            x_pos = max(row["start"], row["end"])

            # Add label aligned at the end of the bar
            ax.text(
                x_pos - 0.01,
                i,  # Small offset from bar end
                f"{formatted_num}\n{row['xDecompPerc']*100:.1f}%",
                ha="right",
                va="center",
                fontsize=9,
                linespacing=0.9,
            )

        # Set y-ticks and labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(waterfall_data["rn"])

        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0%}".format(x)))
        ax.set_xticks(np.arange(0, 1.1, 0.2))

        # Set plot limits
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(waterfall_data) - 0.5)

        # Add legend at top
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=colors["Positive"], label="Positive"),
            Patch(facecolor=colors["Negative"], label="Negative"),
        ]

        # Create legend with white background
        legend = ax.legend(
            handles=legend_elements,
            title="Sign",
            loc="upper left",
            bbox_to_anchor=(0, 1.15),
            ncol=2,
            frameon=True,
            framealpha=1.0,
        )

        # Set title
        ax.set_title("Response Decomposition Waterfall", pad=30, x=0.5, y=1.05)

        # Label axes
        ax.set_xlabel("Contribution")
        ax.set_ylabel(None)

        # Customize grid
        ax.grid(True, axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        logger.debug("Successfully generated of waterfall plot")
        # Adjust layout
        if fig:
            plt.subplots_adjust(right=0.85, top=0.85)
            return fig

        return None

    def generate_fitted_vs_actual(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate time series plot comparing fitted vs actual values.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object
        """

        logger.debug("Starting generation of fitted vs actual plot")

        if solution_id not in self.pareto_result.plot_data_collect:
            raise ValueError(f"Invalid solution ID: {solution_id}")

        # Get data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        ts_data = plot_data["plot5data"]["xDecompVecPlotMelted"].copy()

        # Convert dates and format variables
        ts_data["ds"] = pd.to_datetime(ts_data["ds"])
        ts_data["linetype"] = np.where(
            ts_data["variable"] == "predicted", "solid", "dotted"
        )
        ts_data["variable"] = ts_data["variable"].str.title()

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = None

        # Plot lines with different styles for predicted vs actual
        for var in ts_data["variable"].unique():
            var_data = ts_data[ts_data["variable"] == var]
            linestyle = "solid" if var_data["linetype"].iloc[0] == "solid" else "dotted"
            ax.plot(
                var_data["ds"],
                var_data["value"],
                label=var,
                linestyle=linestyle,
                linewidth=0.6,
            )

        # Format y-axis with abbreviations
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))

        # Add training/validation/test splits if validation is enabled
        if hasattr(self.pareto_result, "train_size"):
            train_size = self.pareto_result.train_size

            # Calculate split points
            days = sorted(ts_data["ds"].unique())
            ndays = len(days)
            train_cut = round(ndays * train_size)
            val_cut = train_cut + round(ndays * (1 - train_size) / 2)

            # Add vertical lines and labels for splits
            splits = [
                (train_cut, f"Train: {train_size*100:.1f}%"),
                (val_cut, f"Validation: {((1-train_size)/2)*100:.1f}%"),
                (ndays - 1, f"Test: {((1-train_size)/2)*100:.1f}%"),
            ]

            for idx, (cut, label) in enumerate(splits):
                # Add vertical line
                ax.axvline(x=days[cut], color="#39638b", alpha=0.8, linestyle="-")

                # Add rotated text label
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(
                    days[cut],
                    1.02,
                    label,
                    rotation=270,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    transform=trans,
                    color="#39638b",
                    alpha=0.5,
                    fontsize=8,
                )

        # Customize plot
        ax.set_title("Actual vs. Predicted Response")
        ax.set_xlabel("Date")
        ax.set_ylabel("Response")

        # Move legend to top
        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0,
        )

        # Grid styling
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

        # Use white background
        ax.set_facecolor("white")

        logger.debug("Successfully generated of fitted vs casual plot")
        if fig:
            plt.tight_layout()
            return fig
        return None

    def generate_diagnostic_plot(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate diagnostic scatter plot of fitted vs residual values.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object
        """

        logger.debug("Starting generation of diagnostic plot")

        if solution_id not in self.pareto_result.plot_data_collect:
            raise ValueError(f"Invalid solution ID: {solution_id}")

        # Get data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        diag_data = plot_data["plot6data"]["xDecompVecPlot"].copy()

        # Calculate residuals
        diag_data["residuals"] = diag_data["actual"] - diag_data["predicted"]

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = None

        # Create scatter plot
        ax.scatter(
            diag_data["predicted"], diag_data["residuals"], alpha=0.5, color="steelblue"
        )

        # Add horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        # Add smoothed line with confidence interval
        from scipy.stats import gaussian_kde

        x_smooth = np.linspace(
            diag_data["predicted"].min(), diag_data["predicted"].max(), 100
        )

        # Fit LOWESS
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(diag_data["residuals"], diag_data["predicted"], frac=0.2)

        # Plot smoothed line
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2, alpha=0.8)

        # Calculate confidence intervals (using standard error bands)
        residual_std = np.std(diag_data["residuals"])
        ax.fill_between(
            smoothed[:, 0],
            smoothed[:, 1] - 2 * residual_std,
            smoothed[:, 1] + 2 * residual_std,
            color="red",
            alpha=0.1,
        )

        # Format axes with abbreviations
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))

        # Set labels and title
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title("Fitted vs. Residual")

        # Customize grid
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

        # Use white background
        ax.set_facecolor("white")

        logger.debug("Successfully generated of diagnostic plot")

        if fig:
            plt.tight_layout()
            return fig
        return None

    def generate_immediate_vs_carryover(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate stacked bar chart comparing immediate vs carryover effects.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            plt.Figure if ax is None, else None
        """

        logger.debug("Starting generation of immediate vs carryover plot")

        if solution_id not in self.pareto_result.plot_data_collect:
            raise ValueError(f"Invalid solution ID: {solution_id}")

        # Get data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        df_imme_caov = plot_data["plot7data"].copy()

        # Set up type factor levels
        df_imme_caov["type"] = pd.Categorical(
            df_imme_caov["type"], categories=["Immediate", "Carryover"], ordered=True
        )

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = None

        # Define colors
        colors = {"Immediate": "#59B3D2", "Carryover": "coral"}

        # Create stacked bar chart
        bottom = np.zeros(len(df_imme_caov["rn"].unique()))
        y_pos = range(len(df_imme_caov["rn"].unique()))

        # Get unique channel names and types
        channels = df_imme_caov["rn"].unique()
        types = ["Immediate", "Carryover"]

        # Create bar chart with labels
        for type_name in types:
            type_data = df_imme_caov[df_imme_caov["type"] == type_name]
            percentages = type_data["percentage"].values

            # Create bars
            bars = ax.barh(
                y_pos,
                percentages,
                left=bottom,
                height=0.5,
                label=type_name,
                color=colors[type_name],
            )

            # Add text labels in center of bars
            for i, (rect, percentage) in enumerate(zip(bars, percentages)):
                width = rect.get_width()
                x_pos = bottom[i] + width / 2
                ax.text(x_pos, i, f"{percentage*100:.0f}%", ha="center", va="center")

            bottom += percentages

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels)

        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
        ax.set_xlim(0, 1)

        # Add legend at top
        ax.legend(
            title=None,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0,
        )

        # Add labels and title
        ax.set_xlabel("% Response")
        ax.set_ylabel(None)
        ax.set_title("Immediate vs. Carryover Response Percentage")

        # Grid customization
        ax.grid(True, axis="x", alpha=0.2)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)

        # Use white background
        ax.set_facecolor("white")

        logger.debug("Successfully generated of immediate vs carryover plot")

        if fig:
            plt.tight_layout()
            return fig
        return None

    def generate_adstock_rate(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate adstock rate visualization based on adstock type.

        Args:
            solution_id: ID of solution to visualize
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated figure if ax is None, otherwise None
        """

        logger.debug("Starting generation of adstock plot")

        # Get the plot data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        adstock_data = plot_data["plot3data"]

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = None

        # Handle different adstock types
        if self.adstock == AdstockType.GEOMETRIC:
            # Get geometric adstock data
            dt_geometric = adstock_data["dt_geometric"].copy()

            # Create bar chart
            bars = ax.barh(
                y=range(len(dt_geometric)),
                width=dt_geometric["thetas"],
                height=0.5,
                color="coral",
            )

            # Add percentage labels
            for i, theta in enumerate(dt_geometric["thetas"]):
                ax.text(
                    theta + 0.01, i, f"{theta*100:.1f}%", va="center", fontweight="bold"
                )

            # Customize axes
            ax.set_yticks(range(len(dt_geometric)))
            ax.set_yticklabels(dt_geometric["channels"])

            # Format x-axis as percentage
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%")
            )
            ax.set_xlim(0, 1)

            # Set title and labels
            interval_type = (
                self.mmm_data.mmmdata_spec.interval_type if self.mmm_data else "day"
            )
            ax.set_title(
                f"Geometric Adstock: Fixed Rate Over Time (Solution {solution_id})"
            )
            ax.set_xlabel(f"Thetas [by {interval_type}]")
            ax.set_ylabel(None)

        elif self.adstock in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            # Get Weibull data
            weibull_data = adstock_data["weibullCollect"]
            wb_type = adstock_data["wb_type"]

            # Get unique channels for subplots
            channels = weibull_data["channel"].unique()
            rows = (len(channels) + 2) // 3  # 3 columns

            if ax is None:
                # Create new figure with subplots
                fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
                axes = axes.flatten()
            else:
                # Create subplot grid within provided axis
                gs = ax.get_gridspec()
                subfigs = ax.figure.subfigures(rows, 3)
                axes = [subfig.subplots() for subfig in subfigs]
                axes = [ax for sublist in axes for ax in sublist]  # flatten

            # Plot each channel
            for idx, channel in enumerate(channels):
                ax_sub = axes[idx]
                channel_data = weibull_data[weibull_data["channel"] == channel]

                # Plot decay curve
                ax_sub.plot(
                    channel_data["x"],
                    channel_data["decay_accumulated"],
                    color="steelblue",
                )

                # Add halflife line
                ax_sub.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
                ax_sub.text(
                    max(channel_data["x"]),
                    0.5,
                    "Halflife",
                    color="gray",
                    va="bottom",
                    ha="right",
                )

                # Customize subplot
                ax_sub.set_title(channel)
                ax_sub.grid(True, alpha=0.2)
                ax_sub.set_ylim(0, 1)

        # Customize grid
        if self.adstock == AdstockType.GEOMETRIC:
            ax.grid(True, axis="x", alpha=0.2)
            ax.grid(False, axis="y")
        ax.set_axisbelow(True)

        # Use white background
        ax.set_facecolor("white")

        logger.debug("Successfully generated of adstock plot")

        if fig:
            plt.tight_layout()
            return fig
        return None

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> None:
        # Generate all plots
        solution_ids = self.pareto_result.pareto_solutions
        figures: Dict[str, plt.Figure] = {}

        for solution_id in solution_ids:
            fig1 = self.generate_waterfall(solution_id)
            if fig1:
                figures["waterfall_" + solution_id] = fig1

            fig2 = self.generate_fitted_vs_actual(solution_id)
            if fig2:
                figures["fitted_vs_actual_" + solution_id] = fig2

            fig3 = self.generate_diagnostic_plot(solution_id)
            if fig3:
                figures["diagnostic_plot_" + solution_id] = fig3

            fig4 = self.generate_immediate_vs_carryover(solution_id)
            if fig4:
                figures["immediate_vs_carryover_" + solution_id] = fig4

            fig5 = self.generate_adstock_rate(solution_id)
            if fig5:
                figures["adstock_rate_" + solution_id] = fig5

            break  # TODO: This will generate too many plots. Only generate plots for the first solution. we can export all plots to a folder if too many to display

        # Display plots if required
        if display_plots:
            self.display_plots(figures)
