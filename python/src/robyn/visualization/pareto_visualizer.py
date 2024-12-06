from pathlib import Path
import re
from typing import Dict, List, Optional, Union
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from robyn.modeling.entities.modeloutputs import ModelOutputs
import seaborn as sns
import logging
from robyn.data.entities.enums import ProphetVariableType
from robyn.data.entities.holidays_data import HolidaysData
from robyn.modeling.entities.featurized_mmm_data import FeaturizedMMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.data.entities.hyperparameters import AdstockType, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.visualization.base_visualizer import BaseVisualizer
from robyn.data.entities.enums import DependentVarType
import math
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class ParetoVisualizer(BaseVisualizer):
    def __init__(
        self,
        pareto_result: ParetoResult,
        mmm_data: MMMData,
        holiday_data: Optional[HolidaysData] = None,
        hyperparameter: Optional[Hyperparameters] = None,
        featurized_mmm_data: Optional[FeaturizedMMMData] = None,
        unfiltered_pareto_result: Optional[ParetoResult] = None,
        model_outputs: Optional[ModelOutputs] = None,
    ):
        super().__init__()
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.hyperparameter = hyperparameter
        self.featurized_mmm_data = featurized_mmm_data
        self.unfiltered_pareto_result = unfiltered_pareto_result
        self.model_outputs = model_outputs

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

        # Create categorical y-axis positions
        y_pos = np.arange(len(waterfall_data))

        # Create horizontal bars
        bars = ax.barh(
            y=y_pos,
            width=waterfall_data["start"] - waterfall_data["end"],
            left=waterfall_data["end"],
            color=[colors[sign] for sign in waterfall_data["sign"]],
            height=0.6,
        )

        # Add text labels
        for idx, row in enumerate(waterfall_data.itertuples()):
            # Format label text
            if abs(row.xDecompAgg) >= 1e9:
                formatted_num = f"{row.xDecompAgg/1e9:.1f}B"
            elif abs(row.xDecompAgg) >= 1e6:
                formatted_num = f"{row.xDecompAgg/1e6:.1f}M"
            elif abs(row.xDecompAgg) >= 1e3:
                formatted_num = f"{row.xDecompAgg/1e3:.1f}K"
            else:
                formatted_num = f"{row.xDecompAgg:.1f}"

            # Calculate x-position as the middle of the bar
            x_pos = (row.start + row.end) / 2

            # Use y_pos[idx] to ensure alignment with bars
            ax.text(
                x_pos,
                y_pos[idx],  # Use the same y-position as the corresponding bar
                f"{formatted_num}\n{row.xDecompPerc*100:.1f}%",
                ha="center",  # Center align horizontally
                va="center",  # Center align vertically
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

        logger.debug("Successfully generated waterfall plot")
        # Adjust layout
        if fig:
            plt.subplots_adjust(right=0.85, top=0.85)
            fig = plt.gcf()
            plt.close(fig)
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

        # Ensure ds column is datetime and remove any NaT values
        ts_data["ds"] = pd.to_datetime(ts_data["ds"])
        ts_data = ts_data.dropna(subset=["ds"])  # Remove rows with NaT dates

        if ts_data.empty:
            logger.warning(f"No valid date data found for solution {solution_id}")
            return None

        ts_data["linetype"] = np.where(
            ts_data["variable"] == "predicted", "solid", "dotted"
        )
        ts_data["variable"] = ts_data["variable"].str.title()

        # Get train_size from x_decomp_agg
        train_size_series = self.pareto_result.x_decomp_agg[
            self.pareto_result.x_decomp_agg["sol_id"] == solution_id
        ]["train_size"]

        if not train_size_series.empty:
            train_size = float(train_size_series.iloc[0])
        else:
            train_size = 0

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        else:
            fig = None

        colors = {
            "Actual": "#FF6B00",  # Darker orange
            "Predicted": "#0066CC",  # Darker blue
        }

        # Plot lines with different styles for predicted vs actual
        for var in ts_data["variable"].unique():
            var_data = ts_data[ts_data["variable"] == var]
            linestyle = "solid" if var_data["linetype"].iloc[0] == "solid" else "dotted"
            ax.plot(
                var_data["ds"],
                var_data["value"],
                label=var,
                linestyle=linestyle,
                linewidth=1,
                color=colors[var],
            )

        # Format y-axis with abbreviations
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))

        # Set y-axis limits with some padding
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.2)  # Add 20% padding at the top

        # Add training/validation/test splits if train_size exists and is valid
        if train_size > 0:
            try:
                # Get unique sorted dates, excluding NaT
                unique_dates = sorted(ts_data["ds"].dropna().unique())
                total_days = len(unique_dates)

                if total_days > 0:
                    # Calculate split points
                    train_cut = int(total_days * train_size)
                    val_cut = train_cut + int(total_days * (1 - train_size) / 2)

                    # Get dates for splits
                    splits = [
                        (train_cut, "Train", train_size),
                        (val_cut, "Validation", (1 - train_size) / 2),
                        (total_days - 1, "Test", (1 - train_size) / 2),
                    ]

                    # Get y-axis limits for text placement
                    y_min, y_max = ax.get_ylim()

                    # Add vertical lines and labels
                    for idx, label, size in splits:
                        if 0 <= idx < len(unique_dates):  # Ensure index is valid
                            date = unique_dates[idx]
                            if pd.notna(date):  # Check if date is valid
                                # Add vertical line - extend beyond the top of the plot
                                ax.axvline(
                                    date, color="#39638b", alpha=0.8, ymin=0, ymax=1.1
                                )

                                # Add rotated text label
                                ax.text(
                                    date,
                                    y_max,
                                    f"{label}: {size*100:.1f}%",
                                    rotation=270,
                                    color="#39638b",
                                    alpha=0.5,
                                    size=9,
                                    ha="left",
                                    va="top",
                                )
            except Exception as e:
                logger.warning(f"Error adding split lines: {str(e)}")
                # Continue with the rest of the plot even if split lines fail

        # Set title and labels
        ax.set_title("Actual vs. Predicted Response", pad=20)
        ax.set_xlabel("Date")
        ax.set_ylabel("Response")

        # Configure legend
        ax.legend(
            bbox_to_anchor=(0.01, 1.02),  # Position at top-left
            loc="lower left",
            ncol=2,  # Two columns side by side
            borderaxespad=0,
            frameon=False,
            fontsize=7,
            handlelength=2,  # Length of the legend lines
            handletextpad=0.5,  # Space between line and text
            columnspacing=1.0,  # Space between columns
        )

        # Grid styling
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")

        # Format dates on x-axis using datetime locator and formatter
        years = mdates.YearLocator()
        years_fmt = mdates.DateFormatter("%Y")
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)

        logger.debug("Successfully generated fitted vs actual plot")
        if fig:
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            fig = plt.gcf()
            plt.close(fig)
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
            fig, ax = plt.subplots(figsize=(16, 10))
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
            fig = plt.gcf()
            plt.close(fig)
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

        plot_data = self.pareto_result.plot_data_collect[solution_id]
        df_imme_caov = plot_data["plot7data"].copy()

        # Ensure percentage is numeric
        df_imme_caov["percentage"] = pd.to_numeric(
            df_imme_caov["percentage"], errors="coerce"
        )

        # Sort channels alphabetically
        df_imme_caov = df_imme_caov.sort_values("rn", ascending=True)

        # Set up type factor levels matching R plot order
        df_imme_caov["type"] = pd.Categorical(
            df_imme_caov["type"], categories=["Immediate", "Carryover"], ordered=True
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))
        else:
            fig = None

        colors = {"Immediate": "#59B3D2", "Carryover": "coral"}

        bottom = np.zeros(len(df_imme_caov["rn"].unique()))
        y_pos = range(len(df_imme_caov["rn"].unique()))
        channels = df_imme_caov["rn"].unique()
        types = ["Immediate", "Carryover"]  # Order changed to Immediate first

        # Normalize percentages to sum to 100% for each channel
        for channel in channels:
            mask = df_imme_caov["rn"] == channel
            total = df_imme_caov.loc[mask, "percentage"].sum()
            if total > 0:  # Avoid division by zero
                df_imme_caov.loc[mask, "percentage"] = (
                    df_imme_caov.loc[mask, "percentage"] / total
                )

        for type_name in types:
            type_data = df_imme_caov[df_imme_caov["type"] == type_name]
            percentages = type_data["percentage"].values

            bars = ax.barh(
                y_pos,
                percentages,
                left=bottom,
                height=0.5,
                label=type_name,
                color=colors[type_name],
            )

            for i, (rect, percentage) in enumerate(zip(bars, percentages)):
                width = rect.get_width()
                x_pos = bottom[i] + width / 2
                try:
                    percentage_text = f"{round(float(percentage) * 100)}%"
                except (ValueError, TypeError):
                    percentage_text = "0%"
                ax.text(x_pos, i, percentage_text, ha="center", va="center")

            bottom += percentages

        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
        ax.set_xlim(0, 1)

        # Reduced legend size
        ax.legend(
            title=None,
            bbox_to_anchor=(0, 1.02, 0.15, 0.1),  # Reduced width from 0.3 to 0.2
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0,
            frameon=False,
            fontsize=7,  # Reduced from 8 to 7
        )

        ax.set_xlabel("% Response")
        ax.set_ylabel(None)
        ax.set_title("Immediate vs. Carryover Response Percentage", pad=50, y=1.2)

        ax.grid(True, axis="x", alpha=0.2)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
        ax.set_facecolor("white")

        if fig:
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            fig = plt.gcf()
            plt.close(fig)
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

        plot_data = self.pareto_result.plot_data_collect[solution_id]
        adstock_data = plot_data["plot3data"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))
        else:
            fig = None

        # Handle different adstock types
        if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
            dt_geometric = adstock_data["dt_geometric"].copy()

            # Sort data alphabetically by channel
            dt_geometric = dt_geometric.sort_values("channels", ascending=True)

            bars = ax.barh(
                y=range(len(dt_geometric)),
                width=dt_geometric["thetas"],
                height=0.5,
                color="coral",
            )

            for i, theta in enumerate(dt_geometric["thetas"]):
                ax.text(
                    theta + 0.01, i, f"{theta*100:.1f}%", va="center", fontweight="bold"
                )

            ax.set_yticks(range(len(dt_geometric)))
            ax.set_yticklabels(dt_geometric["channels"])

            # Format x-axis with 25% increments
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%")
            )
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.25, 0.25))  # Changed to 0.25 increments

            # Set title and labels
            interval_type = (
                self.mmm_data.mmmdata_spec.interval_type if self.mmm_data else "day"
            )
            ax.set_title(
                f"Geometric Adstock: Fixed Rate Over Time (Solution {solution_id})"
            )
            ax.set_xlabel(f"Thetas [by {interval_type}]")
            ax.set_ylabel(None)

        elif self.hyperparameter.adstock in [
            AdstockType.WEIBULL_CDF,
            AdstockType.WEIBULL_PDF,
        ]:
            # [Weibull code remains the same]
            weibull_data = adstock_data["weibullCollect"]
            wb_type = adstock_data["wb_type"]

            channels = sorted(
                weibull_data["channel"].unique()
            )  # Sort channels alphabetically
            rows = (len(channels) + 2) // 3

            if ax is None:
                fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
                axes = axes.flatten()
            else:
                gs = ax.get_gridspec()
                subfigs = ax.figure.subfigures(rows, 3)
                axes = [subfig.subplots() for subfig in subfigs]
                axes = [ax for sublist in axes for ax in sublist]

            for idx, channel in enumerate(channels):
                ax_sub = axes[idx]
                channel_data = weibull_data[weibull_data["channel"] == channel]

                ax_sub.plot(
                    channel_data["x"],
                    channel_data["decay_accumulated"],
                    color="steelblue",
                )

                ax_sub.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
                ax_sub.text(
                    max(channel_data["x"]),
                    0.5,
                    "Halflife",
                    color="gray",
                    va="bottom",
                    ha="right",
                )

                ax_sub.set_title(channel)
                ax_sub.grid(True, alpha=0.2)
                ax_sub.set_ylim(0, 1)

        # Customize grid
        if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
            ax.grid(True, axis="x", alpha=0.2)
            ax.grid(False, axis="y")
        ax.set_axisbelow(True)

        ax.set_facecolor("white")

        logger.debug("Successfully generated adstock plot")

        if fig:
            plt.tight_layout()
            fig = plt.gcf()
            plt.close(fig)
            return fig
        return None

    def create_prophet_decomposition_plot(self):
        """Create Prophet Decomposition Plot."""
        prophet_vars = (
            [ProphetVariableType(var) for var in self.holiday_data.prophet_vars]
            if self.holiday_data and self.holiday_data.prophet_vars
            else []
        )
        factor_vars = self.mmm_data.mmmdata_spec.factor_vars if self.mmm_data else []
        if not (prophet_vars or factor_vars):
            return None
        df = self.featurized_mmm_data.dt_mod.copy()
        prophet_vars_str = [variable.value for variable in prophet_vars]
        prophet_vars_str.sort(reverse=True)
        value_variables = (
            [
                (
                    "dep_var"
                    if hasattr(df, "dep_var")
                    else self.mmm_data.mmmdata_spec.dep_var
                )
            ]
            + factor_vars
            + prophet_vars_str
        )
        df_long = df.melt(
            id_vars=["ds"],
            value_vars=value_variables,
            var_name="variable",
            value_name="value",
        )
        df_long["ds"] = pd.to_datetime(df_long["ds"])
        plt.figure(figsize=(12, 3 * len(df_long["variable"].unique())))
        prophet_decomp_plot = plt.figure(
            figsize=(12, 3 * len(df_long["variable"].unique()))
        )
        gs = prophet_decomp_plot.add_gridspec(len(df_long["variable"].unique()), 1)
        for i, var in enumerate(df_long["variable"].unique()):
            ax = prophet_decomp_plot.add_subplot(gs[i, 0])
            var_data = df_long[df_long["variable"] == var]
            ax.plot(var_data["ds"], var_data["value"], color="steelblue")
            ax.set_title(var)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        plt.suptitle("Prophet decomposition")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def create_hyperparameter_sampling_distribution(self):
        """Create Hyperparameter Sampling Distribution Plot."""
        unfiltered_pareto_results = self.unfiltered_pareto_result
        if unfiltered_pareto_results is None:
            return None
        result_hyp_param = unfiltered_pareto_results.result_hyp_param
        hp_names = list(self.hyperparameter.hyperparameters.keys())
        hp_names = [name.replace("lambda", "lambda_hp") for name in hp_names]
        matching_columns = [
            col
            for col in result_hyp_param.columns
            if any(re.search(pattern, col, re.IGNORECASE) for pattern in hp_names)
        ]
        matching_columns.sort()
        hyp_df = result_hyp_param[matching_columns]
        melted_df = hyp_df.melt(var_name="variable", value_name="value")
        melted_df["variable"] = melted_df["variable"].replace("lambda_hp", "lambda")

        def parse_variable(variable):
            parts = variable.split("_")
            return {"type": parts[-1], "channel": "_".join(parts[:-1])}

        parsed_vars = melted_df["variable"].apply(parse_variable).apply(pd.Series)
        melted_df[["type", "channel"]] = parsed_vars
        melted_df["type"] = pd.Categorical(
            melted_df["type"], categories=melted_df["type"].unique()
        )
        melted_df["channel"] = pd.Categorical(
            melted_df["channel"], categories=melted_df["channel"].unique()[::-1]
        )
        plt.figure(figsize=(12, 7))
        g = sns.FacetGrid(melted_df, col="type", sharex=False, height=6, aspect=1)

        def violin_plot(x, y, **kwargs):
            sns.violinplot(x=x, y=y, **kwargs, alpha=0.8, linewidth=0)

        g.map_dataframe(
            violin_plot, x="value", y="channel", hue="channel", palette="Set2"
        )
        g.set_titles("{col_name}")
        g.set_xlabels("Hyperparameter space")
        g.set_ylabels("")
        g.figure.suptitle("Hyperparameters Optimization Distributions", y=1.05)
        subtitle_text = (
            f"Sample distribution, iterations = "
            f"{self.model_outputs.iterations} x {len(self.model_outputs.trials)} trial"
        )
        g.figure.text(0.5, 0.98, subtitle_text, ha="center", fontsize=10)
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def create_pareto_front_plot(self, is_calibrated):
        """Create Pareto Front Plot."""
        unfiltered_pareto_results = self.unfiltered_pareto_result
        result_hyp_param = unfiltered_pareto_results.result_hyp_param
        pareto_fronts = self.pareto_result.pareto_fronts
        if is_calibrated:
            result_hyp_param["iterations"] = np.where(
                result_hyp_param["robynPareto"].isna(),
                np.nan,
                result_hyp_param["iterations"],
            )
            result_hyp_param = result_hyp_param.sort_values(
                by="robynPareto", na_position="first"
            )
        pareto_fronts_vec = list(range(1, pareto_fronts + 1))
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            result_hyp_param["nrmse"],
            result_hyp_param["decomp.rssd"],
            c=result_hyp_param["iterations"],
            cmap="Blues",
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Iterations")
        if is_calibrated:
            scatter = plt.scatter(
                result_hyp_param["nrmse"],
                result_hyp_param["decomp.rssd"],
                c=result_hyp_param["iterations"],
                cmap="Blues",
                s=result_hyp_param["mape"] * 100,
                alpha=1 - result_hyp_param["mape"],
            )
        for pfs in range(1, max(pareto_fronts_vec) + 1):
            temp = result_hyp_param[result_hyp_param["robynPareto"] == pfs]
            if len(temp) > 1:
                temp = temp.sort_values("nrmse")
                plt.plot(temp["nrmse"], temp["decomp.rssd"], color="coral", linewidth=2)
        plt.title(
            "Multi-objective Evolutionary Performance"
            + (" with Calibration" if is_calibrated else "")
        )
        plt.xlabel("NRMSE")
        plt.ylabel("DECOMP.RSSD")
        plt.suptitle(
            f"2D Pareto fronts with {self.model_outputs.nevergrad_algo or 'Unknown'}, "
            f"for {len(self.model_outputs.trials)} trial{'' if pareto_fronts == 1 else 's'} "
            f"with {self.model_outputs.iterations or 1} iterations each"
        )
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def create_ridgeline_model_convergence(self):
        """Create Ridgeline Model Convergence Plots."""
        all_plots = {}
        x_decomp_agg = self.unfiltered_pareto_result.x_decomp_agg
        paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends
        dt_ridges = x_decomp_agg[x_decomp_agg["rn"].isin(paid_media_spends)].copy()
        dt_ridges["iteration"] = (
            dt_ridges["iterNG"] - 1
        ) * self.model_outputs.cores + dt_ridges["iterPar"]
        dt_ridges = dt_ridges[["rn", "roi_total", "iteration", "trial"]]
        dt_ridges = dt_ridges.sort_values(["iteration", "rn"])
        iterations = self.model_outputs.iterations or 100
        qt_len = (
            1
            if iterations <= 100
            else (20 if iterations > 2000 else int(np.ceil(iterations / 100)))
        )
        set_qt = np.floor(np.linspace(1, iterations, qt_len + 1)).astype(int)
        set_bin = set_qt[1:]
        dt_ridges["iter_bin"] = pd.cut(
            dt_ridges["iteration"], bins=set_qt, labels=set_bin
        )
        dt_ridges = dt_ridges.dropna(subset=["iter_bin"])
        dt_ridges["iter_bin"] = pd.Categorical(
            dt_ridges["iter_bin"],
            categories=sorted(set_bin, reverse=True),
            ordered=True,
        )
        dt_ridges["trial"] = dt_ridges["trial"].astype("category")
        plot_vars = dt_ridges["rn"].unique()
        plot_n = int(np.ceil(len(plot_vars) / 6))
        metric = (
            "ROAS"
            if self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE
            else "CPA"
        )
        for pl in range(1, plot_n + 1):
            start_idx = (pl - 1) * 6
            loop_vars = plot_vars[start_idx : start_idx + 6]
            dt_ridges_loop = dt_ridges[dt_ridges["rn"].isin(loop_vars)]
            fig, axes = plt.subplots(
                nrows=len(loop_vars), figsize=(12, 3 * len(loop_vars)), sharex=False
            )
            if len(loop_vars) == 1:
                axes = [axes]
            for idx, var in enumerate(loop_vars):
                var_data = dt_ridges_loop[dt_ridges_loop["rn"] == var]
                offset = 0
                for iter_bin in sorted(var_data["iter_bin"].unique(), reverse=True):
                    bin_data = var_data[var_data["iter_bin"] == iter_bin]["roi_total"]
                    sns.kdeplot(
                        bin_data,
                        ax=axes[idx],
                        fill=True,
                        alpha=0.6,
                        color=plt.cm.GnBu(offset / len(var_data["iter_bin"].unique())),
                        label=f"Bin {iter_bin}",
                        warn_singular=False,
                    )
                    offset += 1
                axes[idx].set_title(f"{var} {metric}")
                axes[idx].set_ylabel("")
                axes[idx].legend().remove()
                axes[idx].spines["right"].set_visible(False)
                axes[idx].spines["top"].set_visible(False)
            plt.suptitle(f"{metric} Distribution over Iteration Buckets", fontsize=16)
            plt.tight_layout()
            fig = plt.gcf()
            plt.close(fig)
            all_plots[f"{metric}_convergence_{pl}"] = fig
        return all_plots

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> None:
        # Generate all plots
        solution_ids = self.pareto_result.pareto_solutions
        # Clean up nan values
        cleaned_solution_ids = [
            sid
            for sid in solution_ids
            if not (isinstance(sid, float) and math.isnan(sid))
        ]
        # Assign the cleaned list back to self.pareto_result.pareto_solutions
        self.pareto_result.pareto_solutions = cleaned_solution_ids
        figures: Dict[str, plt.Figure] = {}

        for solution_id in cleaned_solution_ids:
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

        if not self.model_outputs.hyper_fixed:
            prophet_decomp_plot = self.create_prophet_decomposition_plot()
            if prophet_decomp_plot:
                figures["prophet_decomp"] = prophet_decomp_plot
            hyperparameters_plot = self.create_hyperparameter_sampling_distribution()
            if hyperparameters_plot:
                figures["hyperparameters_sampling"] = hyperparameters_plot
            pareto_front_plot = self.create_pareto_front_plot(is_calibrated=False)
            if pareto_front_plot:
                figures["pareto_front"] = pareto_front_plot
            ridgeline_plots = self.create_ridgeline_model_convergence()
            figures.update(ridgeline_plots)

        # Display plots if required
        if display_plots:
            self.display_plots(figures)
