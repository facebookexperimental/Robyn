from pathlib import Path
import re
from typing import Dict, List, Optional, Union, Any
from matplotlib import ticker, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
from robyn.data.entities.enums import ProphetVariableType, DependentVarType
from robyn.data.entities.holidays_data import HolidaysData
from robyn.modeling.entities.featurized_mmm_data import FeaturizedMMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.data.entities.hyperparameters import AdstockType, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.visualization.base_visualizer import BaseVisualizer
from robyn.modeling.entities.modeloutputs import ModelOutputs
import matplotlib.dates as mdates
import math

logger = logging.getLogger(__name__)


class ParetoVisualizer(BaseVisualizer):
    """Visualizer for Pareto optimization results."""

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
        """Initialize ParetoVisualizer.
        
        Args:
            pareto_result: Pareto optimization results
            mmm_data: Marketing mix model data
            holiday_data: Holiday data for prophet variables
            hyperparameter: Model hyperparameters
            featurized_mmm_data: Featurized marketing mix model data
            unfiltered_pareto_result: Unfiltered Pareto results
            model_outputs: Model outputs data
        """
        super().__init__()
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.hyperparameter = hyperparameter
        self.featurized_mmm_data = featurized_mmm_data
        self.unfiltered_pareto_result = unfiltered_pareto_result
        self.model_outputs = model_outputs
        logger.info("Initialized ParetoVisualizer")

    def _baseline_vars(self, baseline_level: int, prophet_vars: List[ProphetVariableType] = []) -> list:
        """Returns a list of baseline variables based on the provided level.
        
        Args:
            baseline_level: The level of baseline variables to include (0-5)
            prophet_vars: List of prophet variables to include
            
        Returns:
            List of baseline variable names
        """
        logger.debug(f"Getting baseline variables for level {baseline_level}")
        
        if baseline_level < 0 or baseline_level > 5:
            raise ValueError("baseline_level must be between 0 and 5")

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

    def _validate_solution_id(self, solution_id: str) -> None:
        """Validate that a solution ID exists in the Pareto results.
        
        Args:
            solution_id: Solution ID to validate
            
        Raises:
            ValueError: If solution ID is invalid
        """
        if solution_id not in self.pareto_result.plot_data_collect:
            raise ValueError(f"Invalid solution ID: {solution_id}")

    def _setup_plot_defaults(self) -> Dict[str, Any]:
        """Set up default plotting parameters.
        
        Returns:
            Dictionary of default plotting parameters
        """
        return {
            "colors": {
                "actual": self.colors["secondary"],
                "predicted": self.colors["primary"],
                "positive": self.colors["positive"],
                "negative": self.colors["negative"],
                "neutral": self.colors["neutral"]
            },
            "alphas": {
                "main": self.alpha["primary"],
                "secondary": self.alpha["secondary"],
                "background": self.alpha["background"]
            },
            "line_styles": self.line_styles,
            "markers": self.markers,
            "font_sizes": self.fonts["sizes"]
        }

    def _create_standard_figure(self, figsize: Optional[tuple] = None) -> tuple:
        """Create a figure with standard styling.
        
        Args:
            figsize (Optional[tuple]): Optional figure size tuple (width, height)
                
        Returns:
            tuple: (figure, axes)
        """
        fig, ax = self.create_figure(figsize=figsize if figsize else self.figure_sizes["default"])
        ax.set_facecolor("white")
        return fig, ax
    
    def generate_waterfall(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate waterfall chart for specific solution."""
        logger.debug("Starting generation of waterfall plot")
        
        try:
            if solution_id not in self.pareto_result.plot_data_collect:
                raise ValueError(f"Invalid solution ID: {solution_id}")

            # Get data for specific solution
            plot_data = self.pareto_result.plot_data_collect[solution_id]
            waterfall_data = plot_data["plot2data"]["plotWaterfallLoop"].copy()

            # Get baseline variables
            prophet_vars = self.holiday_data.prophet_vars if self.holiday_data else []
            baseline_vars = self._baseline_vars(baseline_level=0, prophet_vars=prophet_vars)

            # Transform baseline variables
            waterfall_data["rn"] = np.where(
                waterfall_data["rn"].isin(baseline_vars),
                f"Baseline_L0",
                waterfall_data["rn"],
            )

            # Group and summarize data
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

            # Create figure using BaseVisualizer methods
            if ax is None:
                fig, ax = self.create_figure(figsize=self.figure_sizes["medium"])
            else:
                fig = None

            # Define colors using BaseVisualizer color scheme
            colors = {
                "Positive": self.colors["positive"],
                "Negative": self.colors["negative"]
            }

            # Create categorical y-axis positions
            y_pos = np.arange(len(waterfall_data))

            # Create horizontal bars
            bars = ax.barh(
                y=y_pos,
                width=waterfall_data["start"] - waterfall_data["end"],
                left=waterfall_data["end"],
                color=[colors[sign] for sign in waterfall_data["sign"]],
                height=0.6,
                alpha=self.alpha["primary"]
            )

            # Add text labels
            for idx, row in enumerate(waterfall_data.itertuples()):
                # Format value using BaseVisualizer formatter
                formatted_num = self.format_number(row.xDecompAgg)
                
                # Calculate x-position as the middle of the bar
                x_pos = (row.start + row.end) / 2
                
                # Add label with standardized font size
                ax.text(
                    x_pos,
                    y_pos[idx],
                    f"{formatted_num}\n{row.xDecompPerc*100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=self.fonts["sizes"]["annotation"],
                    alpha=self.alpha["annotation"]
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

            # Add standardized styling using BaseVisualizer methods
            self._set_standardized_labels(
                ax,
                xlabel="Contribution",
                ylabel=None,
                title=f"Response Decomposition Waterfall (Solution {solution_id})"
            )
            self._add_standardized_grid(ax, axis='x')
            self._set_standardized_spines(ax)

            # Create legend elements
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=colors["Positive"], label="Positive"),
                plt.Rectangle((0, 0), 1, 1, facecolor=colors["Negative"], label="Negative")
            ]
            legend_labels = ["Positive", "Negative"]

            # Add legend using BaseVisualizer method
            self._add_standardized_legend(
                ax,
                loc='lower right',
                ncol=2,
                handles=legend_handles,
                labels=legend_labels
            )

            # Set white background
            ax.set_facecolor("white")

            # Finalize the figure
            self.finalize_figure(tight_layout=True)

            logger.debug("Successfully generated waterfall plot")
            
            if fig:
                plt.close(fig)
                return fig
            return None

        except Exception as e:
            logger.error(f"Failed to generate waterfall plot: {str(e)}")
            raise

    def generate_fitted_vs_actual(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate time series plot comparing fitted vs actual values.

        Args:
            solution_id (str): The solution ID to generate the plot for
            ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object if ax is None
        """
        logger.debug("Starting generation of fitted vs actual plot")

        try:
            if solution_id not in self.pareto_result.plot_data_collect:
                raise ValueError(f"Invalid solution ID: {solution_id}")

            # Get data for specific solution
            plot_data = self.pareto_result.plot_data_collect[solution_id]
            ts_data = plot_data["plot5data"]["xDecompVecPlotMelted"].copy()

            # Ensure ds column is datetime and remove any NaT values
            ts_data["ds"] = pd.to_datetime(ts_data["ds"])
            ts_data = ts_data.dropna(subset=["ds"])

            if ts_data.empty:
                logger.warning(f"No valid date data found for solution {solution_id}")
                return None

            # Prepare line styles
            ts_data["linetype"] = np.where(ts_data["variable"] == "predicted", "solid", "dotted")
            ts_data["variable"] = ts_data["variable"].str.title()

            # Create figure using BaseVisualizer methods
            if ax is None:
                fig, ax = self.create_figure(figsize=self.figure_sizes["medium"])
            else:
                fig = None

            # Define colors using BaseVisualizer color scheme
            colors = {
                "Actual": self.colors["secondary"],  # Orange from BaseVisualizer
                "Predicted": self.colors["primary"],  # Blue from BaseVisualizer
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
                    linewidth=1.5,
                    color=colors[var],
                    alpha=self.alpha["primary"]
                )

            # Format y-axis with abbreviations using BaseVisualizer formatter
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))

            # Set y-axis limits with padding
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, y_max * 1.1)

            # Add training/validation/test splits if train_size exists
            train_size_series = self.pareto_result.x_decomp_agg[
                self.pareto_result.x_decomp_agg["sol_id"] == solution_id
            ]["train_size"]

            if not train_size_series.empty:
                train_size = float(train_size_series.iloc[0])
                
                if train_size > 0:
                    try:
                        # Get unique sorted dates
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

                            y_min, y_max = ax.get_ylim()

                            # Add vertical lines and labels
                            for idx, label, size in splits:
                                if 0 <= idx < len(unique_dates):
                                    date = unique_dates[idx]
                                    if pd.notna(date):
                                        # Add vertical line
                                        ax.axvline(
                                            date,
                                            color=self.colors["grid"],
                                            alpha=self.alpha["grid"],
                                            ymin=0,
                                            ymax=1.1,
                                            linestyle='--'
                                        )

                                        # Add rotated text label
                                        ax.text(
                                            date,
                                            y_max,
                                            f"{label}: {size*100:.1f}%",
                                            rotation=270,
                                            color=self.colors["annotation"],
                                            alpha=self.alpha["annotation"],
                                            fontsize=self.fonts["sizes"]["annotation"],
                                            ha="left",
                                            va="top"
                                        )
                    except Exception as e:
                        logger.warning(f"Error adding split lines: {str(e)}")

            # Add standardized styling using BaseVisualizer methods
            self._set_standardized_labels(
                ax,
                xlabel="Date",
                ylabel="Response",
                title=f"Actual vs. Predicted Response (Solution {solution_id})"
            )
            self._add_standardized_grid(ax)
            self._set_standardized_spines(ax)
            self._add_standardized_legend(
                ax,
                loc='lower right',
                ncol=2,
            )

            # Format dates on x-axis
            years = mdates.YearLocator()
            years_fmt = mdates.DateFormatter("%Y")
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(years_fmt)

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Set white background
            ax.set_facecolor("white")

            # Finalize the figure
            self.finalize_figure(tight_layout=True)

            logger.debug("Successfully generated fitted vs actual plot")
            
            if fig:
                plt.close(fig)
                return fig
            return None

        except Exception as e:
            logger.error(f"Failed to generate fitted vs actual plot: {str(e)}")
            raise

    def generate_diagnostic_plot(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate diagnostic scatter plot of fitted vs residual values.

        Args:
            solution_id (str): The solution ID to generate the plot for
            ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object if ax is None
        """
        logger.debug("Starting generation of diagnostic plot")

        try:
            if solution_id not in self.pareto_result.plot_data_collect:
                raise ValueError(f"Invalid solution ID: {solution_id}")

            # Get data for specific solution
            plot_data = self.pareto_result.plot_data_collect[solution_id]
            diag_data = plot_data["plot6data"]["xDecompVecPlot"].copy()

            # Calculate residuals
            diag_data["residuals"] = diag_data["actual"] - diag_data["predicted"]
            
            # Create figure using BaseVisualizer methods
            if ax is None:
                fig, ax = self.create_figure(figsize=self.figure_sizes["medium"])
            else:
                fig = None

            # Create scatter plot with BaseVisualizer colors
            scatter = ax.scatter(
                diag_data["predicted"],
                diag_data["residuals"],
                alpha=self.alpha["primary"],
                color=self.colors["primary"],
                label="Residuals"
            )

            # Add horizontal line at y=0
            ax.axhline(
                y=0,
                color=self.colors["baseline"],
                linestyle=self.line_styles["solid"],
                linewidth=0.8,
                alpha=self.alpha["annotation"]
            )

            # Fit LOWESS smoother
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Calculate smooth line
            smoothed = lowess(
                diag_data["residuals"],
                diag_data["predicted"],
                frac=0.2,
                return_sorted=True
            )

            # Plot smoothed line
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color=self.colors["secondary"],
                linewidth=2,
                alpha=self.alpha["secondary"],
                label="Smoothed trend"
            )

            # Calculate and plot confidence intervals
            residual_std = np.std(diag_data["residuals"])
            ax.fill_between(
                smoothed[:, 0],
                smoothed[:, 1] - 2 * residual_std,
                smoothed[:, 1] + 2 * residual_std,
                color=self.colors["secondary"],
                alpha=self.alpha["background"],
                label="95% Confidence interval"
            )

            # Add statistical annotations
            stats_text = (
                f"Standard deviation: {residual_std:.2f}\n"
                f"Mean residual: {np.mean(diag_data['residuals']):.2f}\n"
                f"Median residual: {np.median(diag_data['residuals']):.2f}"
            )
            
            ax.text(
                0.02, 0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=self.alpha["annotation"],
                    edgecolor=self.colors["grid"]
                ),
                fontsize=self.fonts["sizes"]["annotation"]
            )

            # Format axes with abbreviations using BaseVisualizer formatter
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))

            # Add standardized styling using BaseVisualizer methods
            self._set_standardized_labels(
                ax,
                xlabel="Fitted Values",
                ylabel="Residuals",
                title=f"Diagnostic Plot: Fitted vs Residuals (Solution {solution_id})"
            )
            self._add_standardized_grid(ax)
            self._set_standardized_spines(ax)
            self._add_standardized_legend(
                ax,
                loc='lower right',
                title="Components"
            )

            # Set white background
            ax.set_facecolor("white")

            # Calculate and set reasonable axis limits with padding
            x_range = diag_data["predicted"].max() - diag_data["predicted"].min()
            y_range = diag_data["residuals"].max() - diag_data["residuals"].min()
            
            ax.set_xlim(
                diag_data["predicted"].min() - 0.05 * x_range,
                diag_data["predicted"].max() + 0.05 * x_range
            )
            ax.set_ylim(
                diag_data["residuals"].min() - 0.05 * y_range,
                diag_data["residuals"].max() + 0.05 * y_range
            )

            # Add diagnostic lines
            mean_residual = np.mean(diag_data["residuals"])
            if abs(mean_residual) > residual_std * 0.1:  # Only show if mean is notably different from 0
                ax.axhline(
                    y=mean_residual,
                    color=self.colors["annotation"],
                    linestyle=self.line_styles["dotted"],
                    alpha=self.alpha["annotation"],
                    label="Mean residual"
                )

            # Finalize the figure
            self.finalize_figure(tight_layout=True)

            logger.debug("Successfully generated diagnostic plot")
            
            if fig:
                plt.close(fig)
                return fig
            return None

        except Exception as e:
            logger.error(f"Failed to generate diagnostic plot: {str(e)}")
            raise
    
    def generate_immediate_vs_carryover(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate stacked bar chart comparing immediate vs carryover effects.

        Args:
            solution_id: Solution ID to visualize
            ax: Optional matplotlib axes to plot on

        Returns:
            Optional[plt.Figure]: Generated figure if ax is None
        """
        logger.debug("Starting generation of immediate vs carryover plot")

        try:
            self._validate_solution_id(solution_id)

            # Get and prepare data
            plot_data = self.pareto_result.plot_data_collect[solution_id]
            df_imme_caov = plot_data["plot7data"].copy()

            # Ensure percentage is numeric
            df_imme_caov["percentage"] = pd.to_numeric(df_imme_caov["percentage"], errors="coerce")

            # Sort channels alphabetically
            df_imme_caov = df_imme_caov.sort_values("rn", ascending=True)

            # Set up type factor levels
            df_imme_caov["type"] = pd.Categorical(
                df_imme_caov["type"],
                categories=["Immediate", "Carryover"],
                ordered=True
            )

            # Create figure
            if ax is None:
                fig, ax = self._create_standard_figure(figsize=self.figure_sizes["medium"])
            else:
                fig = None

            # Set up colors using BaseVisualizer scheme
            colors = {
                "Immediate": self.colors["primary"],
                "Carryover": self.colors["secondary"]
            }

            # Initialize variables for stacked bars
            bottom = np.zeros(len(df_imme_caov["rn"].unique()))
            y_pos = range(len(df_imme_caov["rn"].unique()))
            channels = df_imme_caov["rn"].unique()
            types = ["Immediate", "Carryover"]

            # Normalize percentages to sum to 100% for each channel
            for channel in channels:
                mask = df_imme_caov["rn"] == channel
                total = df_imme_caov.loc[mask, "percentage"].sum()
                if total > 0:
                    df_imme_caov.loc[mask, "percentage"] = (
                        df_imme_caov.loc[mask, "percentage"] / total
                    )

            # Create stacked bars
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
                    alpha=self.alpha["primary"]
                )

                # Add percentage labels
                for i, (rect, percentage) in enumerate(zip(bars, percentages)):
                    width = rect.get_width()
                    x_pos = bottom[i] + width / 2
                    percentage_text = f"{percentage*100:.0f}%"
                    ax.text(
                        x_pos,
                        i,
                        percentage_text,
                        ha="center",
                        va="center",
                        fontsize=self.fonts["sizes"]["annotation"],
                        color=self.colors["annotation"]
                    )

                bottom += percentages

            # Set up axes
            ax.set_yticks(y_pos)
            ax.set_yticklabels(channels)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
            ax.set_xlim(0, 1)

            # Add styling using BaseVisualizer methods
            self._set_standardized_labels(
                ax,
                xlabel="Response Percentage",
                ylabel=None,
                title=f"Immediate vs. Carryover Response (Solution {solution_id})"
            )
            self._add_standardized_grid(ax, axis='x')
            self._set_standardized_spines(ax)
            self._add_standardized_legend(
                ax,
                loc='lower right',
                ncol=2,
            )

            # Finalize figure
            self.finalize_figure(tight_layout=True)

            logger.debug("Successfully generated immediate vs carryover plot")
            
            if fig:
                plt.close(fig)
                return fig
            return None

        except Exception as e:
            logger.error(f"Failed to generate immediate vs carryover plot: {str(e)}")
            raise

    def generate_adstock_rate(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate adstock rate visualization based on adstock type.

        Args:
            solution_id: Solution ID to visualize
            ax: Optional matplotlib axes to plot on

        Returns:
            Optional[plt.Figure]: Generated figure if ax is None
        """
        logger.debug("Starting generation of adstock plot")

        try:
            self._validate_solution_id(solution_id)
            plot_data = self.pareto_result.plot_data_collect[solution_id]
            adstock_data = plot_data["plot3data"]

            if ax is None:
                fig, ax = self._create_standard_figure(figsize=self.figure_sizes["medium"])
            else:
                fig = None

            if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
                # Handle Geometric Adstock
                dt_geometric = adstock_data["dt_geometric"].copy()
                dt_geometric = dt_geometric.sort_values("channels", ascending=True)

                # Create horizontal bars
                bars = ax.barh(
                    y=range(len(dt_geometric)),
                    width=dt_geometric["thetas"],
                    height=0.5,
                    color=self.colors["secondary"],
                    alpha=self.alpha["primary"]
                )

                # Add value labels
                for i, theta in enumerate(dt_geometric["thetas"]):
                    ax.text(
                        theta + 0.01,
                        i,
                        f"{theta*100:.1f}%",
                        va="center",
                        fontsize=self.fonts["sizes"]["annotation"],
                        color=self.colors["annotation"]
                    )

                # Set up axes
                ax.set_yticks(range(len(dt_geometric)))
                ax.set_yticklabels(dt_geometric["channels"])
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.25, 0.25))

                # Add styling
                interval_type = (
                    self.mmm_data.mmmdata_spec.interval_type if self.mmm_data else "day"
                )
                self._set_standardized_labels(
                    ax,
                    xlabel=f"Thetas [by {interval_type}]",
                    ylabel=None,
                    title=f"Geometric Adstock: Fixed Rate Over Time (Solution {solution_id})"
                )

            elif self.hyperparameter.adstock in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
                # Handle Weibull Adstock
                weibull_data = adstock_data["weibullCollect"]
                channels = sorted(weibull_data["channel"].unique())
                rows = (len(channels) + 2) // 3  # Calculate rows needed for 3 columns

                if ax is None:
                    # Create new figure with subplots
                    fig = plt.figure(figsize=self.figure_sizes["wide"])
                    gs = fig.add_gridspec(rows, 3, hspace=0.4, wspace=0.3)
                    axes = []
                    for i in range(rows):
                        for j in range(3):
                            axes.append(fig.add_subplot(gs[i, j]))
                else:
                    # Use existing axes layout
                    gs = ax.get_gridspec()
                    fig = ax.figure
                    axes = [ax]

                # Create plots for each channel
                for idx, channel in enumerate(channels):
                    if idx < len(axes):
                        ax_sub = axes[idx]
                        channel_data = weibull_data[weibull_data["channel"] == channel]

                        # Plot decay curve
                        ax_sub.plot(
                            channel_data["x"],
                            channel_data["decay_accumulated"],
                            color=self.colors["primary"],
                            alpha=self.alpha["primary"],
                            linewidth=2
                        )

                        # Add halflife line
                        ax_sub.axhline(
                            y=0.5,
                            color=self.colors["grid"],
                            linestyle=self.line_styles["dashed"],
                            alpha=self.alpha["grid"]
                        )

                        # Add halflife label
                        halflife_x = channel_data[
                            channel_data["decay_accumulated"].between(0.49, 0.51)
                        ]["x"].iloc[0]
                        
                        ax_sub.text(
                            halflife_x * 1.1,
                            0.52,
                            f"Halflife: {halflife_x:.1f}",
                            color=self.colors["annotation"],
                            fontsize=self.fonts["sizes"]["annotation"],
                            va="bottom",
                            ha="left"
                        )

                        # Style subplot
                        self._set_standardized_labels(
                            ax_sub,
                            xlabel="Time",
                            ylabel="Decay Rate" if idx % 3 == 0 else None,
                            title=channel
                        )
                        self._add_standardized_grid(ax_sub)
                        self._set_standardized_spines(ax_sub)
                        
                        # Set axis limits
                        ax_sub.set_ylim(0, 1.1)
                        ax_sub.set_xlim(0, max(channel_data["x"]) * 1.2)

                # Hide unused subplots
                for idx in range(len(channels), len(axes)):
                    axes[idx].set_visible(False)

                # Add overall title
                fig.suptitle(
                    f"Weibull {self.hyperparameter.adstock.value} Adstock Decay Curves (Solution {solution_id})",
                    fontsize=self.fonts["sizes"]["title"],
                    y=1.02
                )

            else:
                logger.warning(f"Unsupported adstock type: {self.hyperparameter.adstock}")
                return None

            # Add common styling
            self._add_standardized_grid(ax, axis='x')
            self._set_standardized_spines(ax)
            ax.set_facecolor("white")

            # Finalize figure
            self.finalize_figure(tight_layout=True)

            logger.debug("Successfully generated adstock plot")
            
            if fig:
                plt.close(fig)
                return fig
            return None

        except Exception as e:
            logger.error(f"Failed to generate adstock plot: {str(e)}")
            raise    

    def create_prophet_decomposition_plot(self) -> Optional[plt.Figure]:
        """Create Prophet Decomposition Plot showing model components."""
        logger.debug("Starting generation of prophet decomposition plot")

        try:
            # Get prophet variables
            prophet_vars = (
                [ProphetVariableType(var) for var in self.holiday_data.prophet_vars]
                if self.holiday_data and self.holiday_data.prophet_vars
                else []
            )
            factor_vars = self.mmm_data.mmmdata_spec.factor_vars if self.mmm_data else []

            if not (prophet_vars or factor_vars):
                logger.info("No prophet or factor variables found")
                return None

            # Prepare data
            df = self.featurized_mmm_data.dt_mod.copy()
            prophet_vars_str = [variable.value for variable in prophet_vars]
            prophet_vars_str.sort(reverse=True)

            # Combine variables
            value_variables = (
                [
                    "dep_var"
                    if hasattr(df, "dep_var")
                    else self.mmm_data.mmmdata_spec.dep_var
                ]
                + factor_vars
                + prophet_vars_str
            )

            # Prepare long format data
            df_long = df.melt(
                id_vars=["ds"],
                value_vars=value_variables,
                var_name="variable",
                value_name="value",
            )
            df_long["ds"] = pd.to_datetime(df_long["ds"])

            # Create figure with subplots
            n_vars = len(df_long["variable"].unique())
            fig = plt.figure(figsize=(12, 3 * n_vars))
            
            # Create gridspec with tighter spacing
            gs = fig.add_gridspec(
                n_vars, 
                1,
                height_ratios=[1] * n_vars,
                hspace=0.7  # Adjust vertical space between subplots
            )

            # Create subplot for each variable
            for i, var in enumerate(df_long["variable"].unique()):
                ax = fig.add_subplot(gs[i])
                var_data = df_long[df_long["variable"] == var]

                # Plot time series
                ax.plot(
                    var_data["ds"],
                    var_data["value"],
                    color=self.colors["primary"],
                    alpha=self.alpha["primary"]
                )

                # Style subplot
                self._set_standardized_labels(
                    ax,
                    xlabel=None,
                    ylabel=None,
                    title=var
                )
                self._add_standardized_grid(ax)
                self._set_standardized_spines(ax)
                ax.set_facecolor("white")

                # Format x-axis dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                # Adjust subplot padding
                ax.margins(x=0.02)

            # Add overall title with adjusted position
            fig.suptitle(
                "Prophet Decomposition",
                fontsize=self.fonts["sizes"]["title"],
                y=1  # Moved title closer to the first subplot
            )

            # Fine-tune the layout
            plt.tight_layout()
            
            # Adjust layout to accommodate the title
            plt.subplots_adjust(top=0.95)  # Adjust top margin to prevent title overlap
            
            logger.debug("Successfully generated prophet decomposition plot")
            return fig

        except Exception as e:
            logger.error(f"Failed to generate prophet decomposition plot: {str(e)}")
            raise

    def create_hyperparameter_sampling_distribution(self) -> Optional[plt.Figure]:
        """Create Hyperparameter Sampling Distribution Plot.

        Returns:
            Optional[plt.Figure]: Generated figure showing hyperparameter distributions
        """
        logger.debug("Starting generation of hyperparameter sampling distribution plot")

        try:
            if self.unfiltered_pareto_result is None:
                logger.info("No unfiltered Pareto results available")
                return None

            # Get hyperparameter data
            result_hyp_param = self.unfiltered_pareto_result.result_hyp_param
            hp_names = list(self.hyperparameter.hyperparameters.keys())
            hp_names = [name.replace("lambda", "lambda_hp") for name in hp_names]

            # Find matching columns
            matching_columns = [
                col
                for col in result_hyp_param.columns
                if any(re.search(pattern, col, re.IGNORECASE) for pattern in hp_names)
            ]
            matching_columns.sort()

            if not matching_columns:
                logger.info("No matching hyperparameter columns found")
                return None

            # Prepare data
            hyp_df = result_hyp_param[matching_columns]
            melted_df = hyp_df.melt(var_name="variable", value_name="value")
            melted_df["variable"] = melted_df["variable"].replace("lambda_hp", "lambda")

            # Parse variable names
            def parse_variable(variable):
                parts = variable.split("_")
                return {"type": parts[-1], "channel": "_".join(parts[:-1])}

            parsed_vars = melted_df["variable"].apply(parse_variable).apply(pd.Series)
            melted_df[["type", "channel"]] = parsed_vars

            # Create categorical variables
            melted_df["type"] = pd.Categorical(
                melted_df["type"],
                categories=melted_df["type"].unique()
            )
            melted_df["channel"] = pd.Categorical(
                melted_df["channel"],
                categories=melted_df["channel"].unique()[::-1]
            )

            # Create figure using facet grid
            g = sns.FacetGrid(
                melted_df,
                col="type",
                sharex=False,
                height=6,
                aspect=1
            )

            def violin_plot(x, y, **kwargs):
                sns.violinplot(
                    x=x,
                    y=y,
                    **kwargs,
                    alpha=self.alpha["primary"],
                    linewidth=0,
                    palette=sns.color_palette("Set2")
                )

            g.map_dataframe(
                violin_plot,
                x="value",
                y="channel",
                hue="channel"
            )

            # Style facets
            g.set_titles("{col_name}")
            g.set_xlabels("Hyperparameter space")
            g.set_ylabels("")

            # Add titles
            g.figure.suptitle(
                "Hyperparameters Optimization Distributions",
                y=1.05,
                fontsize=self.fonts["sizes"]["title"]
            )
            subtitle_text = (
                f"Sample distribution, iterations = "
                f"{self.model_outputs.iterations} x {len(self.model_outputs.trials)} trial"
            )
            g.figure.text(
                0.5,
                0.98,
                subtitle_text,
                ha="center",
                fontsize=self.fonts["sizes"]["subtitle"]
            )

            # Finalize figure
            plt.subplots_adjust(top=0.9)
            plt.tight_layout()
            
            logger.debug("Successfully generated hyperparameter sampling distribution plot")
            return g.figure

        except Exception as e:
            logger.error(f"Failed to generate hyperparameter sampling distribution plot: {str(e)}")
            raise

    def create_pareto_front_plot(self, is_calibrated: bool) -> Optional[plt.Figure]:
        """Create Pareto Front Plot showing optimization performance.

        Args:
            is_calibrated: Whether the model is calibrated

        Returns:
            Optional[plt.Figure]: Generated figure showing Pareto fronts
        """
        logger.debug(f"Starting generation of Pareto front plot (calibrated={is_calibrated})")

        try:
            unfiltered_pareto_results = self.unfiltered_pareto_result
            result_hyp_param = unfiltered_pareto_results.result_hyp_param
            pareto_fronts = self.pareto_result.pareto_fronts

            # Create figure using BaseVisualizer
            fig, ax = self._create_standard_figure(figsize=self.figure_sizes["medium"])

            # Handle calibrated case
            if is_calibrated:
                result_hyp_param["iterations"] = np.where(
                    result_hyp_param["robynPareto"].isna(),
                    np.nan,
                    result_hyp_param["iterations"],
                )
                result_hyp_param = result_hyp_param.sort_values(
                    by="robynPareto", na_position="first"
                )

            # Create main scatter plot
            scatter = ax.scatter(
                result_hyp_param["nrmse"],
                result_hyp_param["decomp.rssd"],
                c=result_hyp_param["iterations"],
                cmap="Blues",
                alpha=self.alpha["primary"]
            )
            plt.colorbar(scatter, label="Iterations")

            # Add calibration-specific scatter if needed
            if is_calibrated and "mape" in result_hyp_param.columns:
                scatter = ax.scatter(
                    result_hyp_param["nrmse"],
                    result_hyp_param["decomp.rssd"],
                    c=result_hyp_param["iterations"],
                    cmap="Blues",
                    s=result_hyp_param["mape"] * 100,
                    alpha=1 - result_hyp_param["mape"]
                )

            # Plot Pareto fronts
            pareto_fronts_vec = list(range(1, pareto_fronts + 1))
            for pfs in pareto_fronts_vec:
                temp = result_hyp_param[result_hyp_param["robynPareto"] == pfs]
                if len(temp) > 1:
                    temp = temp.sort_values("nrmse")
                    ax.plot(
                        temp["nrmse"],
                        temp["decomp.rssd"],
                        color=self.colors["secondary"],
                        linewidth=2,
                        alpha=self.alpha["secondary"]
                    )

            # Add styling using BaseVisualizer methods
            self._set_standardized_labels(
                ax,
                xlabel="NRMSE",
                ylabel="DECOMP.RSSD",
                title="Multi-objective Evolutionary Performance" + 
                    (" with Calibration" if is_calibrated else "")
            )
            self._add_standardized_grid(ax)
            self._set_standardized_spines(ax)

            # Add subtitle with algorithm details
            subtitle = (
                f"2D Pareto fronts with {self.model_outputs.nevergrad_algo or 'Unknown'}, "
                f"for {len(self.model_outputs.trials)} trial"
                f"{'s' if pareto_fronts != 1 else ''} "
                f"with {self.model_outputs.iterations or 1} iterations each"
            )
            plt.suptitle(subtitle, y=1.05, fontsize=self.fonts["sizes"]["subtitle"])

            self.finalize_figure(tight_layout=True)
            
            logger.debug("Successfully generated Pareto front plot")
            return fig

        except Exception as e:
            logger.error(f"Failed to generate Pareto front plot: {str(e)}")
            raise

    def create_ridgeline_model_convergence(self) -> Dict[str, plt.Figure]:
        """Create Ridgeline Model Convergence Plots.

        Returns:
            Dict[str, plt.Figure]: Dictionary of generated ridgeline plots
        """
        logger.debug("Starting generation of ridgeline model convergence plots")

        try:
            all_plots = {}
            x_decomp_agg = self.unfiltered_pareto_result.x_decomp_agg
            paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends

            # Prepare data
            dt_ridges = x_decomp_agg[x_decomp_agg["rn"].isin(paid_media_spends)].copy()
            dt_ridges["iteration"] = (
                dt_ridges["iterNG"] - 1
            ) * self.model_outputs.cores + dt_ridges["iterPar"]
            dt_ridges = dt_ridges[["rn", "roi_total", "iteration", "trial"]]
            dt_ridges = dt_ridges.sort_values(["iteration", "rn"])

            # Calculate iteration bins
            iterations = self.model_outputs.iterations or 100
            qt_len = (
                1 if iterations <= 100
                else (20 if iterations > 2000 else int(np.ceil(iterations / 100)))
            )
            set_qt = np.floor(np.linspace(1, iterations, qt_len + 1)).astype(int)
            set_bin = set_qt[1:]

            # Create iteration bins
            dt_ridges["iter_bin"] = pd.cut(
                dt_ridges["iteration"],
                bins=set_qt,
                labels=set_bin
            )
            dt_ridges = dt_ridges.dropna(subset=["iter_bin"])
            dt_ridges["iter_bin"] = pd.Categorical(
                dt_ridges["iter_bin"],
                categories=sorted(set_bin, reverse=True),
                ordered=True
            )
            dt_ridges["trial"] = dt_ridges["trial"].astype("category")

            # Determine metric type
            metric = (
                "ROAS"
                if self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE
                else "CPA"
            )

            # Create plots for each set of variables
            plot_vars = dt_ridges["rn"].unique()
            plot_n = int(np.ceil(len(plot_vars) / 6))

            for pl in range(1, plot_n + 1):
                start_idx = (pl - 1) * 6
                loop_vars = plot_vars[start_idx:start_idx + 6]
                dt_ridges_loop = dt_ridges[dt_ridges["rn"].isin(loop_vars)]

                # Create figure for this set of variables
                fig, axes = plt.subplots(
                    nrows=len(loop_vars),
                    figsize=(12, 3 * len(loop_vars)),
                    sharex=False
                )
                
                if len(loop_vars) == 1:
                    axes = [axes]

                # Create ridge plot for each variable
                for idx, var in enumerate(loop_vars):
                    var_data = dt_ridges_loop[dt_ridges_loop["rn"] == var]
                    offset = 0
                    
                    # Plot distributions for each iteration bin
                    for iter_bin in sorted(var_data["iter_bin"].unique(), reverse=True):
                        bin_data = var_data[var_data["iter_bin"] == iter_bin]["roi_total"]
                        
                        sns.kdeplot(
                            bin_data,
                            ax=axes[idx],
                            fill=True,
                            alpha=self.alpha["secondary"],
                            color=plt.cm.GnBu(offset / len(var_data["iter_bin"].unique())),
                            label=f"Bin {iter_bin}",
                            warn_singular=False
                        )
                        offset += 1

                    # Style subplot
                    axes[idx].set_title(
                        f"{var} {metric}",
                        fontsize=self.fonts["sizes"]["subtitle"]
                    )
                    axes[idx].set_ylabel("")
                    axes[idx].legend().remove()
                    self._set_standardized_spines(axes[idx])

                # Add overall title
                plt.suptitle(
                    f"{metric} Distribution over Iteration Buckets",
                    fontsize=self.fonts["sizes"]["title"]
                )

                # Finalize figure
                self.finalize_figure(tight_layout=True)
                all_plots[f"{metric}_convergence_{pl}"] = fig

            logger.debug("Successfully generated ridgeline model convergence plots")
            return all_plots

        except Exception as e:
            logger.error(f"Failed to generate ridgeline model convergence plots: {str(e)}")
            raise

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> Dict[str, plt.Figure]:
        """Generate and optionally display/export all available plots.

        Args:
            display_plots: Whether to display the plots
            export_location: Optional path to export plots

        Returns:
            Dict[str, plt.Figure]: Dictionary of all generated plots
        """
        logger.info("Generating all Pareto plots")
        figures: Dict[str, plt.Figure] = {}

        try:
            # Clean solution IDs
            cleaned_solution_ids = [
                sid for sid in self.pareto_result.pareto_solutions
                if not (isinstance(sid, float) and math.isnan(sid))
            ]

            if cleaned_solution_ids:
                # Generate plots for first solution only
                solution_id = cleaned_solution_ids[0]
                logger.info(f"Generating plots for solution {solution_id}")

                # Core plots
                plot_methods = {
                    "waterfall": self.generate_waterfall,
                    "fitted_vs_actual": self.generate_fitted_vs_actual,
                    "diagnostic": self.generate_diagnostic_plot,
                    "immediate_vs_carryover": self.generate_immediate_vs_carryover,
                    "adstock_rate": self.generate_adstock_rate
                }

                for name, method in plot_methods.items():
                    try:
                        fig = method(solution_id)
                        if fig:
                            figures[f"{name}_{solution_id}"] = fig
                            logger.debug(f"Generated {name} plot")
                    except Exception as e:
                        logger.error(f"Failed to generate {name} plot: {str(e)}")

            # Generate additional plots if not using fixed hyperparameters
            if not self.model_outputs.hyper_fixed:
                additional_plots = {
                    "prophet_decomp": lambda: self.create_prophet_decomposition_plot(),
                    "hyperparameter_sampling": lambda: self.create_hyperparameter_sampling_distribution(),
                    "pareto_front": lambda: self.create_pareto_front_plot(is_calibrated=False),
                    "pareto_front_calibrated": lambda: self.create_pareto_front_plot(is_calibrated=True)
                }

                for name, method in additional_plots.items():
                    try:
                        fig = method()
                        if fig:
                            figures[name] = fig
                            logger.debug(f"Generated {name} plot")
                    except Exception as e:
                        logger.error(f"Failed to generate {name} plot: {str(e)}")

                # Generate ridgeline plots
                try:
                    ridgeline_plots = self.create_ridgeline_model_convergence()
                    figures.update(ridgeline_plots)
                    logger.debug("Generated ridgeline plots")
                except Exception as e:
                    logger.error(f"Failed to generate ridgeline plots: {str(e)}")

            # Display plots if requested
            if display_plots:
                logger.info(f"Displaying {len(figures)} plots")
                self.display_plots(figures)

            # Export plots if location provided
            if export_location:
                logger.info(f"Exporting plots to {export_location}")
                self.export_plots_fig(export_location, figures)

            return figures

        except Exception as e:
            logger.error(f"Failed to generate all plots: {str(e)}")
            raise        
        