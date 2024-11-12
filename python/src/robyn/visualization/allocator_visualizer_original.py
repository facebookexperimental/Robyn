import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from robyn.allocator.entities.allocation_results import AllocationResult


logger = logging.getLogger(__name__)


class AllocationPlotter:
    """Creates visualizations for allocation results matching R version."""

    def __init__(self, result: AllocationResult):
        """Initialize plotter with allocation results and default settings.

        Args:
            result: AllocationResult containing optimization results to visualize
        """
        logger.info("Initializing AllocationPlotter")
        if result is None:
            logger.error("AllocationResult cannot be None")
            raise ValueError("AllocationResult cannot be None")

        # Store allocation results
        self.result = result
        logger.debug("Stored allocation results: %s", str(result))

        # Use matplotlib's built-in clean style
        plt.style.use("bmh")
        logger.debug("Set matplotlib style to 'bmh'")

        # Set default plot settings
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False
        logger.debug("Applied default plot settings")

        # Store standard figure size and colors
        self.fig_size = (12, 8)
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))
        logger.debug("Set figure size to %s and generated color palette", str(self.fig_size))

        # Set color scheme
        self.current_color = "lightgray"
        self.optimal_color = "#4688C7"  # Steel blue
        self.positive_color = "#2ECC71"  # Green
        self.negative_color = "#E74C3C"  # Red
        logger.debug("Initialized color scheme")

    def plot_all(self) -> Dict[str, plt.Figure]:
        """Generate all one-pager plots for allocation results.

        Returns:
            Dictionary of plot names to figures
        """
        logger.info("Generating all allocation plots")
        plots = {}
        try:
            plots = {
                "spend_allocation": self.plot_spend_allocation(),
                "response_curves": self.plot_response_curves(),
                "efficiency_frontier": self.plot_efficiency_frontier(),
                "spend_vs_response": self.plot_spend_vs_response(),
                "summary_metrics": self.plot_summary_metrics(),
            }
            logger.info("Successfully generated all plots")
            logger.debug("Generated plots: %s", list(plots.keys()))
        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise
        return plots

    def plot_spend_allocation(self) -> plt.Figure:
        """Plot spend allocation comparison between current and optimized."""
        logger.info("Plotting spend allocation comparison")
        
        if self.result is None:
            logger.error("No allocation results available for spend allocation plot")
            raise ValueError("No allocation results available. Call plot_all() first.")

        try:
            fig, ax = plt.subplots(figsize=self.fig_size)
            df = self.result.optimal_allocations
            logger.debug("Processing allocation data with %d channels", len(df))

            channels = df["channel"].values
            x = np.arange(len(channels))
            width = 0.35

            # Plot bars
            current_spend = df["current_spend"].values
            optimal_spend = df["optimal_spend"].values
            logger.debug("Current total spend: %.2f, Optimal total spend: %.2f", 
                        current_spend.sum(), optimal_spend.sum())

            ax.bar(
                x - width / 2, current_spend, width, label="Current", 
                color=self.current_color, edgecolor="gray", alpha=0.7
            )
            ax.bar(
                x + width / 2, optimal_spend, width, label="Optimized",
                color=self.optimal_color, edgecolor="gray", alpha=0.7
            )

            # Add spend change percentage labels
            for i, (curr, opt) in enumerate(zip(current_spend, optimal_spend)):
                pct_change = ((opt / curr) - 1) * 100
                logger.debug("Channel %s spend change: %.1f%%", channels[i], pct_change)
                color = self.positive_color if pct_change >= 0 else self.negative_color
                ax.text(i, max(curr, opt), f"{pct_change:+.1f}%", 
                       ha="center", va="bottom", color=color)

            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha="right")
            ax.set_ylabel("Spend")
            ax.set_title("Media Spend Allocation")
            ax.legend()

            plt.tight_layout()
            logger.info("Successfully created spend allocation plot")
            return fig
        except Exception as e:
            logger.error("Failed to create spend allocation plot: %s", str(e))
            raise

    def plot_response_curves(self) -> plt.Figure:
        """Plot response curves with current and optimal points."""
        logger.info("Plotting response curves")

        if self.result is None:
            logger.error("No allocation results available for response curves plot")
            raise ValueError("No allocation results available. Call plot_all() first.")

        try:
            curves_df = self.result.response_curves
            channels = curves_df["channel"].unique()
            n_channels = len(channels)
            logger.debug("Processing response curves for %d channels", n_channels)

            ncols = min(3, n_channels)
            nrows = (n_channels + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1 or ncols == 1:
                axes = axes.reshape(-1, 1)

            for idx, channel in enumerate(channels):
                logger.debug("Plotting response curve for channel: %s", channel)
                row = idx // ncols
                col = idx % ncols
                ax = axes[row, col]

                channel_data = curves_df[curves_df["channel"] == channel]

                # Plot response curve
                ax.plot(channel_data["spend"], channel_data["response"], 
                       color=self.optimal_color, alpha=0.6)

                # Add current and optimal points
                current_data = channel_data[channel_data["is_current"]]
                optimal_data = channel_data[channel_data["is_optimal"]]

                if not current_data.empty:
                    logger.debug("%s current point: spend=%.2f, response=%.2f", 
                               channel, current_data["spend"].iloc[0], 
                               current_data["response"].iloc[0])
                    ax.scatter(
                        current_data["spend"].iloc[0],
                        current_data["response"].iloc[0],
                        color=self.negative_color,
                        label="Current",
                        s=100,
                    )
                if not optimal_data.empty:
                    logger.debug("%s optimal point: spend=%.2f, response=%.2f", 
                               channel, optimal_data["spend"].iloc[0], 
                               optimal_data["response"].iloc[0])
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

            # Remove empty subplots
            for idx in range(n_channels, nrows * ncols):
                row = idx // ncols
                col = idx % ncols
                fig.delaxes(axes[row, col])

            plt.tight_layout()
            logger.info("Successfully created response curves plot")
            return fig
        except Exception as e:
            logger.error("Failed to create response curves plot: %s", str(e))
            raise

    def plot_efficiency_frontier(self) -> plt.Figure:
        """Plot efficiency frontier showing spend vs response relationship."""
        logger.info("Plotting efficiency frontier")

        if self.result is None:
            logger.error("No allocation results available for efficiency frontier plot")
            raise ValueError("No allocation results available. Call plot_all() first.")

        try:
            fig, ax = plt.subplots(figsize=self.fig_size)
            df = self.result.optimal_allocations

            # Calculate totals
            current_total_spend = df["current_spend"].sum()
            current_total_response = df["current_response"].sum()
            optimal_total_spend = df["optimal_spend"].sum()
            optimal_total_response = df["optimal_response"].sum()

            logger.debug("Current totals - spend: %.2f, response: %.2f", 
                        current_total_spend, current_total_response)
            logger.debug("Optimal totals - spend: %.2f, response: %.2f", 
                        optimal_total_spend, optimal_total_response)

            # Plot points and line
            ax.scatter(
                current_total_spend, current_total_response, 
                color=self.negative_color, s=100, label="Current", zorder=2
            )
            ax.scatter(
                optimal_total_spend, optimal_total_response, 
                color=self.positive_color, s=100, label="Optimal", zorder=2
            )

            ax.plot(
                [current_total_spend, optimal_total_spend],
                [current_total_response, optimal_total_response],
                "--", color="gray", alpha=0.5, zorder=1
            )

            # Add labels
            pct_spend_change = ((optimal_total_spend / current_total_spend) - 1) * 100
            pct_response_change = ((optimal_total_response / current_total_response) - 1) * 100
            
            logger.debug("Total spend change: %.1f%%", pct_spend_change)
            logger.debug("Total response change: %.1f%%", pct_response_change)

            ax.annotate(
                f"Spend: {pct_spend_change:+.1f}%\nResponse: {pct_response_change:+.1f}%",
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
            logger.info("Successfully created efficiency frontier plot")
            return fig
        except Exception as e:
            logger.error("Failed to create efficiency frontier plot: %s", str(e))
            raise

    def plot_spend_vs_response(self) -> plt.Figure:
        """Plot channel-level spend vs response changes."""
        logger.info("Plotting spend vs response comparison")

        if self.result is None:
            logger.error("No allocation results available for spend vs response plot")
            raise ValueError("No allocation results available. Call plot_all() first.")

        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            df = self.result.optimal_allocations
            channels = df["channel"].values
            logger.debug("Processing %d channels for spend vs response comparison", 
                        len(channels))

            x = np.arange(len(channels))

            # Plot spend changes
            spend_pct = ((df["optimal_spend"] / df["current_spend"]) - 1) * 100
            colors = [self.negative_color if x < 0 else self.positive_color 
                     for x in spend_pct]
            
            ax1.bar(x, spend_pct, color=colors, alpha=0.7)
            ax1.set_xticks(x)
            ax1.set_xticklabels(channels, rotation=45, ha="right")
            ax1.set_ylabel("Spend Change %")
            ax1.axhline(y=0, color="black", linestyle="-", alpha=0.2)
            ax1.grid(True, alpha=0.3)

            # Add value labels
            for i, v in enumerate(spend_pct):
                logger.debug("Channel %s spend change: %.1f%%", channels[i], v)
                ax1.text(i, v, f"{v:+.1f}%", ha="center", 
                        va="bottom" if v >= 0 else "top")

            # Plot response changes
            response_pct = ((df["optimal_response"] / df["current_response"]) - 1) * 100
            colors = [self.negative_color if x < 0 else self.positive_color 
                     for x in response_pct]
            
            ax2.bar(x, response_pct, color=colors, alpha=0.7)
            ax2.set_xticks(x)
            ax2.set_xticklabels(channels, rotation=45, ha="right")
            ax2.set_ylabel("Response Change %")
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.2)
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for i, v in enumerate(response_pct):
                logger.debug("Channel %s response change: %.1f%%", channels[i], v)
                ax2.text(i, v, f"{v:+.1f}%", ha="center", 
                        va="bottom" if v >= 0 else "top")

            plt.tight_layout()
            logger.info("Successfully created spend vs response plot")
            return fig
        except Exception as e:
            logger.error("Failed to create spend vs response plot: %s", str(e))
            raise

    def plot_summary_metrics(self) -> plt.Figure:
        """Plot summary metrics including ROI/CPA changes."""
        logger.info("Plotting summary metrics")

        if self.result is None:
            logger.error("No allocation results available for summary metrics plot")
            raise ValueError("No allocation results available. Call plot_all() first.")

        try:
            fig, ax = plt.subplots(figsize=self.fig_size)
            df = self.result.optimal_allocations
            channels = df["channel"].values
            logger.debug("Processing summary metrics for %d channels", len(channels))

            # Calculate ROI or CPA metrics
            metric_name = "ROI" if self.result.metrics.get("dep_var_type") == "revenue" else "CPA"
            logger.debug("Using metric type: %s", metric_name)

            if metric_name == "ROI":
                current_metric = df["current_response"] / df["current_spend"]
                optimal_metric = df["optimal_response"] / df["optimal_spend"]
            else:
                current_metric = df["current_spend"] / df["current_response"]
                optimal_metric = df["optimal_spend"] / df["optimal_response"]

            x = np.arange(len(channels))
            width = 0.35

            ax.bar(
                x - width / 2, current_metric, width, 
                label=f"Current {metric_name}", color=self.current_color, alpha=0.7
            )
            ax.bar(
                x + width / 2, optimal_metric, width, 
                label=f"Optimal {metric_name}", color=self.optimal_color, alpha=0.7
            )

            # Add value labels
            for i, (curr, opt) in enumerate(zip(current_metric, optimal_metric)):
                pct_change = ((opt / curr) - 1) * 100
                logger.debug("Channel %s %s change: %.1f%% (current: %.2f, optimal: %.2f)", 
                           channels[i], metric_name, pct_change, curr, opt)
                
                color = self.positive_color if pct_change >= 0 else self.negative_color
                ax.text(i, max(curr, opt), f"{pct_change:+.1f}%", 
                       ha="center", va="bottom", color=color)

            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Channel {metric_name} Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            logger.info("Successfully created summary metrics plot")
            return fig
        except Exception as e:
            logger.error("Failed to create summary metrics plot: %s", str(e))
            raise

    def save_plots(self, plots: Dict[str, plt.Figure], directory: str) -> None:
        """Save all plots to specified directory."""
        logger.info("Saving plots to directory: %s", directory)
        
        if not plots:
            logger.warning("No plots provided to save")
            return
            
        logger.debug("Preparing to save %d plots: %s", len(plots), list(plots.keys()))
        
        for name, fig in plots.items():
            try:
                filepath = f"{directory}/allocation_{name}.png"
                fig.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.debug("Successfully saved plot '%s' to %s", name, filepath)
                plt.close(fig)
            except Exception as e:
                logger.error("Failed to save plot '%s': %s", name, str(e))
                raise
        
        logger.info("Successfully saved all plots to directory")