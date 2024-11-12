from typing import Dict, Optional, Tuple, Union, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

# Configure logger
logger = logging.getLogger(__name__)

class BaseVisualizer:
    """
    Base class for all Robyn visualization components.
    Provides common plotting functionality and styling.
    """

    def __init__(self, style: str = "bmh"):
        """
        Initialize BaseVisualizer with common plot settings.

        Args:
            style: matplotlib style to use (default: "bmh")
        """
        logger.info("Initializing BaseVisualizer with style: %s", style)
        
        # Store style settings
        self.style = style
        self.default_figsize = (12, 8)

        # Enhanced color schemes
        self.colors = {
            "primary": "#4688C7",  # Steel blue
            "secondary": "#FF9F1C",  # Orange
            "positive": "#2ECC71",  # Green
            "negative": "#E74C3C",  # Red
            "neutral": "#95A5A6",  # Gray
            "current": "lightgray",  # For current values
            "optimal": "#4688C7",  # For optimal values
            "grid": "#E0E0E0",  # For grid lines
        }
        logger.debug("Color scheme initialized: %s", self.colors)

        # Plot settings
        self.font_sizes = {"title": 14, "subtitle": 12, "label": 12, "tick": 10, "annotation": 9, "legend": 10}
        logger.debug("Font sizes configured: %s", self.font_sizes)

        # Default alpha values
        self.alpha = {"primary": 0.7, "secondary": 0.5, "grid": 0.3, "annotation": 0.7}

        # Default spacing
        self.spacing = {"tight_layout_pad": 1.05, "subplot_adjust_hspace": 0.4}

        # Initialize plot tracking
        self.current_figure: Optional[plt.Figure] = None
        self.current_axes: Optional[Union[plt.Axes, np.ndarray]] = None

        # Apply default style
        self._setup_plot_style()
        logger.info("BaseVisualizer initialization completed")

    def _setup_plot_style(self) -> None:
        """Configure default plotting style."""
        logger.debug("Setting up plot style with style: %s", self.style)
        try:
            plt.style.use(self.style)

            plt.rcParams.update(
                {
                    "figure.figsize": self.default_figsize,
                    "axes.grid": True,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "font.size": self.font_sizes["label"],
                    "grid.alpha": self.alpha["grid"],
                    "grid.color": self.colors["grid"],
                }
            )
            logger.debug("Plot style parameters updated successfully")
        except Exception as e:
            logger.error("Failed to setup plot style: %s", str(e))
            raise

    def create_figure(
        self, nrows: int = 1, ncols: int = 1, figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create and track a new figure.

        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            figsize: Optional custom figure size

        Returns:
            Tuple of (figure, axes)
        """
        logger.info("Creating new figure with dimensions %dx%d", nrows, ncols)
        figsize = figsize or self.default_figsize
        logger.debug("Using figure size: %s", figsize)

        try:
            self.current_figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            if nrows == ncols == 1:
                self.current_axes = axes
            else:
                self.current_axes = np.array(axes)
            logger.debug("Figure created successfully")
            return self.current_figure, axes
        except Exception as e:
            logger.error("Failed to create figure: %s", str(e))
            raise

    def setup_axis(
        self,
        ax: plt.Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xticks: Optional[List] = None,
        yticks: Optional[List] = None,
        xticklabels: Optional[List] = None,
        yticklabels: Optional[List] = None,
        rotation: int = 0,
    ) -> None:
        """
        Apply common axis setup and styling.

        Args:
            ax: Matplotlib axes to style
            title: Optional axis title
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
            xticks: Optional list of x-axis tick positions
            yticks: Optional list of y-axis tick positions
            xticklabels: Optional list of x-axis tick labels
            yticklabels: Optional list of y-axis tick labels
            rotation: Rotation angle for tick labels
        """
        logger.debug("Setting up axis with title: %s, xlabel: %s, ylabel: %s", title, xlabel, ylabel)
        
        try:
            if title:
                ax.set_title(title, fontsize=self.font_sizes["title"])
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=self.font_sizes["label"])
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=self.font_sizes["label"])

            if xticks is not None:
                logger.debug("Setting x-ticks: %s", xticks)
                ax.set_xticks(xticks)
            if yticks is not None:
                logger.debug("Setting y-ticks: %s", yticks)
                ax.set_yticks(yticks)

            if xticklabels is not None:
                logger.debug("Setting x-tick labels with rotation: %d", rotation)
                ax.set_xticklabels(xticklabels, rotation=rotation, fontsize=self.font_sizes["tick"])
            if yticklabels is not None:
                logger.debug("Setting y-tick labels")
                ax.set_yticklabels(yticklabels, fontsize=self.font_sizes["tick"])

            ax.tick_params(labelsize=self.font_sizes["tick"])
            ax.grid(True, alpha=self.alpha["grid"], color=self.colors["grid"])
            logger.debug("Axis setup completed successfully")
        except Exception as e:
            logger.error("Failed to setup axis: %s", str(e))
            raise

    def add_percentage_annotation(
        self, ax: plt.Axes, x: float, y: float, percentage: float, va: str = "bottom", ha: str = "center"
    ) -> None:
        """
        Add a percentage change annotation to the plot.

        Args:
            ax: Matplotlib axes to annotate
            x: X-coordinate for annotation
            y: Y-coordinate for annotation
            percentage: Percentage value to display
            va: Vertical alignment
            ha: Horizontal alignment
        """
        logger.debug("Adding percentage annotation at (x=%f, y=%f) with value: %f%%", x, y, percentage)
        try:
            color = self.colors["positive"] if percentage >= 0 else self.colors["negative"]
            ax.text(
                x,
                y,
                f"{percentage:.1f}%",
                color=color,
                va=va,
                ha=ha,
                fontsize=self.font_sizes["annotation"],
                alpha=self.alpha["annotation"],
            )
            logger.debug("Percentage annotation added successfully")
        except Exception as e:
            logger.error("Failed to add percentage annotation: %s", str(e))
            raise

    def add_legend(self, ax: plt.Axes, loc: str = "best", title: Optional[str] = None) -> None:
        """
        Add a formatted legend to the plot.

        Args:
            ax: Matplotlib axes to add legend to
            loc: Legend location
            title: Optional legend title
        """
        logger.debug("Adding legend with location: %s and title: %s", loc, title)
        try:
            legend = ax.legend(fontsize=self.font_sizes["legend"], loc=loc, framealpha=self.alpha["annotation"])
            if title:
                legend.set_title(title, prop={"size": self.font_sizes["legend"]})
            logger.debug("Legend added successfully")
        except Exception as e:
            logger.error("Failed to add legend: %s", str(e))
            raise

    def finalize_figure(self, tight_layout: bool = True, adjust_spacing: bool = False) -> None:
        """
        Apply final formatting to the current figure.

        Args:
            tight_layout: Whether to apply tight_layout
            adjust_spacing: Whether to adjust subplot spacing
        """
        logger.info("Finalizing figure with tight_layout=%s, adjust_spacing=%s", tight_layout, adjust_spacing)
        
        if self.current_figure is None:
            logger.warning("No current figure to finalize")
            return

        try:
            if tight_layout:
                self.current_figure.tight_layout(pad=self.spacing["tight_layout_pad"])
            if adjust_spacing:
                self.current_figure.subplots_adjust(hspace=self.spacing["subplot_adjust_hspace"])
            logger.debug("Figure finalization completed successfully")
        except Exception as e:
            logger.error("Failed to finalize figure: %s", str(e))
            raise

    def save_plot(self, filename: Union[str, Path], dpi: int = 300, cleanup: bool = True) -> None:
        """
        Save the current plot to a file.

        Args:
            filename: Path to save the plot
            dpi: Resolution for saved plot
            cleanup: Whether to close the plot after saving
        """
        logger.info("Saving plot to: %s with DPI: %d", filename, dpi)
        
        if self.current_figure is None:
            error_msg = "No current figure to save"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory structure for: %s", filepath.parent)

            self.current_figure.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
            logger.info("Plot saved successfully to: %s", filepath)

            if cleanup:
                logger.debug("Cleaning up after save")
                self.cleanup()
        except Exception as e:
            logger.error("Failed to save plot to %s: %s", filename, str(e))
            raise

    def cleanup(self) -> None:
        """Close the current plot and clear matplotlib memory."""
        logger.debug("Performing cleanup")
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
            self.current_axes = None
            logger.debug("Cleanup completed successfully")
        else:
            logger.debug("No figure to clean up")