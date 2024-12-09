# pyre-strict

import logging

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
from IPython.display import Image, display

import matplotlib.pyplot as plt
import numpy as np
import base64
import io

logger = logging.getLogger(__name__)


class BaseVisualizer(ABC):
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
        logger.debug("Initializing BaseVisualizer with style: %s", style)

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
        self.font_sizes = {
            "title": 14,
            "subtitle": 12,
            "label": 12,
            "tick": 10,
            "annotation": 9,
            "legend": 10,
        }
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
        logger.debug("BaseVisualizer initialization completed")

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
            self.current_figure, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=figsize
            )
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
        logger.debug(
            "Setting up axis with title: %s, xlabel: %s, ylabel: %s",
            title,
            xlabel,
            ylabel,
        )

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
                ax.set_xticklabels(
                    xticklabels, rotation=rotation, fontsize=self.font_sizes["tick"]
                )
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
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        percentage: float,
        va: str = "bottom",
        ha: str = "center",
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
        logger.debug(
            "Adding percentage annotation at (x=%f, y=%f) with value: %f%%",
            x,
            y,
            percentage,
        )
        try:
            color = (
                self.colors["positive"] if percentage >= 0 else self.colors["negative"]
            )
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

    def add_legend(
        self, ax: plt.Axes, loc: str = "best", title: Optional[str] = None
    ) -> None:
        """
        Add a formatted legend to the plot.

        Args:
            ax: Matplotlib axes to add legend to
            loc: Legend location
            title: Optional legend title
        """
        logger.debug("Adding legend with location: %s and title: %s", loc, title)
        try:
            legend = ax.legend(
                fontsize=self.font_sizes["legend"],
                loc=loc,
                framealpha=self.alpha["annotation"],
            )
            if title:
                legend.set_title(title, prop={"size": self.font_sizes["legend"]})
            logger.debug("Legend added successfully")
        except Exception as e:
            logger.error("Failed to add legend: %s", str(e))
            raise

    def finalize_figure(
        self, tight_layout: bool = True, adjust_spacing: bool = False
    ) -> None:
        """
        Apply final formatting to the current figure.

        Args:
            tight_layout: Whether to apply tight_layout
            adjust_spacing: Whether to adjust subplot spacing
        """
        logger.info(
            "Finalizing figure with tight_layout=%s, adjust_spacing=%s",
            tight_layout,
            adjust_spacing,
        )

        if self.current_figure is None:
            logger.warning("No current figure to finalize")
            return

        try:
            if tight_layout:
                self.current_figure.tight_layout(pad=self.spacing["tight_layout_pad"])
            if adjust_spacing:
                self.current_figure.subplots_adjust(
                    hspace=self.spacing["subplot_adjust_hspace"]
                )
            logger.debug("Figure finalization completed successfully")
        except Exception as e:
            logger.error("Failed to finalize figure: %s", str(e))
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

    @staticmethod
    def export_plots_base64(
        export_location: Union[str, Path], plots: Dict[str, str], dpi: int = 300
    ) -> None:
        logger.info("Exporting base64 plots to: %s", export_location)
        export_path = Path(export_location)
        export_path.mkdir(parents=True, exist_ok=True)

        for plot_name, base64_str in plots.items():
            filename = export_path / f"{plot_name}.png"
            logger.debug("Saving base64 plot: %s to %s", plot_name, filename)
            try:
                image_data = base64.b64decode(base64_str)
                with open(filename, "wb") as f:
                    f.write(image_data)
                logger.info("Base64 plot %s saved successfully", plot_name)
            except Exception as e:
                logger.error("Failed to save base64 plot %s: %s", plot_name, str(e))
                raise
        pass

    @staticmethod
    def export_plots_fig(
        export_location: Union[str, Path], plots: Dict[str, plt.Figure], dpi: int = 300
    ) -> None:
        """
        Save multiple plots to the specified location.

        Args:
            export_location: Directory to save the plots
            plots: Dictionary of plot names and their corresponding figures
            dpi: Resolution for saved plots
        """
        logger.info("Saving multiple plots to: %s", export_location)
        export_path = Path(export_location)
        export_path.mkdir(parents=True, exist_ok=True)

        for plot_name, fig in plots.items():
            filename = export_path / f"{plot_name}.png"
            logger.debug("Saving plot: %s to %s", plot_name, filename)
            try:
                fig.savefig(
                    filename,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )
                logger.info("Plot %s saved successfully", plot_name)
            except Exception as e:
                logger.error("Failed to save plot %s: %s", plot_name, str(e))
                raise

    @abstractmethod
    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> None:
        pass

    @staticmethod
    def display_plots(plot_collect: Dict[str, plt.Figure]) -> None:
        """Display the plot."""
        for plot_name, fig in plot_collect.items():
            display(fig)

    @staticmethod
    def _display_base64_image(base64_image: str):
        """Helper method to display a base64-encoded image."""
        display(Image(data=base64.b64decode(base64_image)))

    def convert_plot_to_base64(self, fig: plt.Figure) -> str:
        logger.debug("Converting plot to base64")
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            logger.debug("Successfully converted plot to base64")
            return graphic.decode("utf-8")
        except Exception as e:
            logger.error("Failed to convert plot to base64: %s", str(e), exc_info=True)
            raise
