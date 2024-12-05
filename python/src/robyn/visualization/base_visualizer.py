# pyre-strict

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from IPython.display import Image, display

# Configure logger
logger = logging.getLogger(__name__)


class BaseVisualizer(ABC):
    """
    Enhanced base class for all Robyn visualization components.
    Provides standardized plotting functionality and styling.
    """

    def __init__(self, style: str = "bmh"):
        """
        Initialize BaseVisualizer with standardized plot settings.

        Args:
            style: matplotlib style to use (default: "bmh")
        """
        logger.info("Initializing BaseVisualizer with style: %s", style)

        self.style = style
        # Standard figure sizes
        self.figure_sizes = {
            "default": (12, 8),
            "wide": (16, 8),
            "square": (10, 10),
            "tall": (8, 12),
            "small": (8, 6),
            "large": (15, 10),
            "medium": (10, 6)
        }

        # Standardized color schemes
        self.colors = {
            # Primary colors for main data series
            "primary": "#4688C7",  # Steel blue
            "secondary": "#FF9F1C",  # Orange
            "tertiary": "#37B067",  # Green
            # Status colors
            "positive": "#2ECC71",  # Green
            "negative": "#E74C3C",  # Red
            "neutral": "#95A5A6",  # Gray
            # Chart elements
            "grid": "#E0E0E0",  # Light gray for grid lines
            "baseline": "#CCCCCC",  # Medium gray for baseline/reference lines
            "annotation": "#666666",  # Dark gray for annotations
            # Channel-specific colors (for consistency across plots)
            "channels": {
                "facebook": "#3B5998",
                "search": "#4285F4",
                "display": "#34A853",
                "youtube": "#FF0000",
                "twitter": "#1DA1F2",
                "email": "#DB4437",
                "print": "#9C27B0",
                "tv": "#E91E63",
                "radio": "#795548",
                "ooh": "#607D8B",
            },
        }

        # Standard line styles
        self.line_styles = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-.",
        }

        # Standard markers
        self.markers = {
            "circle": "o",
            "square": "s",
            "triangle": "^",
            "diamond": "D",
            "plus": "+",
            "cross": "x",
            "star": "*",
        }

        # Font configurations
        self.fonts = {
            "family": "sans-serif",
            "sizes": {
                "title": 14,
                "subtitle": 12,
                "label": 11,
                "tick": 10,
                "annotation": 9,
                "legend": 10,
                "small": 8,
            },
        }

        # Common alpha values
        self.alpha = {
            "primary": 0.8,
            "secondary": 0.6,
            "grid": 0.3,
            "annotation": 0.7,
            "highlight": 0.9,
            "background": 0.2,
        }

        # Standard spacing
        self.spacing = {
            "tight_layout_pad": 1.05,
            "subplot_adjust_hspace": 0.4,
            "label_pad": 10,
            "title_pad": 20,
        }

        # Initialize plot tracking
        self.current_figure: Optional[plt.Figure] = None
        self.current_axes: Optional[Union[plt.Axes, np.ndarray]] = None

        # Apply default style and settings
        self._setup_plot_style()
        logger.info("BaseVisualizer initialization completed")

    def format_number(self, x: float, pos=None) -> str:
        """Format large numbers with K/M/B abbreviations.

        Args:
            x: Number to format
            pos: Position parameter (required by matplotlib formatter but not used)

        Returns:
            Formatted string representation of the number
        """
        try:
            if abs(x) >= 1e9:
                return f"{x/1e9:.1f}B"
            elif abs(x) >= 1e6:
                return f"{x/1e6:.1f}M"
            elif abs(x) >= 1e3:
                return f"{x/1e3:.1f}K"
            else:
                return f"{x:.1f}"
        except (TypeError, ValueError):
            return str(x)    

    def _setup_plot_style(self) -> None:
        """Configure default plotting style."""
        logger.debug("Setting up plot style")
        try:
            plt.style.use(self.style)

            plt.rcParams.update(
                {
                    # Figure settings
                    "figure.figsize": self.figure_sizes["default"],
                    "figure.facecolor": "white",
                    # Font settings
                    "font.family": self.fonts["family"],
                    "font.size": self.fonts["sizes"]["label"],
                    # Axes settings
                    "axes.grid": True,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.labelsize": self.fonts["sizes"]["label"],
                    "axes.titlesize": self.fonts["sizes"]["title"],
                    # Grid settings
                    "grid.alpha": self.alpha["grid"],
                    "grid.color": self.colors["grid"],
                    # Legend settings
                    "legend.fontsize": self.fonts["sizes"]["legend"],
                    "legend.framealpha": self.alpha["annotation"],
                    # Tick settings
                    "xtick.labelsize": self.fonts["sizes"]["tick"],
                    "ytick.labelsize": self.fonts["sizes"]["tick"],
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

    def _add_standardized_grid(
        self,
        ax: plt.Axes,
        axis: str = "both",
        alpha: Optional[float] = None,
        color: Optional[str] = None,
        linestyle: Optional[str] = None
    ) -> None:
        """Add standardized grid to plot."""
        ax.grid(
            True,
            axis=axis,
            alpha=alpha or self.alpha["grid"],
            color=color or self.colors["grid"],
            linestyle=linestyle or self.line_styles["solid"],
            zorder=0
        )
        ax.set_axisbelow(True)

    def _add_standardized_legend(
        self,
        ax: plt.Axes,
        title: Optional[str] = None,
        loc: str = "lower right",
        ncol: int = 1,
        handles: Optional[List] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Add standardized legend to plot.
        
        Args:
            ax: Matplotlib axes to add legend to
            title: Optional legend title
            loc: Legend location
            ncol: Number of columns in legend
            handles: Optional list of legend handles
            labels: Optional list of legend labels
        """
        legend_handles = handles if handles is not None else ax.get_legend_handles_labels()[0]
        legend_labels = labels if labels is not None else ax.get_legend_handles_labels()[1]

        legend = ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            title=title,
            loc=loc,
            ncol=ncol,
            fontsize=self.fonts["sizes"]["legend"],
            framealpha=self.alpha["annotation"],
            title_fontsize=self.fonts["sizes"]["subtitle"]
        )
        if legend:
            legend.get_frame().set_linewidth(0.5)
            legend.get_frame().set_edgecolor(self.colors["grid"])

    def _set_standardized_labels(
        self,
        ax: plt.Axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """Set standardized labels for plot."""
        if xlabel:
            ax.set_xlabel(
                xlabel,
                fontsize=self.fonts["sizes"]["label"],
                labelpad=self.spacing["label_pad"]
            )
        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontsize=self.fonts["sizes"]["label"],
                labelpad=self.spacing["label_pad"]
            )
        if title:
            ax.set_title(
                title,
                fontsize=self.fonts["sizes"]["title"],
                pad=self.spacing["title_pad"]
            )

    def _format_standardized_ticks(
        self,
        ax: plt.Axes,
        x_rotation: int = 0,
        y_rotation: int = 0
    ) -> None:
        """Format tick labels with standardized styling."""
        ax.tick_params(
            axis='both',
            labelsize=self.fonts["sizes"]["tick"]
        )
        plt.setp(
            ax.get_xticklabels(),
            rotation=x_rotation,
            ha='right' if x_rotation > 0 else 'center'
        )
        plt.setp(
            ax.get_yticklabels(),
            rotation=y_rotation,
            va='center'
        )

    def _set_standardized_spines(self, ax: plt.Axes, spines: List[str] = None) -> None:
        """Configure plot spines with standardized styling."""
        if spines is None:
            spines = ['top', 'right']
        for spine in spines:
            ax.spines[spine].set_visible(False)    

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
