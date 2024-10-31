import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseVisualizer:
    """Base class for all visualization components in Robyn."""

    def __init__(self):
        """Initialize base visualizer with common settings."""
        # Set default style
        plt.style.use("bmh")

        # Default figure settings
        self.fig_size = (12, 8)
        plt.rcParams["figure.figsize"] = self.fig_size
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False

        # Common color schemes
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))
        self.color_scheme = {
            "primary": "#4688C7",  # Steel blue
            "secondary": "#95A5A6",  # Gray
            "positive": "#2ECC71",  # Green
            "negative": "#E74C3C",  # Red
            "neutral": "lightgray",
            "highlight": "#F39C12",  # Orange
            "accent1": "#9B59B6",  # Purple
            "accent2": "#3498DB",  # Light blue
        }

    def setup_figure(self, figsize: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create and setup a new figure with common settings.

        Args:
            figsize: Optional tuple of (width, height) for the figure

        Returns:
            Tuple of (Figure, Axes)
        """
        fig, ax = plt.subplots(figsize=figsize or self.fig_size)
        self._setup_axis(ax)
        return fig, ax

    def _setup_axis(self, ax: plt.Axes) -> None:
        """Apply common axis settings.

        Args:
            ax: Matplotlib axis to configure
        """
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def setup_subplots(
        self, nrows: int, ncols: int, figsize: Optional[Tuple[float, float]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create and setup a figure with multiple subplots.

        Args:
            nrows: Number of rows
            ncols: Number of columns
            figsize: Optional figure size

        Returns:
            Tuple of (Figure, array of Axes)
        """
        fig_width = figsize[0] if figsize else self.fig_size[0]
        fig_height = figsize[1] if figsize else self.fig_size[1]

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width * ncols, fig_height * nrows))

        # Handle single subplot case
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(-1, 1)

        # Setup each axis
        for ax in axes.flat:
            self._setup_axis(ax)

        return fig, axes

    def add_value_labels(
        self,
        ax: plt.Axes,
        values: np.ndarray,
        positions: np.ndarray,
        offset: float = 0.0,
        format_str: str = "{:+.1f}%",
    ) -> None:
        """Add value labels to bars or points.

        Args:
            ax: The axis to add labels to
            values: The values to display
            positions: The x-positions of the values
            offset: Vertical offset for label placement
            format_str: String format for the labels
        """
        for pos, val in zip(positions, values):
            color = self.color_scheme["positive"] if val >= 0 else self.color_scheme["negative"]
            ax.text(
                pos, val + offset, format_str.format(val), ha="center", va="bottom" if val >= 0 else "top", color=color
            )

    def save_figure(self, fig: plt.Figure, filename: str, directory: str, dpi: int = 300) -> None:
        """Save figure to file with standard settings.

        Args:
            fig: Figure to save
            filename: Name of the file
            directory: Directory to save to
            dpi: Resolution for the saved figure
        """
        fig.savefig(f"{directory}/{filename}", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)

    def set_plot_style(self, style_dict: Optional[Dict[str, Any]] = None) -> None:
        """Update plot style settings.

        Args:
            style_dict: Dictionary of matplotlib rcParams to update
        """
        if style_dict is None:
            style_dict = {}

        # Default style updates
        defaults = {
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }

        # Update defaults with provided values
        defaults.update(style_dict)
        plt.rcParams.update(defaults)

    def clear_all_plots(self) -> None:
        """Close all open figure windows."""
        plt.close("all")
