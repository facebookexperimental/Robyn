from typing import Dict, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


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
        }

        # Plot settings
        self.font_sizes = {"title": 14, "label": 12, "tick": 10, "annotation": 9}

        # Default alpha values
        self.alpha = {"primary": 0.7, "secondary": 0.5, "grid": 0.3}

        # Initialize plot tracking
        self.current_figure: Optional[plt.Figure] = None
        self.current_axes: Optional[plt.Axes] = None

        # Apply default style
        self._setup_plot_style()

    def _setup_plot_style(self) -> None:
        """Configure default plotting style."""
        plt.style.use(self.style)

        plt.rcParams.update(
            {
                "figure.figsize": self.default_figsize,
                "axes.grid": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": self.font_sizes["label"],
            }
        )

    def create_figure(
        self, nrows: int = 1, ncols: int = 1, figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """Create and track a new figure."""
        figsize = figsize or self.default_figsize
        self.current_figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nrows == ncols == 1:
            self.current_axes = axes
        return self.current_figure, axes

    def save_plot(self, filename: Union[str, Path], dpi: int = 300, cleanup: bool = True) -> None:
        """Save the current plot to a file."""
        if self.current_figure is None:
            raise ValueError("No current figure to save")

        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.current_figure.savefig(filepath, dpi=dpi, bbox_inches="tight")

        if cleanup:
            self.cleanup()

    def cleanup(self) -> None:
        """Close the current plot and clear matplotlib memory."""
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
            self.current_axes = None
