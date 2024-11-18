from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class BaseVisualizer:
    def __init__(self, style: str = "bmh"):
        self.style = style
        self.default_figsize = (12, 8)
        self.colors = {
            "primary": None,
            "secondary": None,
            "positive": None,
            "negative": None,
            "neutral": None,
        }
        self.font_sizes = {
            "title": None,
            "label": None,
            "tick": None,
            "annotation": None,
        }
        self.current_figure = None
        self.current_axes = None
        self._setup_plot_style()

    def _setup_plot_style(self) -> None:
        plt.style.use(self.style)
        plt.rcParams.update(
            {
                "figure.figsize": self.default_figsize,
                "axes.grid": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.labelsize": self.font_sizes["label"],
                "xtick.labelsize": self.font_sizes["tick"],
                "ytick.labelsize": self.font_sizes["tick"],
            }
        )
        sns.set_style(self.style)

    def save_plot(
        self, filename: Union[str, Path], dpi: int = 300, cleanup: bool = True
    ) -> None:
        if self.current_figure is None:
            raise ValueError("No figure to save.")

        # Ensure filename is a Path object
        filename = Path(filename)

        # Create parent directory if it doesn't exist
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        # Save the figure
        self.current_figure.savefig(filename, dpi=dpi, bbox_inches="tight")

        # Clean up if specified
        if cleanup:
            self.cleanup()

    def cleanup(self) -> None:
        if self.current_figure is not None:
            plt.close(self.current_figure)
        self.current_figure = None
        self.current_axes = None
