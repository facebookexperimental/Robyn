from typing import Dict, Any
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer

class InputVisualizer(BaseVisualizer):
    def __init__(self, input_data: Dict[str, Any]):
        super().__init__()
        self.input_data = input_data

    def plot_adstock(self, plot: bool = False) -> plt.Figure:
        """
        Create example plots for adstock hyperparameters.

        Args:
            plot (bool): Whether to display the plot.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def plot_saturation(self, plot: bool = False) -> plt.Figure:
        """
        Create example plots for saturation hyperparameters.

        Args:
            plot (bool): Whether to display the plot.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def plot_spend_exposure_fit(self) -> Dict[str, plt.Figure]:
        """
        Check spend exposure fit if available.

        Returns:
            Dict[str, plt.Figure]: A dictionary of generated figures.
        """
        pass