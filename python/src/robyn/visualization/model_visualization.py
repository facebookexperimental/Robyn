# model_visualization.py

import io
import binascii
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import display
import warnings

from robyn.modeling.entities.modeloutputs import ModelOutputs


class ModelVisualizer:
    """
    A class for creating various plots related to model outputs in the Robyn framework.
    """

    def __init__(self, model_outputs: ModelOutputs):
        """
        Initialize the ModelVisualizer class.

        Args:
            model_outputs (ModelOutputs): The outputs from the model run.
        """
        self.model_outputs = model_outputs

    def plot_outputgraphs(self, graphtype: str, max_size: Tuple[int, int] = (1000, 1500)) -> None:
        """
        Plot and display output graphs based on the model results.

        Args:
            graphtype (str): The type of graph to plot. Options are 'moo_distrb_plot',
                             'moo_cloud_plot', 'ts_validation_plot'.
            max_size (Tuple[int, int]): The maximum size of the rendered images.
                                        Defaults to (1000, 1500).

        Returns:
            None. The function renders the plots and displays them using the `display()` function.
        """
        if graphtype == "moo_distrb_plot":
            self._plot_moo_distrb(max_size)
        elif graphtype == "moo_cloud_plot":
            self._plot_moo_cloud(max_size)
        elif graphtype == "ts_validation_plot":
            self._plot_ts_validation(max_size)
        else:
            warnings.warn("Graphtype does not exist")

    def _plot_moo_distrb(self, max_size: Tuple[int, int]) -> None:
        """Plot the MOO distribution plot."""
        if "moo_distrb_plot" in self.model_outputs.convergence:
            image_data = binascii.unhexlify("".join(self.model_outputs.convergence["moo_distrb_plot"]))
            self._display_image(image_data, max_size)
        else:
            warnings.warn("MOO distribution plot data not available")

    def _plot_moo_cloud(self, max_size: Tuple[int, int]) -> None:
        """Plot the MOO cloud plot."""
        if "moo_cloud_plot" in self.model_outputs.convergence:
            image_data = binascii.unhexlify("".join(self.model_outputs.convergence["moo_cloud_plot"]))
            self._display_image(image_data, max_size)
        else:
            warnings.warn("MOO cloud plot data not available")

    def _plot_ts_validation(self, max_size: Tuple[int, int]) -> None:
        """Plot the time series validation plot."""
        if self.model_outputs.ts_validation_plot:
            image_data = binascii.unhexlify("".join(self.model_outputs.ts_validation_plot))
            self._display_image(image_data, max_size)
        else:
            warnings.warn("Time series validation plot data not available")

    @staticmethod
    def _display_image(image_data: bytes, max_size: Tuple[int, int]) -> None:
        """Display an image from binary data."""
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        display(image)

    def plot_hyperparameter_distribution(self, hyperparameter: str) -> None:
        """
        Plot the distribution of a specific hyperparameter across all trials.

        Args:
            hyperparameter (str): The name of the hyperparameter to plot.
        """
        values = [
            trial.result_hyp_param[hyperparameter].values[0]
            for trial in self.model_outputs.trials
            if hyperparameter in trial.result_hyp_param
        ]

        if not values:
            warnings.warn(f"No data available for hyperparameter: {hyperparameter}")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True)
        plt.xlabel(hyperparameter)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {hyperparameter}")
        plt.show()

    # Add more visualization methods as needed...
