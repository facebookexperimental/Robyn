from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class RobynVisualizer:
    def __init__(self):
        pass

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

    def plot_moo_distribution(self, output_models: Dict[str, Any]) -> plt.Figure:
        """
        Plot MOO (multi-objective optimization) distribution.

        Args:
            output_models (Dict[str, Any]): The output models data.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def plot_moo_cloud(self, output_models: Dict[str, Any]) -> plt.Figure:
        """
        Plot MOO (multi-objective optimization) cloud.

        Args:
            output_models (Dict[str, Any]): The output models data.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def plot_ts_validation(self, output_models: Dict[str, Any]) -> plt.Figure:
        """
        Plot time-series validation.

        Args:
            output_models (Dict[str, Any]): The output models data.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def plot_onepager(self, input_collect: Dict[str, Any], output_collect: Dict[str, Any], select_model: str) -> Dict[str, plt.Figure]:
        """
        Generate one-pager plots for a selected model.

        Args:
            input_collect (Dict[str, Any]): The input collection data.
            output_collect (Dict[str, Any]): The output collection data.
            select_model (str): The selected model identifier.

        Returns:
            Dict[str, plt.Figure]: A dictionary of generated figures.
        """
        pass

    def plot_allocator(self, allocator_collect: Dict[str, Any]) -> plt.Figure:
        """
        Plot allocator's output.

        Args:
            allocator_collect (Dict[str, Any]): The allocator collection data.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def plot_response(self, response_data: Dict[str, Any]) -> plt.Figure:
        """
        Plot response curves.

        Args:
            response_data (Dict[str, Any]): The response data.

        Returns:
            plt.Figure: The generated figure.
        """
        pass
