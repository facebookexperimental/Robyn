from typing import Dict, Any
import matplotlib.pyplot as plt
from robyn.visualization.input_visualizer import InputVisualizer
from robyn.visualization.model_convergence_visualizer import ModelVisualizer
from robyn.visualization.allocator_visualizer import AllocatorVisualizer
from robyn.visualization.response_visualizer import ResponseVisualizer

class RobynVisualizer:
    def __init__(self):
        self.input_visualizer = None
        self.model_visualizer = None
        self.allocator_visualizer = None
        self.response_visualizer = None

    def set_input_data(self, input_data: Dict[str, Any]):
        self.input_visualizer = InputVisualizer(input_data)

    def set_model_data(self, model_data: Dict[str, Any]):
        self.model_visualizer = ModelVisualizer(model_data)

    def set_allocator_data(self, allocator_data: Dict[str, Any]):
        self.allocator_visualizer = AllocatorVisualizer(allocator_data)

    def set_response_data(self, response_data: Dict[str, Any]):
        self.response_visualizer = ResponseVisualizer(response_data)

    def plot_adstock(self) -> plt.Figure:
        return self.input_visualizer.plot_adstock()

    def plot_saturation(self) -> plt.Figure:
        return self.input_visualizer.plot_saturation()

    def plot_moo_distribution(self) -> plt.Figure:
        return self.model_visualizer.plot_moo_distribution()

    def plot_moo_cloud(self) -> plt.Figure:
        return self.model_visualizer.plot_moo_cloud()

    def plot_ts_validation(self) -> plt.Figure:
        return self.model_visualizer.plot_ts_validation()

    def plot_onepager(self, input_collect: Dict[str, Any], output_collect: Dict[str, Any], select_model: str) -> Dict[str, plt.Figure]:
        return self.model_visualizer.plot_onepager(input_collect, output_collect, select_model)

    def plot_allocator(self) -> plt.Figure:
        return self.allocator_visualizer.plot_allocator()

    def plot_response(self) -> plt.Figure:
        return self.response_visualizer.plot_response()

    def plot_marginal_response(self) -> plt.Figure:
        return self.response_visualizer.plot_marginal_response()
