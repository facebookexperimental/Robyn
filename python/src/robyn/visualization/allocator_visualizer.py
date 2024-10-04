from typing import Dict, Any
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer

class AllocatorVisualizer(BaseVisualizer):
    def __init__(self, allocator_data: Dict[str, Any]):
        super().__init__()
        self.allocator_data = allocator_data

    def plot_allocator(self) -> plt.Figure:
        """
        Plot allocator's output.

        Returns:
            plt.Figure: The generated figure.
        """
        pass
