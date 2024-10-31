from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from robyn.data.entities.hyperparameters import AdstockType
from robyn.data.entities.mmmdata import MMMData


from .pareto_visualizer import ParetoVisualizer
from .cluster_visualizer import ClusterVisualizer
from .input_visualizer import InputVisualizer
from .response_visualizer import ResponseVisualizer
from .transformation_visualizer import TransformationVisualizer

class OnePagerVisualizer:
    """
        Class for generating comprehensive one-page visualization reports
        combining multiple visualizer outputs.
    """
    
    def __init__(
        self,
        pareto_result: ParetoResult,
        clustered_result: Optional[ClusteredResult] = None,
        adstock: Optional[AdstockType] = None,
        mmm_data: Optional[MMMData] = None
    ):
        """
        Initialize OnePager with required results objects.
        
        Args:
            pareto_result: Results from Pareto optimization
            clustered_result: Optional clustering results
            adstock: Optional adstock configuration
        """
        self.pareto_result = pareto_result
        self.clustered_result = clustered_result
        self.adstock = adstock
        self.mmm_data = mmm_data
        
        # Initialize visualizers
        self.pareto_viz = ParetoVisualizer(pareto_result, adstock) if adstock else None
        self.cluster_viz = ClusterVisualizer(pareto_result, clustered_result) if clustered_result else None
        self.input_viz = InputVisualizer(pareto_result)
        self.response = ResponseVisualizer(pareto_result, mmm_data)
        self.transfor_viz = TransformationVisualizer(pareto_result)

    
    def generate_one_pager(
        self,
        plots: Optional[List[str]] = None,
        figsize: tuple = (20, 15)
    ) -> plt.Figure:
        """
        Generate a one-page report with multiple visualization plots.
        
        Args:
            plots: List of plot names to include. If None, includes all available plots.
                Valid options: ['waterfall', 'fitted_vs_actual', 'diagnostic', 
                'immediate_vs_carryover', 'adstock_rate', 'bootstrap_confidence',
                'spend_exposure']
            figsize: Figure size in inches (width, height)
            
        Returns:
            plt.Figure: Combined figure with all requested plots
        """

        fig = plt.figure(figsize=figsize)
        return fig