from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from robyn.data.entities.hyperparameters import AdstockType
from robyn.data.entities.mmmdata import MMMData

from robyn.visualization.pareto_visualizer import ParetoVisualizer
from robyn.visualization.cluster_visualizer import ClusterVisualizer
from python.src.robyn.visualization.media_response_visualizer import MediaResponseVisualizer
from robyn.visualization.response_visualizer import ResponseVisualizer
from robyn.visualization.transformation_visualizer import TransformationVisualizer

class OnePagerReporter:
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

        """
            TODO: add exception handling.
        """
        self.pareto_result = pareto_result
        self.clustered_result = clustered_result
        self.adstock = adstock
        self.mmm_data = mmm_data
        
        # Initialize visualizers
        self.pareto_viz = ParetoVisualizer(pareto_result, adstock) if adstock else None
        self.cluster_viz = ClusterVisualizer(pareto_result, clustered_result) if clustered_result else None
        self.input_viz = MediaResponseVisualizer(pareto_result)
        self.response = ResponseVisualizer(pareto_result, mmm_data)
        self.transfor_viz = TransformationVisualizer(pareto_result)

    
    def _setup_grid(self, n_plots: int, figsize: tuple) -> Tuple[plt.Figure, GridSpec]:
        """Set up the grid layout for the one pager."""
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        
        # Create grid with 4 rows and 2 columns
        gs = GridSpec(4, 2, figure=fig)
        return fig, gs
        
    def _get_model_info(self) -> Dict[str, str]:
        """Get model performance metrics and information."""
        # Get first model's data
        model_data = next(iter(self.pareto_result.plot_data_collect.values()))
        
        # Extract performance metrics
        metrics = {
            'rsq_train': f"{model_data['plot5data'].get('rsq', 0):.4f}",
            'nrmse': f"{model_data.get('nrmse', 0):.4f}",
            'decomp_rssd': f"{model_data.get('decomp.rssd', 0):.4f}"
        }
        
        if hasattr(self.pareto_result, 'mape'):
            metrics['mape'] = f"{self.pareto_result.mape:.4f}"
            
        return metrics

    def generate_one_pager(
        self,
        plots: Optional[List[str]] = None,
        figsize: tuple = (20, 15)
    ) -> plt.Figure:
        """
        Generate a one-page report with multiple visualization plots.
        
        Args:
            plots: List of plot names to include. If None, includes all default plots.
            figsize: Figure size in inches (width, height)
            
        Returns:
            plt.Figure: Combined figure with all requested plots
        """
        # Set plot style
        plt.style.use('seaborn-whitegrid')
        
        # Use default plots if none specified
        plots = plots or self.default_plots
        
        # Create figure and grid
        fig, gs = self._setup_grid(len(plots), figsize)
        
        # Get model information
        model_info = self._get_model_info()
        
        # Define plot positions and functions
        plot_config = {
            'waterfall': {
                'position': gs[0, 0],
                'function': lambda: self.pareto_viz.generate_waterfall() if self.pareto_viz else None,
                'title': 'Response Decomposition Waterfall'
            },
            'fitted_vs_actual': {
                'position': gs[0, 1],
                'function': lambda: self.pareto_viz.generate_fitted_vs_actual() if self.pareto_viz else None,
                'title': 'Actual vs. Predicted Response'
            },
            'spend_effect': {
                'position': gs[1, 0],
                'function': lambda: self.transform_viz.generate_spend_effect_comparison(),
                'title': 'Media Spend vs Effect Share'
            },
            'bootstrap': {
                'position': gs[1, 1],
                'function': lambda: self.cluster_viz.generate_bootstrap_confidence() if self.cluster_viz else None,
                'title': 'Bootstrapped Performance Metrics'
            },
            'adstock': {
                'position': gs[2, 0],
                'function': lambda: self.pareto_viz.generate_adstock_rate() if self.pareto_viz else None,
                'title': 'Adstock Rates by Channel'
            },
            'immediate_carryover': {
                'position': gs[2, 1],
                'function': lambda: self.pareto_viz.generate_immediate_vs_carryover() if self.pareto_viz else None,
                'title': 'Immediate vs Carryover Effects'
            },
            'response_curves': {
                'position': gs[3, 0],
                'function': lambda: self.response.generate_response_curves(),
                'title': 'Response Curves by Channel'
            },
            'diagnostic': {
                'position': gs[3, 1],
                'function': lambda: self.pareto_viz.generate_diagnostic_plot() if self.pareto_viz else None,
                'title': 'Model Diagnostics'
            }
        }
        
        # Generate each requested plot
        for plot_name in plots:
            if plot_name in plot_config:
                config = plot_config[plot_name]
                
                # Create subplot
                ax = fig.add_subplot(config['position'])
                
                # Generate and copy plot content
                plot_fig = config['function']()
                if plot_fig is not None:
                    # Copy content from the generated figure to our subplot
                    temp_ax = plot_fig.gca()
                    for item in temp_ax.get_children():
                        try:
                            plot_copy = item.copy()
                            ax.add_artist(plot_copy)
                        except:
                            continue
                            
                    # Copy axis settings
                    ax.set_xlim(temp_ax.get_xlim())
                    ax.set_ylim(temp_ax.get_ylim())
                    ax.set_xlabel(temp_ax.get_xlabel())
                    ax.set_ylabel(temp_ax.get_ylabel())
                    
                    plt.close(plot_fig)
                
                ax.set_title(config['title'])
        
        # Add overall title with model metrics
        metrics_text = (
            f"Model Performance Metrics - "
            f"R²: {model_info['rsq_train']} | "
            f"NRMSE: {model_info['nrmse']} | "
            f"DECOMP.RSSD: {model_info['decomp_rssd']}"
        )
        if 'mape' in model_info:
            metrics_text += f" | MAPE: {model_info['mape']}"
            
        fig.suptitle(
            f"MMM Analysis One-Pager\n{metrics_text}",
            fontsize=14, y=0.98
        )
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig