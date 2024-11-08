from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd

from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from robyn.data.entities.hyperparameters import AdstockType
from robyn.data.entities.mmmdata import MMMData

from robyn.visualization.pareto_visualizer import ParetoVisualizer
from robyn.visualization.cluster_visualizer import ClusterVisualizer
from robyn.visualization.response_visualizer import ResponseVisualizer
from robyn.visualization.transformation_visualizer import TransformationVisualizer

class OnePagerReporter:
    def __init__(
        self,
        pareto_result: ParetoResult,
        clustered_result: Optional[ClusteredResult] = None,
        adstock: Optional[AdstockType] = None,
        mmm_data: Optional[MMMData] = None
    ):
        self.pareto_result = pareto_result
        self.clustered_result = clustered_result
        self.adstock = adstock
        self.mmm_data = mmm_data
        
        # Initialize visualizers
        self.pareto_viz = ParetoVisualizer(pareto_result, adstock, mmm_data) if adstock else None
        self.cluster_viz = ClusterVisualizer(pareto_result, clustered_result, mmm_data) if clustered_result else None
        self.response_viz = ResponseVisualizer(pareto_result, mmm_data)
        self.transfor_viz = TransformationVisualizer(pareto_result, mmm_data)
        
        # Default plots to show
        self.default_plots = [
            'spend_effect', 'waterfall', 'fitted_vs_actual', 'bootstrap',
            'adstock', 'immediate_carryover', 'response_curves', 'diagnostic'
        ]

        # Set up matplotlib style
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure the plotting style for the one-pager."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update({
            'figure.figsize': (20, 15),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    def _setup_grid(self, n_plots: int, figsize: tuple) -> Tuple[plt.Figure, GridSpec]:
        """Set up the grid layout for the one pager."""
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(4, 2, figure=fig)
        return fig, gs
        
    def _safe_format(self, value, precision: int = 4) -> str:
        """Safely format numeric values with specified precision."""
        try:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                value = value.iloc[0] if isinstance(value, pd.Series) else value.iloc[0, 0]
            if pd.isna(value):
                return "0.0000"
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError, IndexError):
            return "0.0000"
            
    def _get_model_info(self) -> Dict[str, str]:
        """Get model performance metrics and information."""
        try:
            # Get first model's data
            model_data = next(iter(self.pareto_result.plot_data_collect.values()))
            
            # Extract RSQ from plot5data safely
            rsq = (
                model_data['plot5data'].get('rsq')
                if isinstance(model_data['plot5data'], dict)
                else model_data['plot5data'].rsq.iloc[0]
                if hasattr(model_data['plot5data'], 'rsq')
                else 0
            )
            
            # Get NRMSE and DECOMP.RSSD values safely
            nrmse = model_data.get('nrmse', 0)
            if isinstance(nrmse, (pd.DataFrame, pd.Series)):
                nrmse = nrmse.iloc[0] if isinstance(nrmse, pd.Series) else nrmse.iloc[0, 0]
                
            decomp_rssd = model_data.get('decomp.rssd', 0)
            if isinstance(decomp_rssd, (pd.DataFrame, pd.Series)):
                decomp_rssd = decomp_rssd.iloc[0] if isinstance(decomp_rssd, pd.Series) else decomp_rssd.iloc[0, 0]
            
            # Format metrics
            metrics = {
                'rsq_train': self._safe_format(rsq),
                'nrmse': self._safe_format(nrmse),
                'decomp_rssd': self._safe_format(decomp_rssd)
            }
            
            # Add MAPE if available
            if hasattr(self.pareto_result, 'mape'):
                mape_value = getattr(self.pareto_result, 'mape')
                metrics['mape'] = self._safe_format(mape_value)
                
            return metrics
            
        except Exception as e:
            print(f"Error getting model info: {str(e)}")
            return {
                'rsq_train': "0.0000",
                'nrmse': "0.0000",
                'decomp_rssd': "0.0000"
            }

    def generate_one_pager(
    self,
    plots: Optional[List[str]] = None,
    figsize: tuple = (20, 15)
) -> plt.Figure:
        """Generate a one-page report with multiple visualization plots."""
        plots = plots or self.default_plots
        
        # Create figure and grid
        fig, gs = self._setup_grid(len(plots), figsize)
        
        # Get model information
        model_info = self._get_model_info()
        
        # Define plot positions and functions
        plot_config = {
            'spend_effect': {
                'position': gs[0, 0],
                'title': 'Share of Total Spend, Effect & Performance'
            },
            'waterfall': {
                'position': gs[1, 0],
                'title': 'Response Decomposition Waterfall'
            },
            'fitted_vs_actual': {
                'position': gs[0, 1],
                'title': 'Actual vs. Predicted Response'
            },
            'diagnostic': {
                'position': gs[1, 1],
                'title': 'Fitted vs. Residual'
            },
            'immediate_carryover': {
                'position': gs[2, 0],
                'title': 'Immediate vs. Carryover Response Percentage'
            },
            'adstock': {
                'position': gs[2, 1],
                'title': 'Adstock Rate Analysis'
            },
            'bootstrap': {
                'position': gs[3, 0],
                'title': 'Bootstrapped Performance Metrics'
            },
            'response_curves': {
                'position': gs[3, 1],
                'title': 'Response Curves and Mean Spends by Channel'
            }
        }
        
        # Generate each requested plot
        for plot_name in plots:
            if plot_name in plot_config:
                config = plot_config[plot_name]
                ax = fig.add_subplot(config['position'])
                
                try:
                    if plot_name == 'spend_effect' and self.transfor_viz:
                        self.transfor_viz.generate_spend_effect_comparison(ax=ax)
                    elif plot_name == 'waterfall' and self.pareto_viz:
                        self.pareto_viz.generate_waterfall(ax=ax)
                    elif plot_name == 'fitted_vs_actual' and self.pareto_viz:
                        self.pareto_viz.generate_fitted_vs_actual(ax=ax)
                    elif plot_name == 'diagnostic' and self.pareto_viz:
                        self.pareto_viz.generate_diagnostic_plot(ax=ax)
                    elif plot_name == 'immediate_carryover' and self.pareto_viz:
                        self.pareto_viz.generate_immediate_vs_carryover(ax=ax)
                    elif plot_name == 'adstock' and self.pareto_viz:
                        self.pareto_viz.generate_adstock_rate(ax=ax)
                    elif plot_name == 'bootstrap' and self.cluster_viz:
                        self.cluster_viz.generate_bootstrap_confidence(ax=ax)
                    elif plot_name == 'response_curves' and self.response_viz:  # Changed from response to response_viz
                        self.response_viz.generate_response_curves(ax=ax)
                    ax.set_title(config['title'])
                except Exception as e:
                    print(f"Error generating plot {plot_name}: {str(e)}")
                    ax.text(0.5, 0.5, f"Error generating {plot_name}", 
                        ha='center', va='center')
        
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
        
        # Update default plots list 
        self.default_plots = [
            'spend_effect',
            'waterfall',
            'fitted_vs_actual',
            'diagnostic',
            'immediate_carryover',
            'adstock',
            'bootstrap',
            'response_curves'
        ]
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig