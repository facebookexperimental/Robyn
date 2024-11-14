import os
from typing import Dict, Optional, List, Tuple, Union
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from robyn.data.entities.holidays_data import HolidaysData
import seaborn as sns
import pandas as pd
import logging

from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from robyn.data.entities.hyperparameters import AdstockType
from robyn.data.entities.mmmdata import MMMData

from robyn.visualization.pareto_visualizer import ParetoVisualizer
from robyn.visualization.cluster_visualizer import ClusterVisualizer
from robyn.visualization.response_visualizer import ResponseVisualizer
from robyn.visualization.transformation_visualizer import TransformationVisualizer

logger = logging.getLogger(__name__)

class OnePagerReporter:
    def __init__(
        self,
        pareto_result: ParetoResult,
        clustered_result: Optional[ClusteredResult] = None,
        adstock: Optional[AdstockType] = None,
        mmm_data: Optional[MMMData] = None,
        holidays_data: Optional[HolidaysData] = None
    ):
        self.pareto_result = pareto_result
        self.clustered_result = clustered_result
        self.adstock = adstock
        self.mmm_data = mmm_data
        self.holidays_data = holidays_data
        
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
            'figure.figsize': (22, 17),  # Increased figure size
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

    def _setup_grid(self, n_plots: int, n_solutions: int, figsize: tuple) -> Tuple[plt.Figure, List[GridSpec]]:
        """Set up the grid layout for multiple solutions.
        
        Args:
            n_plots: Number of plots per solution
            n_solutions: Number of solutions to plot
            figsize: Base figure size (width, height)
            
        Returns:
            Tuple containing:
            - Main figure
            - List of GridSpec objects for each solution
        """
        if n_solutions == 1:
            # Single solution case
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            gs = [GridSpec(4, 2, figure=fig)]
        else:
            # Calculate dimensions for multi-solution layout
            n_cols = min(2, n_solutions)  # Maximum 2 columns
            n_rows = (n_solutions + 1) // 2  # Round up division
            
            # Scale figsize based on number of solutions
            scaled_figsize = (
                figsize[0] * n_cols,  # Scale width by number of columns
                figsize[1] * n_rows   # Scale height by number of rows
            )
            
            # Create main figure
            fig = plt.figure(figsize=scaled_figsize)
            
            # Create subfigures
            subfigs = fig.subfigures(n_rows, n_cols, squeeze=False)
            
            # Create GridSpec for each solution
            gs = []
            for i in range(n_solutions):
                row = i // n_cols
                col = i % n_cols
                subfig = subfigs[row, col]
                gs.append(GridSpec(4, 2, figure=subfig))
                
                # Add padding between subfigures
                subfig.set_facecolor('white')
                subfig.supylabel(f'Solution {i+1}', fontsize=12)
                
            # Hide empty subfigures if any
            for i in range(n_solutions, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                subfigs[row, col].set_visible(False)
                
        return fig, gs
    
    def _get_model_info(self, solution_id: str) -> Dict[str, str]:
        """Get model performance metrics for specific solution."""
        try:
            model_data = self.pareto_result.plot_data_collect[solution_id]
            
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
            
            metrics = {
                'rsq_train': self._safe_format(rsq),
                'nrmse': self._safe_format(nrmse),
                'decomp_rssd': self._safe_format(decomp_rssd)
            }
            
            if hasattr(self.pareto_result, 'mape'):
                mape_value = getattr(self.pareto_result, 'mape')
                metrics['mape'] = self._safe_format(mape_value)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting model info for solution {solution_id}: {str(e)}")
            return {
                'rsq_train': "0.0000",
                'nrmse': "0.0000",
                'decomp_rssd': "0.0000"
            }

    def _generate_solution_plots(
        self,
        solution_id: str,
        plots: List[str],
        gs: GridSpec
    ) -> None:
        """Generate plots for a single solution."""
        # Initialize visualizers
        pareto_viz = ParetoVisualizer(self.pareto_result, self.adstock, self.mmm_data, self.holidays_data) if self.adstock and self.holidays_data else None
        cluster_viz = ClusterVisualizer(self.pareto_result, self.clustered_result, self.mmm_data) if self.clustered_result else None
        response_viz = ResponseVisualizer(self.pareto_result, self.mmm_data)
        transfor_viz = TransformationVisualizer(self.pareto_result, self.mmm_data)
        
        # Adjust GridSpec to have more space between plots
        gs.update(
            top=0.85,     # Space for main title
            bottom=0.1,   # Space for x-labels
            left=0.1,
            right=0.9,
            hspace=0.4,   # Increased vertical space between plots
            wspace=0.3    # Increased horizontal space between plots
        )
        
        # Define plot positions and functions
        plot_config = {
            'spend_effect': {
                'position': gs[0, 0],
                'title': 'Share of Total Spend, Effect & Performance',
                'func': lambda ax: transfor_viz.generate_spend_effect_comparison(solution_id, ax)
            },
            'waterfall': {
                'position': gs[1, 0],
                'title': 'Response Decomposition Waterfall',
                'func': lambda ax: pareto_viz.generate_waterfall(solution_id, ax) if pareto_viz else None
            },
            'fitted_vs_actual': {
                'position': gs[0, 1],
                'title': 'Actual vs. Predicted Response',
                'func': lambda ax: pareto_viz.generate_fitted_vs_actual(solution_id, ax) if pareto_viz else None
            },
            'diagnostic': {
                'position': gs[1, 1],
                'title': 'Fitted vs. Residual',
                'func': lambda ax: pareto_viz.generate_diagnostic_plot(solution_id, ax) if pareto_viz else None
            },
            'immediate_carryover': {
                'position': gs[2, 0],
                'title': 'Immediate vs. Carryover Response Percentage',
                'func': lambda ax: pareto_viz.generate_immediate_vs_carryover(solution_id, ax) if pareto_viz else None
            },
            'adstock': {
                'position': gs[2, 1],
                'title': 'Adstock Rate Analysis',
                'func': lambda ax: pareto_viz.generate_adstock_rate(solution_id, ax) if pareto_viz else None
            },
            'bootstrap': {
                'position': gs[3, 0],
                'title': 'Bootstrapped Performance Metrics',
                'func': lambda ax: cluster_viz.generate_bootstrap_confidence(solution_id, ax) if cluster_viz else None
            },
            'response_curves': {
                'position': gs[3, 1],
                'title': 'Response Curves and Mean Spends by Channel',
                'func': lambda ax: response_viz.generate_response_curves(solution_id, ax)
            }
        }

        # Generate each requested plot
        for plot_name in plots:
            if plot_name in plot_config:
                config = plot_config[plot_name]
                ax = plt.subplot(config['position'])
                
                try:
                    config['func'](ax)
                    ax.set_title(config['title'])
                except Exception as e:
                    logger.error(f"Error generating plot {plot_name} for solution {solution_id}: {str(e)}")
                    ax.text(0.5, 0.5, f"Error generating {plot_name}",
                        ha='center', va='center')
                    raise e

        # Get model info and add title
        model_info = self._get_model_info(solution_id)
        metrics_text = (
            f"Model Performance Metrics - "
            f"R²: {model_info['rsq_train']} | "
            f"NRMSE: {model_info['nrmse']} | "
            f"DECOMP.RSSD: {model_info['decomp_rssd']}"
        )
        if 'mape' in model_info:
            metrics_text += f" | MAPE: {model_info['mape']}"
            
        # Add titles with proper spacing
        fig = gs.figure
        fig.suptitle(
            f"MMM Analysis One-Pager (Solution {solution_id})",
            fontsize=14, y=0.98
        )
        # Add metrics text below main title
        fig.text(
            0.5, 0.94,  # x, y position
            metrics_text,
            fontsize=12,
            ha='center'
        )

    def generate_one_pager(
        self,
        solution_ids: Union[str, List[str]] = 'all',
        plots: Optional[List[str]] = None,
        figsize: tuple = (22, 17),  # Updated default figure size
        save_path: Optional[str] = None,
        top_pareto: bool = False
    ) -> List[plt.Figure]:
        """
        Generate separate one-pager for each solution ID.
        
        Args:
            solution_ids: Single solution ID or list of solution IDs or 'all'
            plots: Optional list of plot types to include
            figsize: Figure size for each page
            save_path: Optional path to save the figures.
            top_pareto: If True, loads from clustered results.
            
        Returns:
            List[plt.Figure]: List of generated figures, one per solution
        """
        plots = plots or self.default_plots
        
        # Handle solution IDs based on top_pareto parameter
        if top_pareto:
            if self.clustered_result is None or not hasattr(self.clustered_result, 'top_solutions'):
                raise ValueError("No clustered results or top solutions available")
                
            try:
                # Try accessing 'sol_id' column if it's a DataFrame
                if isinstance(self.clustered_result.top_solutions, pd.DataFrame):
                    solution_ids = self.clustered_result.top_solutions['sol_id'].tolist()
                elif isinstance(self.clustered_result.top_solutions, pd.Series):
                    solution_ids = self.clustered_result.top_solutions.tolist()
                elif isinstance(self.clustered_result.top_solutions, list):
                    solution_ids = self.clustered_result.top_solutions
                else:
                    raise ValueError(f"Unexpected type for top_solutions: {type(self.clustered_result.top_solutions)}")
                
                solution_ids = [str(sid) for sid in solution_ids if sid is not None and pd.notna(sid)]
                total_solutions = len(solution_ids)
                
                if not solution_ids:
                    raise ValueError("No valid solution IDs found in top solutions")
                    
                logger.debug(f"Loading {total_solutions} top solutions")
                
            except Exception as e:
                raise ValueError(f"Error processing top solutions: {str(e)}")
        else:
            if solution_ids == 'all':
                solution_ids = list(self.pareto_result.plot_data_collect.keys())
            elif isinstance(solution_ids, str):
                solution_ids = [solution_ids]
            elif isinstance(solution_ids, (list, tuple)):
                solution_ids = list(solution_ids)
            else:
                raise ValueError(f"solution_ids must be string or list/tuple, got {type(solution_ids)}")
            
            if len(solution_ids) > 1 and not top_pareto:
                warnings.warn(
                    "Too many one pagers to load, please either select top_pareto=True "
                    "or just specify a solution id. Plotting one pager for the first solution id"
                )
                solution_ids = [solution_ids[0]]
                    
        # Validate solution IDs
        invalid_ids = [sid for sid in solution_ids if sid not in self.pareto_result.plot_data_collect]
        if invalid_ids:
            raise ValueError(f"Invalid solution IDs: {invalid_ids}")
        
        figures = []
        
        try:
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                    
            for i, solution_id in enumerate(solution_ids):
                logger.debug(f"Generating one-pager for solution {solution_id} ({i+1}/{len(solution_ids)})")
                
                # Create figure and grid for this solution
                fig = plt.figure(figsize=figsize)
                gs = GridSpec(4, 2, figure=fig)
                
                # Generate plots for this solution
                self._generate_solution_plots(solution_id, plots, gs)
                
                # Adjust layout with improved spacing
                fig.set_constrained_layout_pads(
                    w_pad=0.15,    # Increased padding between plots horizontally
                    h_pad=0.2,     # Increased padding between plots vertically
                    hspace=0.4,    # Increased height space between subplots
                    wspace=0.3     # Increased width space between subplots
                )
                
                # Update layout to leave more space for titles and labels
                plt.subplots_adjust(
                    top=0.85,    # Space for main title
                    bottom=0.1,  # Space for bottom x-labels
                    left=0.1,    # Space for left y-labels
                    right=0.9,   # Space for right margin
                    hspace=0.4,  # Space between plots vertically
                    wspace=0.3   # Space between plots horizontally
                )
                
                if save_path:
                    save_file = os.path.join(save_path, f'solution_{solution_id}.png')
                    fig.savefig(save_file, dpi=300, bbox_inches='tight')
                    logger.debug(f"Saved figure to {save_file}")
                
                figures.append(fig)
                
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            for fig in figures:
                plt.close(fig)
            raise
                
        return figures
    
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