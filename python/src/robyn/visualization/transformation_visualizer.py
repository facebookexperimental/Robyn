# pyre-strict
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from typing import Tuple

from matplotlib.ticker import PercentFormatter
import numpy as np
from robyn.data.entities.enums import DependentVarType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from matplotlib.patches import Patch
import seaborn as sns

class TransformationVisualizer:
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data

    def create_adstock_plots(self) -> None:
        """
        Generate adstock visualization plots and store them as instance variables.
        """
        pass

    def create_saturation_plots(self) -> None:
        """
        Generate saturation visualization plots and store them as instance variables.
        """
        pass

    def get_adstock_plots(self) -> Optional[Tuple[plt.Figure, plt.Figure]]:
        """
        Retrieve the adstock plots.

        Returns:
            Optional[Tuple[plt.Figure, plt.Figure]]: Tuple of matplotlib figures for adstock plots
        """
        pass

    def get_saturation_plots(self) -> Optional[Tuple[plt.Figure, plt.Figure]]:
        """
        Retrieve the saturation plots.

        Returns:
            Optional[Tuple[plt.Figure, plt.Figure]]: Tuple of matplotlib figures for saturation plots
        """
        pass

    def display_adstock_plots(self) -> None:
        """
        Display the adstock plots.
        """
        pass

    def display_saturation_plots(self) -> None:
        """
        Display the saturation plots.
        """
        pass

    def save_adstock_plots(self, filenames: List[str]) -> None:
        """
        Save the adstock plots to files.

        Args:
            filenames (List[str]): List of filenames to save the plots
        """
        pass

    def save_saturation_plots(self, filenames: List[str]) -> None:
        """
        Save the saturation plots to files.

        Args:
            filenames (List[str]): List of filenames to save the plots
        """
        pass


    def generate_spend_effect_comparison(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """Generate comparison plot of spend share vs effect share."""
        # Get plot data
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        bar_data = plot_data['plot1data']['plotMediaShareLoopBar'].copy()
        line_data = plot_data['plot1data']['plotMediaShareLoopLine'].copy()
        
        # Extract scalar value from ySecScale DataFrame
        y_sec_scale = float(plot_data['plot1data']['ySecScale'].iloc[0])
        
        # Transform variable names
        bar_data['variable'] = bar_data['variable'].str.replace('_', ' ').str.title()
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = None
        
        # Set background color
        ax.set_facecolor('white')
        
        # Set up colors
        type_colour = '#03396C'  # Dark blue for line
        bar_colors = ['#A4C2F4', '#FFB7B2']  # Light blue and light coral for bars
        
        # Set up dimensions
        channels = line_data['rn'].unique()  # Use line_data for consistent ordering
        y_pos = np.arange(len(channels))
        
        # Plot bars for each variable type
        bar_width = 0.35
        for i, (var, color) in enumerate(zip(bar_data['variable'].unique(), bar_colors)):
            var_data = bar_data[bar_data['variable'] == var]
            # Ensure alignment with channels
            values = [var_data[var_data['rn'] == ch]['value'].iloc[0] for ch in channels]
            bars = ax.barh(y=[y + (i-0.5)*bar_width for y in y_pos],
                        width=values,
                        height=bar_width,
                        label=var,
                        color=color,
                        alpha=0.5)
        
        # Convert line values to numpy array with correct dimensions
        line_values = np.array([line_data[line_data['rn'] == ch]['value'].iloc[0] for ch in channels])
        line_x = line_values / y_sec_scale
        
        # Plot line
        ax.plot(line_x, y_pos, 
                color=type_colour,
                marker='o',
                markersize=8,
                zorder=3)
        
        # Add line value labels
        for i, value in enumerate(line_values):
            ax.text(line_x[i], y_pos[i],
                f"{value:.2f}",
                color=type_colour,
                fontweight='bold',
                ha='left',
                va='center',
                zorder=4)
        
        # Set channel labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels)
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
        ax.set_xlim(0, max(1, np.max(line_x) * 1.2))
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.2, linestyle='-')
        ax.set_axisbelow(True)
        
        # Remove unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set title
        metric_type = "ROI" if (self.mmm_data and 
                            hasattr(self.mmm_data.mmmdata_spec, 'dep_var_type') and 
                            self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE) else "CPA"
        ax.set_title(f'Total Spend% VS Effect% with total {metric_type}')
        
        # Add legend
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower left",
                mode="expand",
                ncol=2)
        
        # Add axis labels
        ax.set_xlabel('Total Share by Channel')
        ax.set_ylabel(None)
        
        if fig:
            plt.tight_layout()
            return fig
        return None