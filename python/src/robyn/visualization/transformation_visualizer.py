# pyre-strict
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from typing import Tuple

import numpy as np
from robyn.modeling.pareto.pareto_optimizer import ParetoResult


class TransformationVisualizer:
    def __init__(self, pareto_result: ParetoResult):
        self.pareto_Result = ParetoResult

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


    def generate_spend_effect_comparison(self) -> plt.Figure:
        """Generate bar and line plot comparing spend share vs effect share.
        
        This visualization matches the R implementation's comparison plot showing:
        - Stacked bar chart for spend and effect shares
        - Line plot overlay showing ROI/CPA values
        - Percentage labels for shares
        - Value labels for ROI/CPA
        
        Returns:
            plt.Figure: Plot comparing media spend shares and their effects
        """
        # Get plot data from first solution in pareto results
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        plot1data = plot_data['plot1data']
        
        bar_data = plot1data['plotMediaShareLoopBar']
        line_data = plot1data['plotMediaShareLoopLine'] 
        y_scale = plot1data['ySecScale']
        
        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Calculate bar positions and widths
        n_channels = len(bar_data['rn'].unique())
        channel_positions = np.arange(n_channels)
        bar_width = 0.35
        
        # Create grouped bars for spend and effect shares
        spend_mask = bar_data['variable'] == 'spend_share'
        effect_mask = bar_data['variable'] == 'effect_share'
        
        spend_bars = ax1.bar(channel_positions - bar_width/2, 
                            bar_data[spend_mask]['value'],
                            bar_width, label='Spend Share',
                            color='lightblue', alpha=0.8)
                            
        effect_bars = ax1.bar(channel_positions + bar_width/2,
                             bar_data[effect_mask]['value'],
                             bar_width, label='Effect Share',
                             color='steelblue', alpha=0.8)
        
        # Add percentage labels on bars
        def add_labels(bars, data):
            for bar, value in zip(bars, data['value']):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1%}',
                        ha='center', va='bottom')
                        
        add_labels(spend_bars, bar_data[spend_mask])
        add_labels(effect_bars, bar_data[effect_mask])
        
        # Create secondary axis for ROI/CPA line
        ax2 = ax1.twinx()
        
        # Plot ROI/CPA line
        line = ax2.plot(channel_positions, line_data['value'],
                       color='#03396C', marker='o',
                       linewidth=2, markersize=8,
                       label='ROI/CPA')
                       
        # Add ROI/CPA value labels
        for x, y in zip(channel_positions, line_data['value']):
            ax2.text(x, y, f'{y:.2f}',
                    color='#03396C',
                    ha='left', va='bottom')
            
        # Customize primary axis
        ax1.set_ylabel('Share of Total (%)')
        ax1.set_ylim(0, max(bar_data['value']) * 1.2)
        ax1.set_xticks(channel_positions)
        ax1.set_xticklabels(bar_data[spend_mask]['rn'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Customize secondary axis
        metric = 'ROI' if self.pareto_result.metric_type == 'revenue' else 'CPA'
        ax2.set_ylabel(f'{metric} Value')
        ax2.set_ylim(0, max(line_data['value']) * 1.2)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right', bbox_to_anchor=(1, 1.1))
        
        # Set title
        period = self.pareto_result.period_type
        fig.suptitle(f'Share of Total Spend, Effect & {metric} in {period} Window*',
                    y=1.05, fontsize=12)
                    
        # Add footnote about calculation
        if metric == 'ROI':
            footnote = '* Total ROI = sum of response / sum of spend'
        else:
            footnote = '* Total CPA = sum of spend / sum of response'
            
        fig.text(0.01, -0.05, footnote, fontsize=8, style='italic')
        
        plt.tight_layout()
        
        return fig
