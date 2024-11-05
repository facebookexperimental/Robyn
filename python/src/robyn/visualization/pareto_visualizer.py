from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.data.entities.hyperparameters import AdstockType
from robyn.data.entities.mmmdata import MMMData

class ParetoVisualizer:
    """
    Class for visualizing pareto results.
    """
    def __init__(self, pareto_result: ParetoResult, adstock: AdstockType, mmm_data: MMMData):
        self.pareto_result = pareto_result
        self.adstock = adstock
        self.mmm_data = mmm_data

    def generate_waterfall(self, ax: Optional[plt.Axes] = None, baseline_level: int = 0) -> Optional[plt.Figure]:
        """Generate waterfall chart showing response decomposition by predictor."""
        # Get data for first model ID
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        waterfall_data = plot_data['plot2data']['plotWaterfallLoop'].copy()

        # Get baseline variables
        bvars = []
        if self.mmm_data and hasattr(self.mmm_data.mmmdata_spec, 'prophet_vars'):
            bvars = ['(Intercept)'] + self.mmm_data.mmmdata_spec.prophet_vars

        # Transform baseline variables
        waterfall_data.loc[waterfall_data['rn'].isin(bvars), 'rn'] = f'Baseline_L{baseline_level}'

        # Group and summarize
        waterfall_data = (waterfall_data.groupby('rn', as_index=False)
                        .agg({
                            'xDecompAgg': 'sum',
                            'xDecompPerc': 'sum'
                        }))
        
        # Sort by percentage contribution
        waterfall_data = waterfall_data.sort_values('xDecompPerc', ascending=True)
        
        # Calculate waterfall positions
        waterfall_data['end'] = 1 - waterfall_data['xDecompPerc'].cumsum()
        waterfall_data['start'] = waterfall_data['end'].shift(1)
        waterfall_data['start'] = waterfall_data['start'].fillna(1)
        waterfall_data['sign'] = np.where(waterfall_data['xDecompPerc'] >= 0, 'Positive', 'Negative')

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = None
            
        # Define colors
        colors = {'Positive': '#59B3D2', 'Negative': '#E5586E'}
        
        # Create categorical y-axis
        y_pos = range(len(waterfall_data))
        
        # Create horizontal bars
        bars = ax.barh(y=y_pos,
                    width=waterfall_data['start'] - waterfall_data['end'],
                    left=waterfall_data['end'],
                    color=[colors[sign] for sign in waterfall_data['sign']],
                    height=0.6)
        
        # Add text labels
        for i, row in waterfall_data.iterrows():
            # Format label text
            if abs(row['xDecompAgg']) >= 1e9:
                formatted_num = f"{row['xDecompAgg']/1e9:.1f}B"
            elif abs(row['xDecompAgg']) >= 1e6:
                formatted_num = f"{row['xDecompAgg']/1e6:.1f}M"
            elif abs(row['xDecompAgg']) >= 1e3:
                formatted_num = f"{row['xDecompAgg']/1e3:.1f}K"
            else:
                formatted_num = f"{row['xDecompAgg']:.1f}"
                
            # Calculate x-position as the right edge of each positive bar
            x_pos = max(row['start'], row['end'])
            
            # Add label aligned at the end of the bar
            ax.text(x_pos - 0.01, i,  # Small offset from bar end
                f"{formatted_num}\n{row['xDecompPerc']*100:.1f}%",
                ha='right', va='center',
                fontsize=9,
                linespacing=0.9)
        
        # Set y-ticks and labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(waterfall_data['rn'])
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        
        # Set plot limits
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(waterfall_data) - 0.5)
        
        # Add legend at top
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['Positive'], label='Positive'),
            Patch(facecolor=colors['Negative'], label='Negative')
        ]
        
        # Create legend with white background
        legend = ax.legend(handles=legend_elements,
                        title='Sign',
                        loc='upper left',
                        bbox_to_anchor=(0, 1.15),
                        ncol=2,
                        frameon=True,
                        framealpha=1.0)
        
        # Set title
        ax.set_title('Response Decomposition Waterfall', 
                    pad=30,
                    x=0.5,
                    y=1.05)
        
        # Label axes
        ax.set_xlabel('Contribution')
        ax.set_ylabel(None)
        
        # Customize grid
        ax.grid(True, axis='x', alpha=0.2)
        ax.set_axisbelow(True)
        
        # Adjust layout
        if fig:
            plt.subplots_adjust(right=0.85, top=0.85)
            return fig
            
        return None
           
    def generate_fitted_vs_actual(self) -> plt.Figure:
        """Generate time series plot comparing fitted vs actual values.
            
        Returns:
            plt.Figure: Line plot comparing predicted and actual values
        """
        fig, ax = plt.subplots()

    def generate_diagnostic_plot(self) -> plt.Figure:
        """Generate diagnostic scatter plot of fitted vs residual values.
        
            
        Returns:
            plt.Figure: Scatter plot with trend line
        """
        fig, ax = plt.subplots()

    def generate_immediate_vs_carryover(self) -> plt.Figure:
        """Generate stacked bar chart comparing immediate vs carryover effects.
        
            
        Returns:
            plt.Figure: Stacked bar plot of effect types
        """
        fig, ax = plt.subplots()

    def generate_adstock_rate(self) -> plt.Figure:
        """Generate plot showing adstock rates over time by channel.
            
        Returns:
            plt.Figure: Line plot of adstock decay rates
        """

        """
            NOTE: Missing intervalType mapping in data_mapper from input collect.
        """
        fig, ax = plt.subplots()