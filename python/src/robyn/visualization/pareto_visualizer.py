from typing import Optional
import matplotlib.pyplot as plt
from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from robyn.data.entities.hyperparameters import AdstockType
from robyn.data.entities.mmmdata import MMMData
import pandas as pd
import numpy as np


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
        """Generate time series plot comparing fitted vs actual values."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get fitted vs actual data from plot5data
        plot_data = self.pareto_result.plot_data_collect
        ts_data = next(iter(plot_data.values()))['plot5data']['xDecompVecPlotMelted']
        
        # Plot actual and predicted values
        for var in ts_data['variable'].unique():
            var_data = ts_data[ts_data['variable'] == var]
            linestyle = '-' if var == 'predicted' else '--'
            ax.plot(pd.to_datetime(var_data['ds']), var_data['value'], 
                   label=var.title(), linestyle=linestyle)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Response')
        ax.set_title('Actual vs. Predicted Response')
        ax.legend()
        
        plt.tight_layout()
        return fig

    def generate_diagnostic_plot(self) -> plt.Figure:
        """Generate diagnostic scatter plot of fitted vs residual values."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get diagnostic data from plot6data
        plot_data = self.pareto_result.plot_data_collect
        diag_data = next(iter(plot_data.values()))['plot6data']['xDecompVecPlot']
        
        # Calculate residuals
        residuals = diag_data['actual'] - diag_data['predicted']
        
        # Create scatter plot
        ax.scatter(diag_data['predicted'], residuals, alpha=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add trend line
        z = np.polyfit(diag_data['predicted'], residuals, 1)
        p = np.poly1d(z)
        ax.plot(diag_data['predicted'], p(diag_data['predicted']), 
                "r--", alpha=0.8)
        
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Fitted vs Residual Plot')
        
        plt.tight_layout()
        return fig

    def generate_immediate_vs_carryover(self) -> plt.Figure:
        """Generate stacked bar chart comparing immediate vs carryover effects."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get immediate vs carryover data from plot7data
        plot_data = self.pareto_result.plot_data_collect
        effect_data = next(iter(plot_data.values()))['plot7data']
        
        # Pivot data for stacking
        pivot_data = effect_data.pivot(index='rn', columns='type', values='percentage')
        
        # Create stacked bar chart
        pivot_data.plot(kind='bar', stacked=True, ax=ax, 
                       color=['#59B3D2', 'coral'])
        
        # Add percentage labels
        for c in pivot_data.columns:
            for i, value in enumerate(pivot_data[c]):
                ax.text(i, pivot_data.iloc[i,:c].sum() + value/2, 
                       f'{value:.1%}', ha='center', va='center')
        
        ax.set_xlabel(None)
        ax.set_ylabel('% Response')
        ax.set_title('Immediate vs Carryover Response Percentage')
        ax.legend(title=None)
        
        plt.tight_layout()
        return fig

    def generate_adstock_rate(self) -> plt.Figure:
        """Generate plot showing adstock rates over time by channel."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get adstock data from plot3data
        plot_data = self.pareto_result.plot_data_collect
        adstock_data = next(iter(plot_data.values()))['plot3data']
        
        if self.adstock == AdstockType.GEOMETRIC:
            # Plot geometric adstock rates
            dt_geometric = adstock_data['dt_geometric']
            ax.bar(dt_geometric['channels'], dt_geometric['thetas'])
            ax.set_title('Geometric Adstock: Fixed Rate Over Time')
            ax.set_ylabel('Theta')
            
        elif self.adstock in [AdstockType.WEIBULL_PDF, AdstockType.WEIBULL_CDF]:
            # Plot Weibull adstock decay curves
            weibull_data = adstock_data['weibullCollect']
            for channel in weibull_data['channel'].unique():
                channel_data = weibull_data[weibull_data['channel'] == channel]
                ax.plot(channel_data['x'], channel_data['decay_accumulated'], 
                       label=channel)
            
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[1], 0.5, 'Halflife', 
                   verticalalignment='bottom', horizontalalignment='right')
            
            ax.set_title(f'Weibull {self.adstock} Adstock: Flexible Rate Over Time')
            ax.set_xlabel('Time unit')
            ax.set_ylabel('Decay Rate')
            ax.legend()
            
        plt.tight_layout()
        return fig