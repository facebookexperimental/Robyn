from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult


class ResponseVisualizer():
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data

    def plot_response(self) -> plt.Figure:
        """
        Plot response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def plot_marginal_response(self) -> plt.Figure:
        """
        Plot marginal response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        pass

    def generate_response_curves(self, ax: Optional[plt.Axes] = None, trim_rate: float = 1.3) -> Optional[plt.Figure]:
        """Generate response curves showing relationship between spend and response by channel.

        Creates line plots with shaded areas showing the relationship between spend and 
        response for each channel, with mean spend points marked.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure.
            trim_rate: Rate for trimming extreme values. Defaults to 1.3.
            
        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object if ax is None, otherwise None
        """
        # Get plot data
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        curve_data = plot_data['plot4data']['dt_scurvePlot'].copy()
        mean_data = plot_data['plot4data']['dt_scurvePlotMean'].copy()
        
        # Add channel if missing in mean data
        if 'channel' not in mean_data.columns:
            mean_data['channel'] = mean_data['rn']
        
        # Trim data if specified
        if trim_rate > 0:
            max_spend = mean_data['mean_spend_adstocked'].max() * trim_rate
            max_response = mean_data['mean_response'].max() * trim_rate
            
            # Filter curve data
            curve_data = curve_data[
                (curve_data['spend'] < max_spend) &
                (curve_data['response'] < max_response)
            ]
            
            # Add mean carryover information
            curve_data = curve_data.merge(
                mean_data[['channel', 'mean_carryover']],
                on='channel',
                how='left'
            )
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = None
        
        # Set up colors using Set2 colormap
        channels = curve_data['channel'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(channels)))
        
        # Plot response curves for each channel
        for idx, channel in enumerate(channels):
            # Get channel data and sort by spend for smooth curve
            channel_data = curve_data[curve_data['channel'] == channel].sort_values('spend')
            
            # Plot response curve
            ax.plot(channel_data['spend'], 
                    channel_data['response'],
                    color=colors[idx],
                    label=channel,
                    zorder=2)
            
            # Add shaded area up to mean carryover
            if 'mean_carryover' in channel_data.columns:
                carryover_data = channel_data[channel_data['spend'] <= channel_data['mean_carryover'].iloc[0]]
                ax.fill_between(carryover_data['spend'],
                            carryover_data['response'],
                            color='grey',
                            alpha=0.2,
                            zorder=1)
        
        # Add mean points and labels
        for idx, row in mean_data.iterrows():
            # Add point
            ax.scatter(row['mean_spend_adstocked'],
                    row['mean_response'],
                    color=colors[idx],
                    s=100,
                    zorder=3)
            
            # Add label with abbreviated formatting
            if abs(row['mean_spend_adstocked']) >= 1e9:
                formatted_spend = f"{row['mean_spend_adstocked']/1e9:.1f}B"
            elif abs(row['mean_spend_adstocked']) >= 1e6:
                formatted_spend = f"{row['mean_spend_adstocked']/1e6:.1f}M"
            elif abs(row['mean_spend_adstocked']) >= 1e3:
                formatted_spend = f"{row['mean_spend_adstocked']/1e3:.1f}K"
            else:
                formatted_spend = f"{row['mean_spend_adstocked']:.1f}"
                
            ax.text(row['mean_spend_adstocked'],
                    row['mean_response'],
                    formatted_spend,
                    ha='left',
                    va='bottom',
                    fontsize=9,
                    color=colors[idx])
        
        # Format axes with K/M/B notation
        def format_axis_labels(x, p):
            if abs(x) >= 1e9:
                return f"{x/1e9:.0f}B"
            elif abs(x) >= 1e6:
                return f"{x/1e6:.0f}M"
            elif abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis_labels))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_axis_labels))
        
        # Customize plot
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        
        # Remove unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set title and labels
        ax.set_title('Response Curves and Mean Spends by Channel')
        ax.set_xlabel('Spend (carryover + immediate)')
        ax.set_ylabel('Response')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.02, 0.5),
                loc='center left',
                frameon=True,
                framealpha=0.8,
                facecolor='white',
                edgecolor='none')
        
        if fig:
            plt.tight_layout()
            return fig
        return None