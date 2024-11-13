from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult

logger = logging.getLogger(__name__)

class ResponseVisualizer:
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        logger.debug("Initializing ResponseVisualizer with pareto_result=%s, mmm_data=%s", 
                    pareto_result, mmm_data)
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data

    def plot_response(self) -> plt.Figure:
        """
        Plot response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        logger.info("Starting response curve plotting")
        pass

    def plot_marginal_response(self) -> plt.Figure:
        """
        Plot marginal response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        logger.info("Starting marginal response curve plotting")
        pass
    
    def generate_response_curves(
        self,
        solution_id: str,
        ax: Optional[plt.Axes] = None,
        trim_rate: float = 1.3
    ) -> Optional[plt.Figure]:
        """
        Generate response curves showing relationship between spend and response by channel.
        
        Args:
            solution_id: ID of solution to visualize
            ax: Optional matplotlib axes to plot on. If None, creates new figure
            trim_rate: Rate for trimming extreme values. Defaults to 1.3
            
        Returns:
            Optional[plt.Figure]: Generated figure if ax is None, otherwise None
        """
        logger.debug("Generating response curves with trim_rate=%.2f", trim_rate)

        if solution_id not in self.pareto_result.plot_data_collect:
            raise ValueError(f"Invalid solution ID: {solution_id}")
            

        try:
            # Get plot data for specific solution
            logger.debug("Extracting plot data from pareto results")
            plot_data = self.pareto_result.plot_data_collect[solution_id]
            curve_data = plot_data['plot4data']['dt_scurvePlot'].copy()
            mean_data = plot_data['plot4data']['dt_scurvePlotMean'].copy()
            
            logger.debug("Initial curve data shape: %s", curve_data.shape)
            logger.debug("Initial mean data shape: %s", mean_data.shape)
            
            # Add mean carryover information
            curve_data = curve_data.merge(
                mean_data[['channel', 'mean_carryover']],
                on='channel',
                how='left'
            )
        
            # Create figure if no axes provided
            if ax is None:
                logger.debug("Creating new figure with axes")
                fig, ax = plt.subplots(figsize=(12, 8))
            else:
                logger.debug("Using provided axes")
                fig = None
            
            # Set up colors using Set2 colormap
            channels = curve_data['channel'].unique()
            logger.debug("Processing %d unique channels: %s", len(channels), channels)
            colors = plt.cm.Set2(np.linspace(0, 1, len(channels)))
            
            # Plot response curves for each channel
            for idx, channel in enumerate(channels):
                logger.debug("Plotting response curve for channel: %s", channel)
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
                    logger.debug("Adding carryover shading for channel: %s", channel)
                    carryover_data = channel_data[channel_data['spend'] <= channel_data['mean_carryover'].iloc[0]]
                    ax.fill_between(carryover_data['spend'],
                                carryover_data['response'],
                                color='grey',
                                alpha=0.2,
                                zorder=1)
            
            # Add mean points and labels
            logger.debug("Adding mean points and labels")
            for idx, row in mean_data.iterrows():
                # Add point
                ax.scatter(row['mean_spend_adstocked'],
                        row['mean_response'],
                        color=colors[idx],
                        s=100,
                        zorder=3)
                
                # Format and add label with abbreviated values
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
            
            logger.debug("Formatting axis labels")
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
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set labels
            ax.set_title(f'Response Curves and Mean Spends by Channel (Solution {solution_id})')
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
                logger.debug("Adjusting layout")
                plt.tight_layout()
                logger.debug("Successfully generated response curves figure")
                return fig
            
            logger.debug("Successfully added response curves to existing axes")
            return None
    
        except Exception as e:
            logger.error("Error generating response curves: %s", str(e), exc_info=True)
            raise