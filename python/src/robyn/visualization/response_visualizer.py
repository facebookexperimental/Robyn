import matplotlib.pyplot as plt
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.pareto.pareto_optimizer import ParetoResult
import seaborn as sns

class ResponseVisualizer():
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        self.response_data = pareto_result
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

    def generate_response_curves(self, trim_rate: float = 1.3) -> plt.Figure:
        """Generate response curves with mean spend points.
        
        This visualization matches the R implementation's response curves plot showing:
        - Response curves for each media channel
        - Mean spend points marked on curves
        - Shaded areas for historical spend ranges
        - Clear labeling of mean spend values
        
        Args:
            trim_rate: Factor for trimming extreme values to focus plot range
            
        Returns:
            plt.Figure: Response curves plot with mean spend points
        """
        # Get plot data from first solution in pareto results
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        plot4data = plot_data['plot4data']
        
        # Extract curve and mean data
        curve_data = plot4data['dt_scurvePlot']
        mean_data = plot4data['dt_scurvePlotMean']
        
        # Apply trimming if specified
        if trim_rate > 0:
            max_mean_spend = mean_data['mean_spend_adstocked'].max()
            max_mean_response = mean_data['mean_response'].max()
            
            curve_data = curve_data[
                (curve_data['spend'] < max_mean_spend * trim_rate) & 
                (curve_data['response'] < max_mean_response * trim_rate)
            ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot response curves for each channel
        channels = curve_data['channel'].unique()
        colors = sns.color_palette('husl', n_colors=len(channels))
        color_dict = dict(zip(channels, colors))
        
        for channel in channels:
            # Get channel data
            channel_curve = curve_data[curve_data['channel'] == channel]
            channel_mean = mean_data[mean_data['channel'] == channel].iloc[0]
            
            # Plot main response curve
            ax.plot(channel_curve['spend'], channel_curve['response'],
                   label=channel, color=color_dict[channel],
                   linewidth=2)
            
            # Add shaded area up to mean carryover
            mask = channel_curve['spend'] <= channel_mean['mean_carryover']
            ax.fill_between(channel_curve[mask]['spend'],
                          channel_curve[mask]['response'],
                          color=color_dict[channel],
                          alpha=0.2)
            
            # Plot mean spend point
            ax.scatter(channel_mean['mean_spend_adstocked'],
                      channel_mean['mean_response'],
                      color=color_dict[channel],
                      s=100, zorder=5)
            
            # Add mean spend label
            ax.annotate(
                f"{channel_mean['mean_spend_adstocked']:,.0f}",
                xy=(channel_mean['mean_spend_adstocked'],
                    channel_mean['mean_response']),
                xytext=(10, 0), textcoords='offset points',
                va='center',
                color=color_dict[channel],
                fontweight='bold'
            )
        
        # Customize plot
        ax.set_xlabel('Spend (carryover + immediate)')
        ax.set_ylabel('Response')
        ax.set_title('Response Curves and Mean Spends by Channel')
        
        # Format axis labels for large numbers
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Adjust legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                 borderaxespad=0., frameon=True,
                 fancybox=True, framealpha=0.8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Ensure positive axes
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Add annotation about shaded areas
        fig.text(0.02, -0.02,
                'Note: Grey shaded areas represent historical carryover ranges',
                style='italic', fontsize=8)
        
        plt.tight_layout()
        
        return fig