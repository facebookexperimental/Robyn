import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    def generate_waterfall(self, baseline_level: int = 0) -> plt.Figure:
        """Generate waterfall chart showing response decomposition by predictor.
        
        Args:
            baseline_level: Aggregation level for baseline variables (0-5)
            
        Returns:
            plt.Figure: Waterfall plot of response contributions
        """
        fig, ax = plt.subplots()

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

    def generate_immediate_vs_carryover(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """Generate stacked bar chart comparing immediate vs carryover effects.
        
        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure
            
        Returns:
            plt.Figure if ax is None, else None
        """
        # Get the plot data
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        df_imme_caov = plot_data['plot7data'].copy()
        
        # Set up type factor levels
        df_imme_caov['type'] = pd.Categorical(df_imme_caov['type'],
                                            categories=['Immediate', 'Carryover'],
                                            ordered=True)
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = None
        
        # Define colors
        colors = {'Immediate': '#59B3D2', 'Carryover': 'coral'}
        
        # Create stacked bar chart
        bottom = np.zeros(len(df_imme_caov['rn'].unique()))
        y_pos = range(len(df_imme_caov['rn'].unique()))
        
        # Get unique channel names and types
        channels = df_imme_caov['rn'].unique()
        types = ['Immediate', 'Carryover']
        
        # Create bar chart with labels
        for type_name in types:
            type_data = df_imme_caov[df_imme_caov['type'] == type_name]
            percentages = type_data['percentage'].values
            
            # Create bars
            bars = ax.barh(y_pos, percentages, 
                        left=bottom,
                        height=0.5,
                        label=type_name,
                        color=colors[type_name])
            
            # Add text labels in center of bars
            for i, (rect, percentage) in enumerate(zip(bars, percentages)):
                width = rect.get_width()
                x_pos = bottom[i] + width/2
                ax.text(x_pos, i, 
                    f"{percentage*100:.0f}%",
                    ha='center', va='center')
            
            bottom += percentages
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels)
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
        ax.set_xlim(0, 1)
        
        # Add legend at top
        ax.legend(title=None,
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc='lower left',
                ncol=2,
                mode="expand",
                borderaxespad=0)
        
        # Add labels and title
        ax.set_xlabel('% Response')
        ax.set_ylabel(None)
        ax.set_title('Immediate vs. Carryover Response Percentage')
        
        # Grid customization
        ax.grid(True, axis='x', alpha=0.2)
        ax.grid(False, axis='y')
        ax.set_axisbelow(True)
        
        # Use white background
        ax.set_facecolor('white')
        
        if fig:
            plt.tight_layout()
            return fig
        return None

    def generate_adstock_rate(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """Generate adstock rate visualization based on adstock type.
        
        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure
            
        Returns:
            plt.Figure if ax is None, else None
        """
        # Get the plot data
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        adstock_data = plot_data['plot3data']
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = None
        
        # Handle different adstock types
        if self.adstock == AdstockType.GEOMETRIC:
            # Get geometric adstock data
            dt_geometric = adstock_data['dt_geometric'].copy()
            
            # Create bar chart
            bars = ax.barh(y=range(len(dt_geometric)), 
                        width=dt_geometric['thetas'],
                        height=0.5,
                        color='coral')
            
            # Add percentage labels
            for i, (theta) in enumerate(dt_geometric['thetas']):
                ax.text(theta + 0.01, i,
                    f"{theta*100:.1f}%",
                    va='center',
                    fontweight='bold')
            
            # Customize axes
            ax.set_yticks(range(len(dt_geometric)))
            ax.set_yticklabels(dt_geometric['channels'])
            
            # Format x-axis as percentage
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
            ax.set_xlim(0, 1)
            
            # Set title and labels
            interval_type = self.mmm_data.mmmdata_spec.interval_type if self.mmm_data else "day"
            ax.set_title('Geometric Adstock: Fixed Rate Over Time')
            ax.set_xlabel(f'Thetas [by {interval_type}]')
            ax.set_ylabel(None)
            
        elif self.adstock in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            # Get Weibull data
            weibull_data = adstock_data['weibullCollect']
            wb_type = adstock_data['wb_type']
            
            # Get unique channels for subplots
            channels = weibull_data['channel'].unique()
            rows = (len(channels) + 2) // 3  # 3 columns
            
            if ax is None:
                # Create new figure with subplots
                fig, axes = plt.subplots(rows, 3, 
                                    figsize=(15, 4*rows),
                                    squeeze=False)
                axes = axes.flatten()
            else:
                # Create subplot grid within provided axis
                gs = ax.get_gridspec()
                subfigs = ax.figure.subfigures(rows, 3)
                axes = [subfig.subplots() for subfig in subfigs]
                axes = [ax for sublist in axes for ax in sublist]  # flatten
            
            # Plot each channel
            for idx, channel in enumerate(channels):
                ax_sub = axes[idx]
                channel_data = weibull_data[weibull_data['channel'] == channel]
                
                # Plot decay curve
                ax_sub.plot(channel_data['x'], 
                        channel_data['decay_accumulated'],
                        color='steelblue')
                
                # Add halflife line
                ax_sub.axhline(y=0.5, color='gray', 
                            linestyle='--', alpha=0.5)
                ax_sub.text(max(channel_data['x']), 0.5,
                        'Halflife',
                        color='gray',
                        va='bottom', ha='right')
                
                # Customize subplot
                ax_sub.set_title(channel)
                ax_sub.grid(True, alpha=0.2)
                ax_sub.set_ylim(0, 1)
                
            # Remove empty subplots if any
            for idx in range(len(channels), len(axes)):
                if ax is None:
                    fig.delaxes(axes[idx])
                else:
                    ax.figure.delaxes(axes[idx])
            
            # Set overall title and labels
            interval_type = self.mmm_data.mmmdata_spec.intervalType if self.mmm_data else "day"
            if ax is None:
                fig.suptitle(f'Weibull {wb_type} Adstock: Flexible Rate Over Time',
                            y=1.02)
                fig.text(0.5, 0.02, f'Time unit [{interval_type}s]',
                        ha='center')
        
        # Customize grid
        if self.adstock == AdstockType.GEOMETRIC:
            ax.grid(True, axis='x', alpha=0.2)
            ax.grid(False, axis='y')
        ax.set_axisbelow(True)
        
        # Use white background
        ax.set_facecolor('white')
        
        if fig:
            plt.tight_layout()
            return fig
        return None