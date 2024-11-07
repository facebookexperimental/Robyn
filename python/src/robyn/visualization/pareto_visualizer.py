from typing import Optional
from matplotlib import ticker
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

    def format_number(self, x: float, pos=None) -> str:
        """Format large numbers with K/M/B abbreviations.
        
        Args:
            x: Number to format
            pos: Position (required by matplotlib FuncFormatter but not used)
            
        Returns:
            Formatted string
        """
        if abs(x) >= 1e9:
            return f"{x/1e9:.1f}B"
        elif abs(x) >= 1e6:
            return f"{x/1e6:.1f}M"
        elif abs(x) >= 1e3:
            return f"{x/1e3:.1f}K"
        else:
            return f"{x:.1f}"    

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

    def generate_diagnostic_plot(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """Generate diagnostic scatter plot of fitted vs residual values.
        
        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure
            
        Returns:
            plt.Figure if ax is None, else None
        """
        # Get the plot data
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        diag_data = plot_data['plot6data']['xDecompVecPlot'].copy()
        
        # Calculate residuals
        diag_data['residuals'] = diag_data['actual'] - diag_data['predicted']
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = None
            
        # Create scatter plot
        ax.scatter(diag_data['predicted'], diag_data['residuals'], 
                alpha=0.5, color='steelblue')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add smoothed line with confidence interval
        from scipy.stats import gaussian_kde
        x_smooth = np.linspace(diag_data['predicted'].min(), 
                            diag_data['predicted'].max(), 100)
        
        # Fit LOWESS
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(diag_data['residuals'], 
                        diag_data['predicted'],
                        frac=0.2)
        
        # Plot smoothed line
        ax.plot(smoothed[:, 0], smoothed[:, 1], 
                color='red', linewidth=2, alpha=0.8)
        
        # Calculate confidence intervals (using standard error bands)
        residual_std = np.std(diag_data['residuals'])
        ax.fill_between(smoothed[:, 0],
                    smoothed[:, 1] - 2*residual_std,
                    smoothed[:, 1] + 2*residual_std,
                    color='red', alpha=0.1)
        
        # Format axes with abbreviations
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))
        
        # Set labels and title
        ax.set_xlabel('Fitted')
        ax.set_ylabel('Residual')
        ax.set_title('Fitted vs. Residual')
        
        # Customize grid
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        
        # Use white background
        ax.set_facecolor('white')
        
        if fig:
            plt.tight_layout()
            return fig
        return None

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