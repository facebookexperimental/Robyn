from typing import Optional
from matplotlib import ticker, transforms
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

    def generate_fitted_vs_actual(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """Generate time series plot comparing fitted vs actual values.
        
        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure
            
        Returns:
            plt.Figure if ax is None, else None
        """
        # Get the plot data
        plot_data = next(iter(self.pareto_result.plot_data_collect.values()))
        ts_data = plot_data['plot5data']['xDecompVecPlotMelted'].copy()
        
        # Convert dates and format variables
        ts_data['ds'] = pd.to_datetime(ts_data['ds'])
        ts_data['linetype'] = np.where(ts_data['variable'] == 'predicted', 'solid', 'dotted')
        ts_data['variable'] = ts_data['variable'].str.title()
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = None
            
        # Plot lines with different styles for predicted vs actual
        for var in ts_data['variable'].unique():
            var_data = ts_data[ts_data['variable'] == var]
            linestyle = 'solid' if var_data['linetype'].iloc[0] == 'solid' else 'dotted'
            ax.plot(var_data['ds'], var_data['value'],
                    label=var,
                    linestyle=linestyle,
                    linewidth=0.6)
        
        # Format y-axis with abbreviations
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))
        
        # Add training/validation/test splits if validation is enabled
        if hasattr(self.pareto_result, 'train_size'):
            train_size = self.pareto_result.train_size
            
            # Calculate split points
            days = sorted(ts_data['ds'].unique())
            ndays = len(days)
            train_cut = round(ndays * train_size)
            val_cut = train_cut + round(ndays * (1 - train_size) / 2)
            
            # Add vertical lines and labels for splits
            splits = [
                (train_cut, f"Train: {train_size*100:.1f}%"),
                (val_cut, f"Validation: {((1-train_size)/2)*100:.1f}%"),
                (ndays-1, f"Test: {((1-train_size)/2)*100:.1f}%")
            ]
            
            for idx, (cut, label) in enumerate(splits):
                # Add vertical line
                ax.axvline(x=days[cut], color='#39638b', alpha=0.8, linestyle='-')
                
                # Add rotated text label
                trans = transforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                )
                ax.text(days[cut], 1.02, label,
                    rotation=270,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=trans,
                    color='#39638b',
                    alpha=0.5,
                    fontsize=8)
        
        # Customize plot
        ax.set_title('Actual vs. Predicted Response')
        ax.set_xlabel('Date')
        ax.set_ylabel('Response')
        
        # Move legend to top
        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc='lower left',
            ncol=2,
            mode="expand",
            borderaxespad=0,
        )
        
        # Grid styling
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        
        # Use white background
        ax.set_facecolor('white')
        
        if fig:
            plt.tight_layout()
            return fig
        return None

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