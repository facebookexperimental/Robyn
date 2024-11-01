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

    def generate_waterfall(self, baseline_level: int = 0) -> plt.Figure:
        """Generate waterfall chart showing response decomposition by predictor.
        
        Args:
            baseline_level: Aggregation level for baseline variables (0-5)
            
        Returns:
            plt.Figure: Waterfall plot of response contributions
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get waterfall data from plot2data
        plot_data = self.pareto_result.plot_data_collect
        waterfall_data = next(iter(plot_data.values()))['plot2data']['plotWaterfallLoop']
        
        # Create the waterfall chart
        for idx, row in waterfall_data.iterrows():
            color = '#59B3D2' if row['sign'] == 'Positive' else '#E5586E'
            ax.bar(row['id'], row['end'] - row['start'], bottom=row['start'], 
                  color=color, width=0.8)
            
            # Add value labels
            label_y = row['start'] + (row['end'] - row['start'])/2
            ax.text(row['id'], label_y, f"{row['xDecompAgg']:,.0f}\n{row['xDecompPerc']:.1%}", 
                   ha='center', va='center')
            
        # Customize plot
        ax.set_xticks(waterfall_data['id'])
        ax.set_xticklabels(waterfall_data['rn'], rotation=45, ha='right')
        ax.set_ylabel('Response Decomposition (%)')
        ax.set_title('Response Decomposition Waterfall by Predictor')
        
        plt.tight_layout()
        return fig

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