import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import io
import base64
from robyn.modeling.entities.modeloutputs import Trial
from typing import Tuple
import plotly.graph_objects as go
from plot_data import PlotData

class ParetoVisualizer:

    def generate_waterfall(data: PlotData, baseline_level: int = 0) -> go.Figure:
        """Generate waterfall chart showing response decomposition by predictor.
        
        Args:
            data: PlotData instance containing required data
            baseline_level: Aggregation level for baseline variables (0-5)
            
        Returns:
            go.Figure: Waterfall plot of response contributions
        """
        pass

    def generate_fitted_vs_actual(data: PlotData) -> go.Figure:
        """Generate time series plot comparing fitted vs actual values.
        
        Args:
            data: PlotData instance containing required data
            
        Returns:
            go.Figure: Line plot comparing predicted and actual values
        """
        pass

    def generate_diagnostic_plot(data: PlotData) -> go.Figure:
        """Generate diagnostic scatter plot of fitted vs residual values.
        
        Args:
            data: PlotData instance containing required data
            
        Returns:
            go.Figure: Scatter plot with trend line
        """
        pass

    def generate_immediate_vs_carryover(data: PlotData) -> go.Figure:
        """Generate stacked bar chart comparing immediate vs carryover effects.
        
        Args:
            data: PlotData instance containing required data
            
        Returns:
            go.Figure: Stacked bar plot of effect types
        """
        pass

    def generate_adstock_rate(data: PlotData) -> go.Figure:
        """Generate plot showing adstock rates over time by channel.
        
        Args:
            data: PlotData instance containing required data
            
        Returns:
            go.Figure: Line plot of adstock decay rates
        """
        pass