import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

class AllocationPlotter:
    """Generates plots for allocation results."""
    
    def create_plots(
        self,
        optimal_result: Dict[str, Any],
        initial_metrics: Dict[str, Any],
        final_metrics: Dict[str, Any],
        config: Any
    ) -> Dict[str, plt.Figure]:
        """Create all allocation plots."""
        plots = {}
        
        # Create spend comparison plot
        plots["spend_comparison"] = self.plot_spend_comparison(
            optimal_result["allocations"],
            initial_metrics["spends"]
        )
        
        # Create response curves
        plots["response_curves"] = self.plot_response_curves(
            optimal_result["curves"],
            optimal_result["allocations"]
        )
        
        # Create ROI comparison
        plots["roi_comparison"] = self.plot_roi_comparison(
            final_metrics,
            initial_metrics
        )
        
        return plots

    def plot_spend_comparison(
        self,
        optimal_allocations: pd.DataFrame