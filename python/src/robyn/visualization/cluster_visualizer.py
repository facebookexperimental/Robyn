# pyre-strict
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from robyn.data.entities.enums import DependentVarType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult

class ClusterVisualizer:

    def __init__(self, pareto_result: ParetoResult, clustered_result: ClusteredResult, mmm_data: MMMData):
        self.pareto_result = pareto_result
        self.clustered_result = clustered_result
        self.mmm_data = mmm_data

    def plot_wss(self) -> None:
        """
        Plot the Within-Cluster Sum of Squares (WSS) for different numbers of clusters.
        """
        pass

    def plot_correlations(self) -> None:
        """
        Plot the correlations between variables for each cluster.
        """
        pass

    def plot_cluster_means(self) -> None:
        """
        Plot the mean values of variables for each cluster.
        """
        pass

    def plot_dimensionality_reduction(self) -> None:
        """
        Plot the results of dimensionality reduction (PCA or t-SNE).
        """
        pass

    def plot_confidence_intervals(self) -> None:
        """
        Creates a plot of the bootstrapped confidence intervals for model performance metrics.

        Args:
            confidence_data (Dict[str, float]): The data containing confidence intervals for plotting.
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            None
        """
        pass

    def plot_top_solutions(self) -> None:
        """
        Creates plots for the top solutions based on their performance metrics.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            None
        """
        pass

    def generate_bootstrap_confidence(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """Generate error bar plot showing bootstrapped ROI/CPA confidence intervals."""
        # Check if we have confidence intervals
        x_decomp_agg = self.pareto_result.x_decomp_agg
        if 'ci_low' not in x_decomp_agg.columns:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No bootstrap results", ha='center', va='center')
                return fig
            else:
                ax.text(0.5, 0.5, "No bootstrap results", ha='center', va='center')
                return None
                
        # Get specific model ID (similar to sid in R code)
        model_id = x_decomp_agg['solID'].iloc[0]

        # Filter data for specific model
        bootstrap_data = (x_decomp_agg[
            (~x_decomp_agg['ci_low'].isna()) & 
            (x_decomp_agg['solID'] == model_id)
        ][['rn', 'solID', 'boot_mean', 'ci_low', 'ci_up']])
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, min(8, 3 + len(bootstrap_data) * 0.3)))
        else:
            fig = None

        # Set clean background
        ax.set_facecolor('white')
        
        # Determine metric type
        metric_type = "ROI" if (self.mmm_data and 
                            hasattr(self.mmm_data.mmmdata_spec, 'dep_var_type') and 
                            self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE) else "CPA"
        
        # Create plot with proper y-axis labels
        y_pos = range(len(bootstrap_data))
        
        # Add error bars
        ax.errorbar(
            x=bootstrap_data['boot_mean'],
            y=y_pos,
            xerr=[(bootstrap_data['boot_mean'] - bootstrap_data['ci_low']),
                (bootstrap_data['ci_up'] - bootstrap_data['boot_mean'])],
            fmt='o',
            color='black',
            capsize=3,
            markersize=3,
            elinewidth=1,
            zorder=3
        )
        
        # Add labels
        for i, row in enumerate(bootstrap_data.itertuples()):
            # Mean value
            ax.text(row.boot_mean, i,
                    f"{float(f'{row.boot_mean:.2g}')}",
                    va='bottom', ha='center',
                    fontsize=10,
                    color='black')
            
            # CI values
            ax.text(row.ci_low, i,
                    f"{float(f'{row.ci_low:.2g}')}",
                    va='center', ha='right',
                    fontsize=9,
                    color='black')
            
            ax.text(row.ci_up, i,
                    f"{float(f'{row.ci_up:.2g}')}",
                    va='center', ha='left',
                    fontsize=9,
                    color='black')
        
        # Set y-axis labels properly
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bootstrap_data['rn'], fontsize=9)
        
        # Remove unnecessary spines but keep left spine for labels
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add ROAS reference line if applicable
        if metric_type == "ROI":
            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, zorder=2)
        
        # Set title
        title = f"In-cluster bootstrapped {metric_type} with 95% CI & mean"
        if self.clustered_result is not None:
            cluster_info = self.clustered_result.cluster_data
            if not cluster_info.empty:
                cluster_txt = f" {cluster_info['cluster'].iloc[0]}"
                n_models = len(cluster_info)
                if n_models > 1:
                    title += f" ({n_models} IDs)"
        ax.set_title(title, pad=20, fontsize=11)
        
        # Set proper x limits
        x_min = bootstrap_data['ci_low'].min()
        x_max = bootstrap_data['ci_up'].max()
        margin = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - margin, x_max + margin)
        
        # Add x grid
        ax.grid(True, axis='x', color='lightgray', linestyle='-', alpha=0.3, zorder=1)
        ax.set_axisbelow(True)
        
        if fig:
            plt.tight_layout()
            return fig
        return None