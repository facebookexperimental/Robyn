# pyre-strict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult

class ClusterVisualizer:

    def __init__(self, pareto_result: ParetoResult, clustered_result: ClusteredResult):
        self.pareto_result = pareto_result
        self.clustered_result = clustered_result

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

    def generate_bootstrap_confidence(self) -> plt.figure:
        """Generate error bar plot showing bootstrapped ROI/CPA confidence intervals.
        
        This visualization matches the R implementation's confidence interval plot from
        robyn_clusters(), showing bootstrapped performance metrics with error bars for 
        each media channel within clusters.
        
        Returns:
            plt.Figure: Error bar plot with confidence intervals
        """
        # Extract bootstrap data from cluster results
        df_cluster_ci = self.clustered_result.df_cluster_ci
        
        if df_cluster_ci is None or len(df_cluster_ci) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No bootstrap results available", 
                   ha='center', va='center')
            return fig

        # Create figure with appropriate size based on number of variables
        n_vars = len(df_cluster_ci['rn'].unique())
        n_clusters = len(df_cluster_ci['cluster'].unique())
        fig_height = max(8, n_vars * 0.5)
        
        fig, axes = plt.subplots(n_clusters, 1, 
                                figsize=(12, fig_height * n_clusters),
                                squeeze=False)

        # Get metric type
        metric = "ROAS" if self.pareto_result.metric_type == "revenue" else "CPA"

        # Plot for each cluster
        for cluster_idx, cluster in enumerate(sorted(df_cluster_ci['cluster'].unique())):
            ax = axes[cluster_idx, 0]
            
            # Filter data for current cluster
            cluster_data = df_cluster_ci[df_cluster_ci['cluster'] == cluster].copy()
            
            # Count models in cluster
            n_models = len(self.clustered_result.data[
                self.clustered_result.data['cluster'] == cluster
            ])
            
            # Sort channels by mean values
            cluster_data['rn'] = pd.Categorical(
                cluster_data['rn'],
                categories=cluster_data.sort_values('boot_mean')['rn'],
                ordered=True
            )

            # Plot error bars
            error_bars = ax.errorbar(
                x=cluster_data['boot_mean'],
                y=cluster_data['rn'],
                xerr=np.vstack([
                    cluster_data['boot_mean'] - cluster_data['ci_low'],
                    cluster_data['ci_up'] - cluster_data['boot_mean']
                ]),
                fmt='o',
                capsize=5,
                markersize=8,
                color='steelblue',
                ecolor='grey',
                alpha=0.7
            )

            # Add mean value labels
            for idx, row in cluster_data.iterrows():
                ax.text(row['boot_mean'], row['rn'], 
                       f"{row['boot_mean']:.2f}",
                       va='center', ha='left',
                       fontweight='bold',
                       fontsize=10)
                
                # Add CI labels
                ax.text(row['ci_low'], row['rn'], 
                       f"{row['ci_low']:.2f}",
                       va='center', ha='right',
                       fontsize=9,
                       color='grey')
                ax.text(row['ci_up'], row['rn'], 
                       f"{row['ci_up']:.2f}",
                       va='center', ha='left',
                       fontsize=9,
                       color='grey')

            # Add reference line at 1 for ROAS
            if metric == "ROAS":
                ax.axvline(x=1, color='grey', linestyle='--', alpha=0.5)

            # Customize plot
            title = (f"Cluster {cluster} ({n_models} models)\n"
                    f"Bootstrapped {metric} [95% CI & mean]")
            ax.set_title(title, pad=20)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.grid(True, alpha=0.3)

            # Adjust plot limits
            x_min = min(cluster_data['ci_low'].min() * 0.9, 0)
            x_max = cluster_data['ci_up'].max() * 1.1
            ax.set_xlim(x_min, x_max)

        # Overall plot adjustments
        plt.tight_layout()
        fig.align_ylabels(axes[:, 0])
        
        return fig    