# pyre-strict
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.pyplot import FixedLocator
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from robyn.modeling.clustering.clustering_config import ClusteringConfig
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from robyn.modeling.entities.ci_collection_data import ConfidenceIntervalCollectionData


class ClusterVisualizer:

    def __init__(self, pareto_result: Optional[ParetoResult], clustered_result: Optional[ClusteredResult]):
        if clustered_result is not None:
            self.results = clustered_result
        if pareto_result is not None:
            self.pareto_result = pareto_result

    def create_wss_plot(self, nclusters: pd.DataFrame, k: Optional[int]) -> Figure:
        """
        Creates a WSS plot for the given DataFrame.

        Args:
            nclusters (pd.DataFrame): The DataFrame containing the data to cluster.
            k (int): The maximum number of clusters to consider.
            seed (int): Random seed for reproducibility.

        Returns:
            plt.Figure: The WSS plot figure.
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=nclusters, x="n", y="wss", marker="o")
        plt.title("Total Number of Clusters")
        plt.suptitle("HINT: Where does the curve level?")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Within Groups Sum of Squares")
        plt.grid(True)
        plt.gcf().set_facecolor("white")

        # If k is determined, add a horizontal line and update the subtitle
        if k is not None:
            yintercept = nclusters.loc[nclusters["n"] == k, "wss"].values[0]
            plt.axhline(y=yintercept, color="red", linestyle="--")
            plt.suptitle(f"Number of clusters selected: {k}")

        fig = plt.gcf()  
        plt.close(fig)
        return fig # Return the current figure

    def plot_confidence_intervals(
        self,
        ci_results: ConfidenceIntervalCollectionData,
        config: ClusteringConfig,
    ) -> Figure:
        """
        Creates a plot of the bootstrapped confidence intervals for model performance metrics.

        Args:
            sim_collect (pd.DataFrame): The DataFrame containing the bootstrapped data,
            confidence_interval_df (pd.DataFrame): The data containing confidence intervals for plotting.
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            Figure: The matplotlib figure object containing the plot.
        """
        sim_collect = ci_results.sim_collect
        confidence_interval_df = ci_results.confidence_interval_df
        
        # Determine metric type
        temp = "CPA" if config.dep_var_type == "conversion" else "ROAS"
        
        # Drop NA values
        confidence_interval_df = confidence_interval_df.dropna()
        
        # Setup the figure
        unique_clusters = sorted(sim_collect['cluster_title'].unique())
        n_clusters = len(unique_clusters)
        n_cols = min(2, n_clusters)
        n_rows = int(np.ceil(n_clusters / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        if n_clusters == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Calculate figure-level x_range for consistent scaling
        x_min, x_max = sim_collect['x_sim'].min(), sim_collect['x_sim'].max()
        
        # Main plotting loop
        for idx, (cluster, ax) in enumerate(zip(unique_clusters, axes)):
            # Filter data for current cluster
            cluster_data = sim_collect[sim_collect['cluster_title'] == cluster].copy()
            cluster_ci = confidence_interval_df[confidence_interval_df['cluster_title'] == cluster]
            
            # Get unique rn values preserving original order
            unique_rns = pd.Categorical(cluster_data['rn']).categories
            
            # Create data matrix for efficient plotting
            data_matrix = []
            y_positions = []
            
            for i, rn_val in enumerate(unique_rns):
                rn_data = cluster_data[cluster_data['rn'] == rn_val]['x_sim'].values
                if len(rn_data) > 0:
                    # Calculate KDE more efficiently
                    kde = stats.gaussian_kde(rn_data, bw_method=0.5)
                    x_range = np.linspace(x_min, x_max, 200)
                    density = kde(x_range)
                    
                    # Scale density for better visualization
                    density = density * 3  # Matches scale=3 in R code
                    
                    data_matrix.append(density)
                    y_positions.append(i)
            
            # Convert to numpy array for faster operations
            data_matrix = np.array(data_matrix)
            
            # Plot ridges
            for i, density in enumerate(data_matrix):
                y_base = y_positions[i]
                x_range = np.linspace(x_min, x_max, len(density))
                
                # Fill area
                ax.fill_between(
                    x_range,
                    y_base,
                    y_base + density,
                    alpha=0.5,
                    color='#4682B4'  # Steel blue color
                )
                
                # Add outline
                ax.plot(x_range, y_base + density, color='#4682B4', linewidth=0.5)
            
            # Add CI text for each unique rn value
            for i, rn_val in enumerate(unique_rns):
                # Get the corresponding CI for the current rn value
                cluster_ci_rn = cluster_ci[cluster_ci['rn'] == rn_val]
                if not cluster_ci_rn.empty:
                    y_pos = y_positions[i] + 0.1
                    x_pos = cluster_ci_rn['boot_mean'].iloc[0] - 0.02
                    ax.text(
                        x_pos,
                        y_pos,
                        cluster_ci_rn['boot_ci'].iloc[0],
                        color='#4D4D4D',
                        size=9,
                    )
            
            # Add vertical line at x=1
            ax.axvline(x=1, linestyle='--', linewidth=0.5, color='#BFBFBF')
            
            # Set axis properties
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.5, len(unique_rns))
            ax.yaxis.set_major_locator(FixedLocator(y_positions))
            ax.set_yticklabels(unique_rns)
            
            # Add horizontal line for ROAS
            if temp == "ROAS":
                ax.axhline(y=1, alpha=0.5, color='#808080', linestyle='--')
            
            # Style the plot
            ax.set_title(f"Cluster {cluster}")
            ax.set_xlabel(temp)
            ax.set_ylabel("Density")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('white')
        
        # Hide empty subplots
        for ax in axes[n_clusters:]:
            ax.set_visible(False)
        
        # Set overall title and subtitle
        fig.suptitle(
            f"In-Cluster {temp} & Bootstrapped 95% CI\n"
            "Sampling distribution of cluster mean",
            y=0.95
        )
        
        # Add caption
        fig.text(
            0.5,
            0.02,
            f"Based on {ci_results.boot_n} bootstrap results with {ci_results.sim_n} simulations",
            ha='center'
        )
        
        plt.tight_layout(rect=(0, 0.05, 1, 0.92))
        
        return fig

    def plot_top_solutions_errors(
        self,
        df: pd.DataFrame,
        top_sols: pd.DataFrame,
        limit: int = 1,
        balance: list[float] = [1, 1, 1],
    ) -> Figure:
        """
        Plot the top solutions errors.
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            top_sols (pd.DataFrame): The DataFrame containing the top solutions.
            limit (int): The number of top solutions to plot. Default is 1.
            balance (Dict[str, float]): The weights for the NRMSE, DECOMP.RSSD, and MAPE metrics. Default is {"nrmse": 1.0, "decomp_rssd": 1.0, "mape": 1.0}.
        Returns:
            Figure: The matplotlib figure object containing the plot.
        """
        # Normalize the balance weights
        balance = [b / sum(balance) for b in balance]

        # Merge the DataFrames
        merged_df = pd.merge(df, top_sols.iloc[:, :3], on="solID", how='left')
        merged_df["alpha"] = np.where(merged_df["cluster"].isna(), 0.6, 1)
        merged_df["label"] = merged_df.apply(
            lambda row: f"[{row['cluster']}:{row['rank']}]" if not pd.isna(row["cluster"]) else None,
            axis=1,
        )

        # Create a new column 'highlight' to track whether each row should be highlighted or not
        merged_df["highlight"] = merged_df["label"].notna()

        # Plot the data
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=merged_df[~merged_df["highlight"]], x="nrmse", y="decomp.rssd", color="gray"
        )
        sns.scatterplot(
            data=merged_df[merged_df["highlight"]], x="nrmse", y="decomp.rssd", hue="label", 
            palette=sns.color_palette("husl", n_colors=len(merged_df.loc[merged_df['highlight'], 'label'].unique()), as_cmap=False)
        )

        for i, row in merged_df.iterrows():
            if row["highlight"]:
                plt.scatter(row['nrmse'], row['decomp.rssd'], alpha=row['alpha'])
        plt.title(f"Selecting Top {limit} Performing Models by Cluster\nBased on minimum (weighted) distance to origin")
        plt.xlabel("NRMSE")
        plt.ylabel("DECOMP.RSSD")
        plt.legend(title="Cluster")
        plt.figtext(0.5, 0.01, 
                    f"Weights: NRMSE {round(100 * balance[0])}%, DECOMP.RSSD {round(100 * balance[1])}%, MAPE {round(100 * balance[2])}%", 
                    ha="center", fontsize=10)
        return plt.gcf()

    def plot_topsols_rois(
        self,
        df: pd.DataFrame,
        top_sols: pd.DataFrame,
        all_media: list[str]
    ) -> Figure:
        """
        Plot the top performing models by media.
        Args:
            df (pd.DataFrame): The input DataFrame.
            top_sols (pd.DataFrame): The top solutions DataFrame.
            all_media (list[str]): A list of media names.
            limit (int, optional): The number of top performing models to select. Defaults to 1.
        Returns:
            plt.Figure: The matplotlib figure object containing the plot.
        """
        # Select real columns from df
        real_rois = df.drop(columns=["mape", "nrmse", "decomp.rssd"]).copy()
        real_rois.columns = ["real_" + col for col in real_rois.columns]
        # Merge DataFrames
        merged_df = pd.merge(top_sols, real_rois, left_on="solID", right_on="real_solID", how="left")
        # Create a new column 'label'
        merged_df["label"] = merged_df.apply(
            lambda row: f'[{row["cluster"]}:{row["rank"]}] {row["solID"]}', axis=1
        )
        # Melt the DataFrame
        columns_to_melt = [col for col in merged_df.columns if any(media in col for media in all_media)]
        melted_df = pd.melt(
            merged_df,
            id_vars=["label"],
            value_vars=columns_to_melt,
            var_name="media",
            value_name="perf"
        )
        filtered_df = melted_df[melted_df["media"].str.contains("real_")]
        filtered_df.loc[:, "media"] = filtered_df["media"].apply(lambda x: x.replace("real_", ""))

        # Plot the data
        # Facet grid
        g = sns.FacetGrid(filtered_df, row="label", height=7, aspect=1)
        g.map(sns.barplot, "perf", "media", order=filtered_df["media"].unique())
        g.set_titles(col_template="{col_name}")
        g.figure.suptitle("Top Performing Models by Media", ha='center', va='top', fontsize=20)
        g.figure.tight_layout()
        for ax in g.axes.flat:
            ax.tick_params(axis='x', bottom=True, which='both')
            ax.set_xlabel("Mean metric per media")
            ax.set_ylabel("")
            ax.set_title(ax.get_title(), rotation=270, ha='right', x=1, y=0.5)
        
        g.figure.subplots_adjust(top=0.95, hspace=0.1)
        return plt.gcf()

    def create_correlations_heatmap(self, correlations: pd.DataFrame) -> Figure:
        """
        Creates a heatmap for the correlations.

        Args:
            correlations (pd.DataFrame): The DataFrame containing correlation values.

        Returns:
            plt.Figure: The heatmap figure.
        """
        raise NotImplementedError

    def plot_dimensionality_reduction(self) -> None:
        """
        Plot the results of dimensionality reduction (PCA or t-SNE).
        """
        raise NotImplementedError

    def generate_bootstrap_confidence(self) -> plt.figure:
        """Generate error bar plot showing bootstrapped ROI/CPA confidence intervals.
        
        Returns:
            plt.Figure: The generated figure.
        """
        fig, ax = plt.subplots()
        # Add plotting logic here
        return fig    
