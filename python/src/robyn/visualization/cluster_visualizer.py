# pyre-strict
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from robyn.modeling.clustering.clustering_config import (
    ClusteringConfig,
    DependentVarType,
)
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from scipy import stats


class ClusterVisualizer:

    def __init__(self, pareto_result: Optional[ParetoResult], clustered_result: Optional[ClusteredResult], mmm_data: Optional[MMMData]):
        if clustered_result is not None:
            self.results = clustered_result
        if pareto_result is not None:
            self.pareto_result = pareto_result
        if mmm_data is not None:
            self.mmm_data = mmm_data 

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
        sim_collect: pd.DataFrame,
        confidence_interval_df: pd.DataFrame,
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
        # Determine the type of plot to create
        temp = "CPA" if config.dep_var_type == DependentVarType.CONVERSION else "ROAS"
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the density ridges
        # sns.kdeplot(
        #     data=sim_collect, x="x_sim", y="rn", hue="cluster_title", fill=True, ax=ax
        # )
        kde = stats.gaussian_kde(sim_collect["x_sim"], bw_method=0.1)
        x_grid = np.linspace(
            sim_collect["x_sim"].min(), sim_collect["x_sim"].max(), 1000
        )
        y_grid = kde(x_grid)
        ax.plot(x_grid, y_grid, color="gray", alpha=0.5)
        # Add text labels for the CI
        for i, row in confidence_interval_df.iterrows():
            ax.text(row.boot_mean, row.rn, row.boot_ci, ha="left", va="center")
        # Set the title and labels
        ax.set_title(f"In-Cluster {temp} & Bootstrapped 95% CI")
        ax.set_xlabel(temp)
        ax.set_ylabel("Density")
        # Add a vertical line at x=1
        ax.axvline(x=1, color="gray", linestyle="--")
        # Show the plot
        plt.show()
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
        # merged_df = pd.merge(df, top_sols.iloc[:, :3], on="solID", how='left')
        # merged_df['alpha'] = merged_df.apply(lambda row: 0.6 if pd.isna(row['cluster']) else 1, axis=1)
        # merged_df['label'] = merged_df.apply(lambda row: f"[{row['cluster']}.{row['rank']}]" if not pd.isna(row['cluster']) else None, axis=1)
        merged_df = pd.merge(df, top_sols.iloc[:, :3], on="solID", how='left')
        # Create a new column 'alpha'
        merged_df["alpha"] = merged_df.apply(
            lambda row: 0.6 if pd.isna(row["cluster"]) else 1, axis=1
        )
        # Create a new column 'label'
        merged_df["label"] = merged_df.apply(
            lambda row: (
                f'[{row["cluster"]}.{row["rank"]}]'
                if not pd.isna(row["cluster"])
                else None
            ),
            axis=1,
        )
        # Plot the data
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=merged_df, x="nrmse", y="decomp.rssd", hue="cluster", alpha="alpha"
        )
        sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
        # Add text labels
        for i, row in merged_df.iterrows():
            if not pd.isna(row["label"]):
                plt.text(
                    row["nrmse"],
                    row["decomp.rssd"],
                    row["label"],
                    ha="left",
                    va="center",
                )
        # Set title and labels
        plt.title(f"Selecting Top {limit} Performing Models by Cluster")
        plt.xlabel("NRMSE")
        plt.ylabel("DECOMP.RSSD")
        # Set caption
        caption = f"Weights: NRMSE {round(100 * balance[0])}%, DECOMP.RSSD {round(100 * balance[1])}%, MAPE {round(100 * balance[2])}%"
        plt.figtext(0.5, 0.01, caption, ha="center")
        # Show the plot
        plt.show()
        return plt.gcf()

    def plot_topsols_rois(
        self,
        df: pd.DataFrame,
        top_sols: pd.DataFrame,
        all_media: list[str],
        limit: int = 1,
    ) -> Figure:
        """
        Plot the top performing models by media.
        Args:
            df (pd.DataFrame): The input DataFrame.
            top_sols (pd.DataFrame): The top solutions DataFrame.
            all_media (list[str]): A list of media names.
            limit (int, optional): The number of top performing models to select. Defaults to 1.
        Returns:
            Figure: The matplotlib figure object containing the plot.
        """
        # Select real columns from df
        real_cols = [
            col for col in df.columns if col not in ["mape", "nrmse", "decomp.rssd"]
        ]
        real_rois = df[real_cols].copy()
        real_rois.columns = ["real_" + col for col in real_rois.columns]
        # Merge DataFrames
        merged_df = pd.merge(
            top_sols, real_rois, left_on="solID", right_on="real_solID"
        )
        # Create a new column 'label'
        merged_df["label"] = merged_df.apply(
            lambda row: f'[{row["cluster"]}.{row["rank"]}] {row["solID"]}', axis=1
        )
        # Melt the DataFrame
        melted_df = pd.melt(
            merged_df,
            id_vars=["label"],
            value_vars=[col for col in merged_df.columns if col.startswith("real_")],
        )
        # Filter and rename columns
        filtered_df = melted_df[melted_df["variable"].str.contains("real_")]
        filtered_df["media"] = filtered_df["variable"].apply(
            lambda x: x.replace("real_", "")
        )
        # Filter data based on all_media
        filtered_df = filtered_df[filtered_df["media"].isin(all_media)]
        # Sort and limit data
        filtered_df = filtered_df.sort_values(by="value", ascending=False).head(
            limit * len(all_media)
        )
        # Plot the data
        plt.figure(figsize=(10, 8))
        sns.barplot(data=filtered_df, x="media", y="value", hue="label")
        plt.xlabel("")
        plt.ylabel("Mean metric per media")
        plt.title("Top Performing Models")
        plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1), title="Label")
        plt.show()
        # Facet grid
        g = sns.FacetGrid(filtered_df, col="label", height=5, aspect=0.7)
        g.map(sns.barplot, "media", "value")
        g.set_titles(col_template="{col_name}")
        g.figure.tight_layout()
        plt.show()
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