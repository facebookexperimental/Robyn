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
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from scipy import stats


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

    def generate_bootstrap_confidence(self) -> plt.figure:
        """Generate error bar plot showing bootstrapped ROI/CPA confidence intervals.
        
        Returns:
            plt.Figure: The generated figure.
        """
        fig, ax = plt.subplots()
        # Add plotting logic here
        return fig    
