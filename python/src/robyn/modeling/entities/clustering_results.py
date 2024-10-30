# pyre-strict
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class ClusterPlotResults:
    """Collection of visualization data frames generated during cluster analysis.

    Args:
        plot_clusters_ci: Data for confidence interval plots by cluster.
        plot_models_errors: Data for model error distribution plots.
        plot_models_rois: Data for ROI comparison plots of top models.
    """

    plot_clusters_ci: Optional[pd.DataFrame] = None
    plot_models_errors: Optional[pd.DataFrame] = None
    plot_models_rois: Optional[pd.DataFrame] = None


@dataclass
class ClusterConfidenceIntervals:
    """Statistical confidence intervals for cluster analysis results.

    Args:
        cluster_ci: DataFrame containing confidence intervals for cluster metrics.
        boot_n: Number of bootstrap iterations used for CI calculations.
        sim_n: Number of simulations performed for CI estimation.
    """

    cluster_ci: pd.DataFrame
    boot_n: int
    sim_n: int


@dataclass
class PCAResults:
    """Principal Component Analysis results from clustering process.

    Args:
        pca_explained: Series containing explained variance ratios.
        pcadf: DataFrame with PCA-transformed data.
        plot_explained: Optional DataFrame with explained variance visualization data.
        plot: Optional dictionary containing additional PCA plot data.
    """

    pca_explained: pd.Series
    pcadf: pd.DataFrame
    plot_explained: Optional[pd.DataFrame] = None
    plot: Optional[Dict] = None


@dataclass
class ClusteredResult:
    """Complete clustering analysis results from Robyn's clustering process.

    Args:
        cluster_data: DataFrame with primary clustering results and model assignments.
        top_solutions: DataFrame containing best performing models per cluster.
        cluster_ci: Confidence interval calculations for clustering results.
        n_clusters: Number of clusters identified in the analysis.
        errors_weights: List of weights applied to different error metrics.
        clusters_means: DataFrame of mean values for each cluster.
        wss: DataFrame containing within-sum-of-squares metrics.
        correlations: DataFrame of correlation analysis between clusters.
        clusters_pca: Optional PCA dimensionality reduction results.
        clusters_tsne: Optional t-SNE analysis results.
        plots: Optional collection of visualization data frames.
    """

    cluster_data: pd.DataFrame
    top_solutions: pd.DataFrame
    cluster_ci: ClusterConfidenceIntervals
    n_clusters: int
    errors_weights: List[float]
    clusters_means: pd.DataFrame
    wss: pd.DataFrame
    correlations: pd.DataFrame
    clusters_pca: Optional[PCAResults] = None
    clusters_tsne: Optional[pd.DataFrame] = None
    plots: Optional[ClusterPlotResults] = None
