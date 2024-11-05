# pyre-strict
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from matplotlib.figure import Figure


@dataclass
class ClusterPlotResults:
    """Collection of visualization data frames generated during cluster analysis.

    Args:
        top_solutions_errors_plot: Data for model error distribution plots.
        top_solutions_rois_plot: Data for ROI comparison plots of top models.
    """

    top_solutions_errors_plot: Optional[Figure] = None
    top_solutions_rois_plot: Optional[Figure] = None


@dataclass
class ClusterConfidenceIntervals:
    """Statistical confidence intervals for cluster analysis results.

    Args:
        cluster_confidence_interval_df: DataFrame containing confidence intervals for cluster metrics.
        boot_n: Number of bootstrap iterations used for CI calculations.
        sim_n: Number of simulations performed for CI estimation.
        clusters_confidence_interval_plot: Data for confidence interval plots by cluster.
    """

    cluster_confidence_interval_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    boot_n: int = 0
    sim_n: int = 0
    clusters_confidence_interval_plot: Optional[Figure] = None


@dataclass
class DimentionalityReductionResults:
    """Principal Component Analysis or t-Distributed Stochastic Neighbor Embedding results from clustering process.

    Args:
        explained_variance: Series containing explained variance ratios.
        df: DataFrame with PCA-transformed data.
        plot_explained_variance: Optional DataFrame with explained variance visualization data.
        plot: Optional dictionary containing additional PCA plot data.
    """

    explained_variance: pd.Series = field(default_factory=pd.Series)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    plot_explained: Optional[Figure] = None
    plot: Optional[Figure] = None


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
        wss: plot containing within-sum-of-squares metrics.
        correlations: plot of correlation analysis between clusters.
        clusters_pca: Optional PCA dimensionality reduction results.
        clusters_tsne: Optional t-SNE dimensionality reduction results.
        plots: Optional collection of visualization data frames.
    """

    cluster_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_solutions: pd.DataFrame = field(default_factory=pd.DataFrame)
    cluster_ci: ClusterConfidenceIntervals = field(
        default_factory=ClusterConfidenceIntervals
    )
    n_clusters: int = 0
    errors_weights: List[float] = field(default_factory=list)
    clusters_means: pd.DataFrame = field(default_factory=pd.DataFrame)
    wss: Figure = field(default_factory=Figure)
    correlations: Optional[Figure] = None
    clusters_pca: Optional[DimentionalityReductionResults] = None
    clusters_tsne: Optional[DimentionalityReductionResults] = None
    plots: ClusterPlotResults = field(default_factory=ClusterPlotResults)
