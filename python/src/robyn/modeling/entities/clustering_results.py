# pyre-strict
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class ClusterPlotResults:
    """Represents the plots generated during the clustering process."""

    plot_clusters_ci: Optional[pd.DataFrame] = None
    plot_models_errors: Optional[pd.DataFrame] = None
    plot_models_rois: Optional[pd.DataFrame] = None


@dataclass
class ClusterConfidenceIntervals:
    """Represents the confidence intervals for the clusters."""

    cluster_ci: pd.DataFrame
    boot_n: int
    sim_n: int


@dataclass
class PCAResults:
    """PCA results matching R output structure."""

    pca_explained: pd.Series
    pcadf: pd.DataFrame
    plot_explained: Optional[pd.DataFrame] = None
    plot: Optional[Dict] = None  # Keep as Dict since it's plot data


@dataclass
class ClusteredResult:
    """
    Represents the overall results of the clustering process.
    Structure matches exactly what's in the R output.
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
