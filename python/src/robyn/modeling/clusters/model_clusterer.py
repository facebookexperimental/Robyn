# pyre-strict

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class ClusteringResult:
    data: pd.DataFrame
    df_cluster_ci: pd.DataFrame
    n_clusters: int
    boot_n: int
    sim_n: int
    errors_weights: List[float]
    clusters_means: pd.DataFrame
    models: pd.DataFrame
    wss_data: pd.DataFrame
    pca_data: pd.DataFrame
    tsne_data: pd.DataFrame


class ModelClusterer:
    def __init__(self, input_data: pd.DataFrame, dep_var_type: str, all_media: List[str]):
        self.input_data = input_data
        self.dep_var_type = dep_var_type
        self.all_media = all_media

    def cluster_models(
        self,
        cluster_by: str = "hyperparameters",
        k: str = "auto",
        wss_var: float = 0.06,
        max_clusters: int = 10,
        limit: int = 1,
        weights: List[float] = [1, 1, 1],
        dim_red: str = "PCA",
        quiet: bool = False,
        seed: int = 123,
    ) -> ClusteringResult:
        """
        Perform clustering on the models to reduce the number and create bootstrapped confidence intervals.

        Args:
            cluster_by (str): Either "performance" or "hyperparameters".
            k (str or int): Number of clusters or "auto".
            wss_var (float): WSS variance threshold for automatic k selection.
            max_clusters (int): Maximum number of clusters.
            limit (int): Top N results per cluster.
            weights (List[float]): Weights for NRMSE, DECOMP.RSSD, and MAPE.
            dim_red (str): Dimension reduction method, either "PCA" or "tSNE".
            quiet (bool): If True, suppress output messages.
            seed (int): Random seed for reproducibility.

        Returns:
            ClusteringResult: Object containing clustering results.
        """
        # Implementation here
        pass

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for clustering"""
        # Implementation here
        pass

    def _perform_clustering(self, data: pd.DataFrame, k: int) -> Dict[str, Any]:
        """Perform K-means clustering"""
        # Implementation here
        pass

    def _calculate_confidence_intervals(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate confidence intervals for clustered data"""
        # Implementation here
        pass

    def _select_top_models(self, cluster_data: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Select top models based on error scores"""
        # Implementation here
        pass

    @staticmethod
    def errors_scores(df: pd.DataFrame, balance: List[float] = [1, 1, 1], ts_validation: bool = True) -> np.ndarray:
        """Calculate error scores for models"""
        # Implementation here
        pass

    @staticmethod
    def _min_max_norm(x: np.ndarray, min_val: float = 0, max_val: float = 1) -> np.ndarray:
        """Perform min-max normalization"""
        # Implementation here
        pass

    @staticmethod
    def _bootstrap_ci(sample: np.ndarray, boot_n: int, seed: int = 1) -> Dict[str, Any]:
        """Calculate bootstrap confidence intervals"""
        # Implementation here
        pass
