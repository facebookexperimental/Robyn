# pyre-strict
import pandas as pd
import numpy as np
from typing import List
from robyn.modeling.clustering.clustering_config import ClusteringConfig
from robyn.modeling.entities.clustering_results import ClusteringResults, KMeansResults, ConfidenceIntervals
from robyn.modeling.entities.modeloutputs import ModelOutputs


class ModelClusterer:
    """
    Main class for performing clustering on model results.
    """

    def __init__(self, config: ClusteringConfig):
        """
        Initialize the ModelClusterer with the given configuration.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.
        """
        pass

    def cluster_models(self, model_outputs: ModelOutputs) -> ClusteringResults:
        """
        Perform clustering on the given model outputs.

        Args:
            model_outputs (ModelOutputs): Results from the model run.

        Returns:
            ClusteringResults: Results of the clustering process.
        """
        pass

    def _preprocess_model_data_for_clustering(self, model_outputs: ModelOutputs) -> pd.DataFrame:
        """
        Preprocess and transform model output data for clustering analysis.

        This method prepares the data from model outputs for clustering by performing
        the following steps:
        1. Selects relevant features based on the clustering configuration (performance or hyperparameters).
        2. Handles any missing data or outliers.
        3. Normalizes or standardizes the data as needed for clustering algorithms.
        4. Applies dimensionality reduction if specified in the configuration.

        The exact preprocessing steps depend on the ClusteringConfig settings, particularly
        the 'cluster_by' and 'dim_reduction' parameters.

        Args:
            model_outputs (ModelOutputs): Results from the model run, containing performance
                metrics, hyperparameters, and other relevant data for each model.

        Returns:
            pd.DataFrame: A preprocessed and transformed DataFrame ready for clustering analysis.
                Each row represents a model, and columns represent the features used for clustering.
        """
        pass

    def _perform_kmeans(self, data: pd.DataFrame) -> KMeansResults:
        """
        Perform K-means clustering on the prepared data.

        Args:
            data (pd.DataFrame): Prepared data for clustering.

        Returns:
            KMeansResults: Results of K-means clustering.
        """
        pass

    def _calculate_confidence_intervals(self, data: pd.DataFrame, clusters: List[int]) -> ConfidenceIntervals:
        """
        Calculate confidence intervals for each cluster.

        Args:
            data (pd.DataFrame): Clustered data.
            clusters (List[int]): Cluster assignments.

        Returns:
            ConfidenceIntervals: Confidence intervals for each cluster.
        """
        pass

    def _select_top_models(self, data: pd.DataFrame, clusters: List[int]) -> List[str]:
        """
        Select top performing models from each cluster.

        Args:
            data (pd.DataFrame): Clustered data.
            clusters (List[int]): Cluster assignments.

        Returns:
            List[str]: IDs of top performing models.
        """
        pass
