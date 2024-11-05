from typing import Dict, List

import pandas as pd
from robyn.modeling.clustering.clustering_config import ClusteringConfig
from robyn.modeling.entities.clustering_results import ClusteredResult

from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.pareto.pareto_optimizer import ParetoResult


class ClusterBuilder:
    """
    An interface for clustering models based on performance metrics and hyperparameters.

    This interface defines methods for clustering models, calculating confidence intervals,
    and generating error scores, among other functionalities.
    """

    def __init__(self, pareto_result: ParetoResult):
        """
        Initializes the ClusterBuilder with global instances of ModelOutputs and ParetoResult.

        Args:
            pareto_result (ParetoResult): The results of the Pareto optimization process.
        """
        self.pareto_result: ParetoResult = pareto_result

    def cluster_models(self, config: ClusteringConfig) -> ClusteredResult:
        """
        Clusters models based on specified criteria.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            ClusteredResult: The results of the clustering process, including cluster assignments and confidence intervals.
        """
        pass

    def _calculate_confidence_intervals(
        self, config: ClusteringConfig
    ) -> Dict[str, float]:
        """
        Calculates bootstrapped confidence intervals for model performance metrics.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            Dict[str, float]: A dictionary containing confidence intervals for each model.
        """
        pass

    def _compute_error_scores(
        self, weights: List[float], config: ClusteringConfig
    ) -> pd.DataFrame:
        """
        Computes error scores for the models based on specified performance metrics.

        Args:
            weights (List[float]): A list of weights for the error metrics.
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            pd.DataFrame: A DataFrame containing the computed error scores for each model.
        """
        pass

    def _select_optimal_clusters(
        self, df: pd.DataFrame, config: ClusteringConfig
    ) -> int:
        """
        Selects the optimal number of clusters based on WSS variance.

        Args:
            df (pd.DataFrame): The prepared data for clustering.
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            int: The optimal number of clusters.
        """
        pass

    def _prepare_data_for_clustering(self, config: ClusteringConfig) -> pd.DataFrame:
        """
        Prepares the data frame for clustering based on specified criteria.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            pd.DataFrame: A DataFrame prepared for clustering.
        """
        pass

    def _normalize_values(
        self, values: List[float], min_val: float = 0, max_val: float = 1
    ) -> List[float]:
        """
        Normalizes a list of values to a specified range.

        Args:
            values (List[float]): The values to normalize.
            min_val (float): The minimum value of the normalized range.
            max_val (float): The maximum value of the normalized range.

        Returns:
            List[float]: A list of normalized values.
        """
        pass

    def _select_top_models(
        self, clustered_df: pd.DataFrame, config: ClusteringConfig
    ) -> pd.DataFrame:
        """
        Selects the top models based on their distance to the origin.

        Args:
            clustered_df (pd.DataFrame): The DataFrame containing cluster assignments.
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            pd.DataFrame: DataFrame of top models.
        """
        pass

    def _bootstrap_sampling(self, sample: List[float], boot_n: int) -> Dict[str, float]:
        """
        Performs bootstrap sampling to calculate confidence intervals for a given sample.

        Args:
            sample (List[float]): The sample data for bootstrap sampling.
            boot_n (int): The number of bootstrap samples to generate.

        Returns:
            Dict[str, float]: A dictionary containing bootstrap means and confidence intervals.
        """
        pass
