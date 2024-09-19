# pyre-strict

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List


@dataclass
class KMeansResults:
    """
    Results of K-means clustering.

    Attributes:
        cluster_assignments (List[int]): Cluster assignments for each data point.
        centroids (np.ndarray): Coordinates of cluster centroids.
        inertia (float): Sum of squared distances of samples to their closest cluster center.
    """

    cluster_assignments: List[int]
    centroids: np.ndarray
    inertia: float


@dataclass
class ConfidenceIntervals:
    """
    Confidence intervals for clustered data.

    Attributes:
        lower_bounds (pd.DataFrame): Lower bounds of confidence intervals.
        upper_bounds (pd.DataFrame): Upper bounds of confidence intervals.
        means (pd.DataFrame): Mean values for each cluster.
    """

    lower_bounds: pd.DataFrame
    upper_bounds: pd.DataFrame
    means: pd.DataFrame


class ClusteringResults:
    """
    Class to hold and manage the results of the clustering process.
    """

    def __init__(self, clustered_data: pd.DataFrame, confidence_intervals: ConfidenceIntervals, top_models: List[str]):
        """
        Initialize the ClusteringResults with the clustering outputs.

        Args:
            clustered_data (pd.DataFrame): Data with cluster assignments.
            confidence_intervals (ConfidenceIntervals): Confidence intervals for each cluster.
            top_models (List[str]): IDs of top performing models.
        """
        pass

    def export_results(self, path: str) -> None:
        """
        Export the clustering results to files.

        Args:
            path (str): Directory path to save the results.
        """
        pass
