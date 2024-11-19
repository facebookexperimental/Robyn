# pyre-strict

from dataclasses import dataclass
from enum import Enum
from sys import maxsize
from typing import List, Literal, Optional

from robyn.data.entities.enums import DependentVarType


class ClusterBy(Enum):
    PERFORMANCE = "performance"
    HYPERPARAMETERS = "hyperparameters"


@dataclass
class ClusteringConfig:
    """
    Configuration for the clustering process.

    Attributes:
        dep_var_type (DependentVarType): Type of dependent variable (revenue or conversion).
        cluster_by (ClusterBy): Attribute to cluster by (performance or hyperparameters).
        max_clusters (int): Maximum number of clusters to consider.
        limit (int): Top N results per cluster.
        weights (List[float]): Weights for NRMSE, DECOMP.RSSD, and MAPE errors.
        dim_reduction (Literal["PCA", "tSNE"]): Dimensionality reduction technique.
        export (bool): Whether to export results.
        seed (int): Random seed for reproducibility.
    """

    weights: List[float]
    dep_var_type: DependentVarType
    cluster_by: ClusterBy = ClusterBy.HYPERPARAMETERS
    max_clusters: int = 10
    min_clusters: int = 3
    k_clusters: int = maxsize
    limit: int = 1
    dim_reduction: Literal["PCA", "tSNE"] = "PCA"
    export: bool = False
    seed: int = 123
    all_media: Optional[List[str]] = None

    def __str__(self) -> str:
        """Returns a human-readable string representation of the clustering configuration."""
        return (
            f"ClusteringConfig(\n"
            f"  dep_var_type: {self.dep_var_type}\n"
            f"  cluster_by: {self.cluster_by.value}\n"
            f"  weights: {self.weights}\n"
            f"  max_clusters: {self.max_clusters}\n"
            f"  min_clusters: {self.min_clusters}\n"
            f"  k_clusters: {self.k_clusters}\n"
            f"  limit: {self.limit}\n"
            f"  dim_reduction: {self.dim_reduction}\n"
            f"  export: {self.export}\n"
            f"  seed: {self.seed}\n"
            f"  all_media: {self.all_media}\n"
            ")"
        )
