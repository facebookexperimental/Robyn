# pyre-strict
from robyn.modeling.entities.clustering_results import ClusteredResult


class ClusterVisualizer:
    """
    Class for visualizing clustering results.
    """

    def __init__(self, results: ClusteredResult):
        """
        Initialize the ClusterVisualizer with clustering results.

        Args:
            results (ClusteredResult): Results of the clustering process.
        """
        self.results = results

    def plot_wss(self) -> None:
        """
        Plot the Within-Cluster Sum of Squares (WSS) for different numbers of clusters.
        """
        pass

    def plot_correlations(self) -> None:
        """
        Plot the correlations between variables for each cluster.
        """
        pass

    def plot_cluster_means(self) -> None:
        """
        Plot the mean values of variables for each cluster.
        """
        pass

    def plot_dimensionality_reduction(self) -> None:
        """
        Plot the results of dimensionality reduction (PCA or t-SNE).
        """
        pass

    def plot_confidence_intervals(
        self, confidence_data: Dict[str, float], config: ClusteringConfig
    ) -> None:
        """
        Creates a plot of the bootstrapped confidence intervals for model performance metrics.

        Args:
            confidence_data (Dict[str, float]): The data containing confidence intervals for plotting.
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            None
        """
        pass

    def plot_top_solutions(self, config: ClusteringConfig) -> None:
        """
        Creates plots for the top solutions based on their performance metrics.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            None
        """
        pass
