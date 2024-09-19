# pyre-strict
from robyn.modeling.entities.clustering_results import ClusteringResults


class ClusterVisualizer:
    """
    Class for visualizing clustering results.
    """

    def __init__(self, results: ClusteringResults):
        """
        Initialize the ClusterVisualizer with clustering results.

        Args:
            results (ClusteringResults): Results of the clustering process.
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

    def plot_confidence_intervals(self) -> None:
        """
        Plot the confidence intervals for each cluster.
        """
        pass

    def plot_top_models_errors(self) -> None:
        """
        Plot the errors of the top models selected from each cluster.
        """
        pass

    def plot_top_models_performance(self) -> None:
        """
        Plot the performance metrics of the top models selected from each cluster.
        """
        pass
