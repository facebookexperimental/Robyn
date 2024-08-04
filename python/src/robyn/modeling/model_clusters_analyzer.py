from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

class ModelClustersAnalyzer:
    def __init__(self) -> None:
        pass

    def model_clusters_analyze(
        self,
        input: Dict[str, Any],
        dep_var_type: str,
        cluster_by: str = 'hyperparameters',
        all_media: Optional[List[str]] = None,
        k: str = 'auto',
        limit: int = 1,
        weights: Optional[Dict[str, float]] = None,
        dim_red: str = 'PCA',
        quiet: bool = False,
        export: bool = False,
        seed: int = 123
    ) -> Dict[str, Any]:
        """
        Clusters the data based on specified parameters and returns a dictionary containing various outputs.

        Args:
            input: The input data, either a model_clusters_analyze object or a dataframe.
            dep_var_type: The type of dependent variable ('continuous' or 'categorical').
            cluster_by: The variable to cluster by, either 'hyperparameters' or 'performance'.
            all_media: The list of media variables.
            k: The number of clusters.
            limit: The maximum number of top solutions to select.
            weights: The weights for balancing the clusters.
            dim_red: The dimensionality reduction technique to use.
            quiet: Whether to suppress print statements.
            export: Whether to export the results.
            seed: The random seed for reproducibility.

        Returns:
            A dictionary containing various outputs such as cluster data, cluster confidence intervals, number of clusters, etc.
        """
        pass

    def _determine_optimal_k(self, df: pd.DataFrame, max_clusters: int, random_state: int = 42) -> int:
        """
        Determines the optimal number of clusters using the elbow method.

        Args:
            df: The input dataframe.
            max_clusters: The maximum number of clusters to consider.
            random_state: Random state for reproducibility.

        Returns:
            The optimal number of clusters.
        """
        pass

    def _clusterKmeans_auto(
        self,
        df: pd.DataFrame,
        min_clusters: int = 3,
        limit_clusters: int = 10,
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, int, List[float], Any, np.ndarray, np.ndarray]:
        """
        Performs automatic K-means clustering and dimensionality reduction.

        Args:
            df: The input dataframe.
            min_clusters: The minimum number of clusters.
            limit_clusters: The maximum number of clusters.
            seed: Random seed for reproducibility.

        Returns:
            A tuple containing the clustered dataframe, optimal number of clusters, WSS values, KMeans object, PCA and t-SNE results.
        """
        pass

    def _plot_wss_and_save(self, wss: List[float], path: str, dpi: int = 500, width: int = 5, height: int = 4) -> None:
        """
        Creates and saves a WSS plot.

        Args:
            wss: Array of WSS values.
            path: File path for the saved plot.
            dpi: Dots per inch (resolution) of the saved plot.
            width: Width of the figure in inches.
            height: Height of the figure in inches.
        """
        pass

    def _prepare_df(
        self,
        x: pd.DataFrame,
        all_media: List[str],
        dep_var_type: str,
        cluster_by: str
    ) -> pd.DataFrame:
        """
        Prepares the dataframe for clustering.

        Args:
            x: The input dataframe.
            all_media: List of media variables.
            dep_var_type: The type of dependent variable.
            cluster_by: The variable to cluster by.

        Returns:
            The prepared dataframe.
        """
        pass

    def _clusters_df(
        self,
        df: pd.DataFrame,
        all_paid: List[str],
        balance: Optional[Dict[str, float]],
        limit: int,
        ts_validation: bool
    ) -> pd.DataFrame:
        """
        Selects top models by minimum (weighted) distance to zero.

        Args:
            df: The input dataframe.
            all_paid: List of paid media variables.
            balance: Weights for balancing.
            limit: The maximum number of top solutions to select.
            ts_validation: Whether time series validation is used.

        Returns:
            A dataframe of top solutions.
        """
        pass

    def _confidence_calcs(
        self,
        xDecompAgg: pd.DataFrame,
        df: pd.DataFrame,
        all_paid: List[str],
        dep_var_type: str,
        k: int,
        cluster_by: str,
        boot_n: int = 1000,
        sim_n: int = 10000,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Performs confidence interval calculations.

        Args:
            xDecompAgg: The input data for statistical calculations.
            df: The dataframe with cluster information.
            all_paid: The list of paid values.
            dep_var_type: The type of dependent variable.
            k: The number of clusters.
            cluster_by: The method of clustering.
            boot_n: The number of bootstrap iterations.
            sim_n: The number of simulations.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing confidence interval results, simulation results, and other statistics.
        """
        pass

    def _plot_clusters_ci(
        self,
        sim_collect: pd.DataFrame,
        df_ci: pd.DataFrame,
        dep_var_type: str,
        boot_n: int,
        sim_n: int
    ) -> Any:
        """
        Plots cluster confidence intervals.

        Args:
            sim_collect: The simulation results.
            df_ci: The confidence interval results.
            dep_var_type: The type of dependent variable.
            boot_n: The number of bootstrap iterations.
            sim_n: The number of simulations.

        Returns:
            A plot object.
        """
        pass
