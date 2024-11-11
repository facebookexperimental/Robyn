# pyre-strict

import logging
from dataclasses import dataclass
from sys import maxsize
from typing import Dict, List, Optional

import re
import numpy as np
import pandas as pd
from robyn.modeling.clustering.clustering_config import (
    ClusterBy,
    ClusteringConfig
)
from robyn.data.entities.enums import DependentVarType
from robyn.modeling.entities.clustering_results import (
    ClusterConfidenceIntervals,
    ClusteredResult,
    ClusterPlotResults,
)
from robyn.common.constants import HYPERPARAMETER_NAMES

from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.ci_collection_data import ConfidenceIntervalCollectionData
from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.visualization.cluster_visualizer import ClusterVisualizer
from scipy import stats
from sklearn.cluster import KMeans

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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed output
        self.logger.info("Initializing ClusterBuilder")

        self.pareto_result: ParetoResult = pareto_result
        self.cluster_visualizer = ClusterVisualizer(None, None)

    def cluster_models(self, config: ClusteringConfig) -> ClusteredResult:
        """
        Clusters models based on specified criteria.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            ClusteredResult: The results of the clustering process, including cluster assignments and confidence intervals.
        """
        allSolutions = self.pareto_result.pareto_solutions
        x_decomp_agg = self.pareto_result.x_decomp_agg[
            self.pareto_result.x_decomp_agg["sol_id"].isin(allSolutions)
        ]
        result_hyp_param = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["sol_id"].isin(allSolutions)
        ]

        if config.all_media is None:
            aux = self.pareto_result.media_vec_collect.columns
            type_index = aux.get_loc("type")
            # Select all columns except the first one and those from 'type' to the end
            config.all_media = aux[1:type_index].tolist()

        if config.cluster_by == ClusterBy.HYPERPARAMETERS:
            # Prepare data for clustering
            df = self._prepare_hyperparameter_data_for_clustering(result_hyp_param)
        elif config.cluster_by == ClusterBy.PERFORMANCE:
            # Prepare data for clustering
            df = self._prepare_performance_data_for_clustering(x_decomp_agg, config)
        else:
            raise ValueError("Invalid clustering method")

        ignored_columns = [
            "sol_id",
            "mape",
            "decomp.rssd",
            "nrmse",
            "nrmse_test",
            "nrmse_train",
            "nrmse_val",
            "pareto",
        ]
        limit_clusters = min(len(df) - 1, config.max_clusters)
        # Determine number of clusters if not specified
        config.k_clusters = self._select_optimal_clusters(
            df, config, limit_clusters, ignored_columns
        )

        if (
            config.k_clusters < config.min_clusters
            or config.k_clusters > limit_clusters
        ):
            raise ValueError(
                f"Number of clusters {config.k_clusters} is outside the specified range {config.min_clusters} to {config.max_clusters}."
            )

        result = self.clusterKmeans(
            df=df,
            config=config,
            k=config.k_clusters,
            limit=limit_clusters,
            ignored_columns=ignored_columns,
        )

        grouped = result.cluster_data.groupby("cluster", observed=False)
        # Add a new column 'n' with the count of rows in each group
        result.cluster_data["n"] = grouped["cluster"].transform("size")

        # Select top solutions
        all_paid = [
            col
            for col in result.cluster_data.columns
            if col not in ignored_columns + ["cluster"]
        ]
        top_solutions = self._select_top_solutions(result.cluster_data, config)

        # Calculate confidence intervals
        ci_results: ConfidenceIntervalCollectionData = (
            self._calculate_confidence_intervals(
                x_decomp_agg, result.cluster_data, config, all_paid
            )
        )

        # Prepare plot data
        plots = ClusterPlotResults(
            top_solutions_errors_plot=self.cluster_visualizer.plot_top_solutions_errors(
                df, top_solutions, balance=config.weights
            ),
            top_solutions_rois_plot=self.cluster_visualizer.plot_topsols_rois(
                df, top_solutions, config.all_media
            ),
        )

        # Create confidence intervals object
        
        cluster_ci = ClusterConfidenceIntervals(
            cluster_confidence_interval_df=ci_results.confidence_interval_df.reset_index(
                drop=True
            ).drop(
                columns=["cluster_title"]
            ),
            boot_n=ci_results.boot_n,
            sim_n=ci_results.sim_n,
            clusters_confidence_interval_plot=self.cluster_visualizer.plot_confidence_intervals(
                    ci_results, config
                ),
        )

        # Return clustered results
        return ClusteredResult(
            cluster_data=result.cluster_data.assign(
                top_sol=result.cluster_data["sol_id"].isin(top_solutions["sol_id"]),
                cluster=result.cluster_data["cluster"].astype(int),
            ),
            top_solutions=top_solutions,
            wss=result.wss,
            cluster_ci=cluster_ci,
            n_clusters=result.n_clusters,
            errors_weights=config.weights,
            clusters_means=result.clusters_means,
            clusters_pca=None,  # result.cluster_pca (implementation should be in clusterKmeans)
            clusters_tsne=None,  # result.cluster_tsne (implementation should be in clusterKmeans)
            correlations=None,  # result.correlations (implementation should be in clusterKmeans)
            plots= plots
        )

    def _calculate_confidence_intervals(
        self,
        x_decomp_agg: pd.DataFrame,
        clustered_data: pd.DataFrame,
        config: ClusteringConfig,
        all_paid: List[str],
        boot_n: int = 1000,
        sim_n: int = 10000,
    ) -> ConfidenceIntervalCollectionData:
        """
        Calculate confidence intervals for a given dataset using bootstrap resampling.
        Args:
            x_decomp_agg (DataFrame): The input dataset.
            clustered_data: A dataframe containing the clustered data.
            all_paid (list): A list of paid channels.
            boot_n (int, optional): The number of bootstrap iterations. Defaults to 1000.
            sim_n (int, optional): The number of simulations. Defaults to 10000.
        Returns:
            dict: A dictionary containing the confidence intervals, simulation results, and other metadata.
        """
        df_clusters_outcome = x_decomp_agg.dropna(subset=['total_spend']) \
            .merge(clustered_data[['sol_id', 'cluster']], on='sol_id', how='left') \
            [['sol_id', 'cluster', 'rn', 'roi_total', 'cpa_total', 'robynPareto']] \
            .groupby(['cluster', 'rn'], observed=False).apply(lambda x: x.assign(n=len(x))) \
            .reset_index(drop=True)
        df_clusters_outcome = df_clusters_outcome\
            .dropna(subset=['cluster']) \
            .sort_values(by=['cluster', 'rn'])

        # Initialize lists to store results
        cluster_collect = []
        chn_collect = []
        sim_collect = []
        # Loop through each cluster
        for j in range(1, config.k_clusters + 1):
            df_outcome = df_clusters_outcome[df_clusters_outcome["cluster"] == j]
            if len(df_outcome["sol_id"].unique()) < 3:
                self.logger.warning(
                    f"Warning: Cluster {j} does not contain enough models to calculate CI"
                )
            else:
                if config.cluster_by == ClusterBy.HYPERPARAMETERS:
                    pattern = "|".join(["_" + name for name in HYPERPARAMETER_NAMES])
                    replacement = ""
                    all_paid = list(set([re.sub(pattern, replacement, str(x)) for x in all_paid]))
                # Loop through each channel
                for i in all_paid:
                    # Bootstrap CI
                    if config.dep_var_type == DependentVarType.CONVERSION:
                        # Drop CPA == Inf
                        df_chn = df_outcome[
                            (df_outcome["rn"] == i)
                            & (df_outcome["cpa_total"] != np.inf)
                        ]
                        v_samp = df_chn["cpa_total"]
                    else:
                        df_chn = df_outcome[df_outcome["rn"] == i]
                        v_samp = df_chn["roi_total"]
                    boot_res = self._bootstrap_sampling(v_samp, boot_n)
                    boot_mean = np.mean(boot_res["boot_means"] if len(boot_res["boot_means"]) > 0 else [0.0])
                    boot_se = boot_res["se"][0]
                    ci_low = max(0, boot_res["ci"][0])
                    ci_up = boot_res["ci"][1]
                    # Collect loop results
                    chn_collect.append(
                        df_chn.assign(
                            ci_low=ci_low,
                            ci_up=ci_up,
                            n=len(v_samp),
                            boot_se=boot_se,
                            boot_mean=boot_mean,
                            cluster=j,
                        ).reset_index(drop=True)
                    )
                    sim_collect.append(
                        pd.DataFrame(
                            {
                                "cluster": j,
                                "rn": i,
                                "n": len(v_samp),
                                "boot_mean": boot_mean,
                                "x_sim": np.random.normal(
                                    loc=boot_mean, scale=boot_se, size=sim_n
                                ),
                            }
                        ).assign(
                            y_sim=lambda x: stats.norm.pdf(
                                x["x_sim"], loc=boot_mean, scale=boot_se
                            )
                        ).reset_index(drop=True)
                    )
                cluster_collect.append(
                    {"chn_collect": chn_collect, "sim_collect": sim_collect}
                )
        # Bind rows and filter
        sim_collect = pd.concat([pd.concat(x['sim_collect']) for x in cluster_collect])
        sim_collect = sim_collect[sim_collect['n'] > 0]
        sim_collect['cluster_title'] = sim_collect.apply(lambda row: f"Cl.{row['cluster']} (n={row['n']})", axis=1)
        sim_collect = sim_collect.reset_index(drop=True)

        # Calculate CI
        df_ci = pd.concat([pd.concat(x['chn_collect']) for x in cluster_collect])
        df_ci['cluster_title'] = df_ci.apply(lambda row: f"Cl.{row['cluster']} (n={row['n']})", axis=1)
        df_ci = df_ci[['rn', 'cluster_title', 'n', 'cluster', 'boot_mean', 'boot_se', 'ci_low', 'ci_up']]
        df_ci = df_ci.drop_duplicates()
        df_ci = df_ci.groupby(['rn', 'cluster_title', 'cluster']).apply(lambda x: pd.Series({
            'n': x['n'].iloc[0],
            'boot_mean': x['boot_mean'].iloc[0],
            'boot_se': x['boot_se'].iloc[0],
            'boot_ci': f"[{round(x['ci_low'].iloc[0], 2)}, {round(x['ci_up'].iloc[0], 2)}]",
            'ci_low': x['ci_low'].iloc[0],
            'ci_up': x['ci_up'].iloc[0],
            'sd': x['boot_se'].iloc[0] * ((x['n'].iloc[0] - 1) ** 0.5),
            'dist100': (x['ci_up'].iloc[0] - x['ci_low'].iloc[0] + 2 * x['boot_se'].iloc[0] * ((x['n'].iloc[0] - 1) ** 0.5)) / 99
        })).reset_index()
        return ConfidenceIntervalCollectionData(
            confidence_interval_df=df_ci,
            sim_collect=sim_collect,
            boot_n=boot_n,
            sim_n=sim_n,
        )

    def clusterKmeans(
        self,
        df: pd.DataFrame,
        config: ClusteringConfig,
        k: Optional[int] = None,
        limit=30,
        ignored_columns: Optional[List[str]] = None,
        wss_var: float = 0.0,
    ) -> ClusteredResult:
        """
        Automated K-Means Clustering + PCA/t-SNE.
        Parameters:
        df (pandas.DataFrame): Input dataframe.
        k (int): Number of clusters. If None, it will be determined automatically.
        limit (int): Maximum number of clusters to consider.
        ignore (list): List of column names to ignore.
        Returns:
        ClusteredResult: containing the results of the clustering.
        """
        np.random.seed(config.seed)

        # Check if ignored_columns is not None and its first element is NA
        if ignored_columns is not None and pd.isna(ignored_columns[0]):
            ignored_columns = None

        # Ensure ignored_columns is a list of unique values
        if ignored_columns is not None:
            ignored_columns = list(set(ignored_columns))
            order = df.columns.tolist()
            aux = df[[col for col in df.columns if col in ignored_columns]]
            df = df[[col for col in df.columns if col not in ignored_columns]]
            self.logger.info(f"Ignored features: {', '.join(ignored_columns)}")

        wss = df.var(axis=0).sum() * (len(df) - 1)
        limit = min(len(df) - 1, limit)
        wss_values = [wss]
        for i in range(2, limit + 1):
            kmeans = KMeans(n_clusters=i).fit(df)
            wss_values.append(kmeans.inertia_)
        nclusters = pd.DataFrame({"n": range(1, limit + 1), "wss": wss_values})

        result = ClusteredResult()
        result.cluster_data = nclusters
        result.wss = self.cluster_visualizer.create_wss_plot(nclusters, k=None)

        if wss_var > 0.0 and k is None:
            nclusters["pareto"] = nclusters["wss"] / nclusters["wss"].iloc[0]
            nclusters["dif"] = nclusters["pareto"].shift(1) - nclusters["pareto"]
            k_candidates = nclusters[nclusters["dif"] > wss_var]["n"]
            k = k_candidates.max() if not k_candidates.empty else None
            self.logger.info(
                f">> Auto selected k = {k} (clusters) based on minimum WSS variance of {wss_var * 100}%"
            )

        if k is not None:
            if ignored_columns is not None:
                result.cluster_data = pd.concat([df, aux], axis=1).reindex(
                    columns=order + [c for c in df.columns if c not in order]
                )
            else:
                result.cluster_data = df

            result.wss = self.cluster_visualizer.create_wss_plot(nclusters, k=k)
            result.n_clusters = k
            df_copy = df.copy()
            kmeans = KMeans(n_clusters=k, algorithm="lloyd", max_iter=limit)
            df_copy.loc[:, "cluster"] = pd.Categorical(kmeans.fit_predict(df_copy))
            result.cluster_data = pd.concat(
                [result.cluster_data, df_copy["cluster"]], axis=1
            )
            df = df_copy

            # Group by 'cluster' and calculate the mean for each group
            cluster_means: pd.DataFrame = df.groupby("cluster", observed=False).mean()
            # Calculate the count of each cluster
            cluster_counts = df["cluster"].value_counts()
            # Add the counts as a new column to the cluster_means DataFrame
            cluster_means["n"] = cluster_counts

            result.clusters_means = cluster_means.reset_index()
            result.correlations = None
            result.clusters_pca = None
            result.clusters_tsne = None

            #  TODO: Implement this logic if it is needed
            # Assuming corr_cross is defined elsewhere
            # result.correlations = corr_cross(
            #     df, contains="cluster_", quiet=True, ignore=ignore
            # )
            # valid_reductions = ["PCA", "tSNE", "all", "none"]
            # if config.dim_reduction not in valid_reductions:
            #     raise ValueError(
            #         f"Invalid option: {config.dim_reduction}. Valid options are {valid_reductions}."
            #     )
            # if "all" in config.dim_reduction:
            #     dim_reduction = ["PCA", "tSNE"]
            # else:
            #     dim_reduction = config.dim_reduction
            # Assuming reduce_pca is a function defined elsewhere
            # if "PCA" in dim_reduction:
            # plot PCA dimentionality reduction
            # result.clusters_pca = self.cluster_visualizer.plot_dimensionality_reduction(pca_df)
            # Assuming reduce_tsne is a function defined elsewhere
            # if "tSNE" in dim_reduction:
            # result.clusters_tsne = self.cluster_visualizer.plot_dimensionality_reduction(tsne_df)

        return result

    def _select_optimal_clusters(
        self,
        df: pd.DataFrame,
        config: ClusteringConfig,
        limit_clusters: int,
        ignored_columns: Optional[List[str]] = None,
        wss_var: float = 0.06,
    ) -> int:
        """
        Selects the optimal number of clusters based on WSS variance.

        Args:
            df (pd.DataFrame): The prepared data for clustering.
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            int: The optimal number of clusters.
        """
        if config.k_clusters == maxsize:
            try:
                cls = self.clusterKmeans(
                    df,
                    config,
                    k=None,
                    limit=limit_clusters,
                    ignored_columns=ignored_columns,
                )
                if cls.cluster_data is None or cls.cluster_data.empty:
                    raise ValueError("cls.cluster_data is None or empty")
                nclusters = cls.cluster_data
                # Step 1: Add 'pareto' and 'dif' columns
                nclusters["pareto"] = nclusters["wss"] / nclusters["wss"].iloc[0]
                nclusters["dif"] = nclusters["pareto"].shift(1) - nclusters["pareto"]
                # Step 2: Filter rows where 'dif' is greater than 'wss_var'
                filtered_nclusters = nclusters[nclusters["dif"] > wss_var]
                # Step 3: Extract the 'n' column
                n_values = filtered_nclusters["n"]
                # Step 4: Find the maximum value, ignoring NaNs
                k = n_values.max(skipna=True)

                if k < config.min_clusters:
                    self.logger.info(
                        f"Too few clusters: {k}. Setting to {config.min_clusters}"
                    )
                    k = config.min_clusters
                if k > limit_clusters:
                    self.logger.info(
                        f"Too many clusters: {k}. Lowering to {limit_clusters} (max_clusters)"
                    )
                    k = limit_clusters
                self.logger.info(
                    f">> Auto selected k = {k} (clusters) based on minimum WSS variance"
                )
                return k
            except Exception as e:
                self.logger.error(f"Couldn't automatically create clusters: {e}")
                raise e
        else:
            return config.k_clusters

    def _prepare_performance_data_for_clustering(
        self, x_decomp_agg: pd.DataFrame, config: ClusteringConfig
    ) -> pd.DataFrame:
        """
        Prepares the data frame for clustering based on specified criteria.

        Args:
            config (ClusteringConfig): Configuration for the clustering process.

        Returns:
            pd.DataFrame: A DataFrame prepared for clustering.
        """
        outcome: pd.DataFrame = pd.DataFrame()
        if config.dep_var_type == DependentVarType.REVENUE:
            outcome = x_decomp_agg[["sol_id", "rn", "roi_total"]].pivot(
                index="sol_id", columns="rn", values="roi_total"
            )
            outcome = outcome.dropna(axis=1, how="all")
            outcome = outcome.reset_index()[
                ["sol_id"] + [col for col in outcome.columns if col in config.all_media]
            ]
        elif config.dep_var_type == DependentVarType.CONVERSION:
            outcome = x_decomp_agg[["sol_id", "rn", "cpa_total"]].dropna(
                subset=["cpa_total"]
            )
            outcome = outcome.pivot(index="sol_id", columns="rn", values="cpa_total")
            outcome = outcome.dropna(axis=1, how="all")
            outcome = outcome.reset_index()[
                ["sol_id"] + [col for col in outcome.columns if col in config.all_media]
            ]

        # Select distinct errors
        errors = x_decomp_agg[
            ["sol_id"]
            + [col for col in x_decomp_agg.columns if col.startswith("nrmse")]
            + ["decomp.rssd", "mape"]
        ].drop_duplicates()

        # Merge outcome with errors
        return outcome.merge(errors, on="sol_id", how="left")

    def _prepare_hyperparameter_data_for_clustering(
        self, result_hyp_param: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepares the data frame for clustering based on specified criteria.

        Args:
            result_hyp_param (pd.DataFrame): The data frame containing the hyperparameter results.
        Returns:
            pd.DataFrame: A DataFrame prepared for clustering.
        """
        # Select columns using list comprehension and set operations for efficiency
        selected_columns = (
            ["sol_id"]
            + [
                col
                for col in result_hyp_param.columns
                if any(hyp in col for hyp in HYPERPARAMETER_NAMES)
            ]
            + [
                col
                for col in result_hyp_param.columns
                if any(metric in col for metric in ["nrmse", "decomp.rssd", "mape"])
            ]
        )

        # Create a DataFrame with the selected columns
        outcome: pd.DataFrame = result_hyp_param[selected_columns]

        # Remove columns with all NA values
        return outcome.dropna(axis=1, how="all")

    def _compute_error_scores(
        self,
        df: pd.DataFrame,
        weights: List[float] = [1, 1, 1],
        ts_validation: bool = True,
    ) -> pd.Series:
        """
        Calculate error scores for a given data frame.
        Args:
            df (pandas DataFrame): Input data frame.
            balance (list, optional): Balance weights for error scores. Defaults to [1, 1, 1].
            ts_validation (bool, optional): Whether to use time series validation. Defaults to True.
        Returns:
            pandas Series: Error scores for the input data frame.
        """
        assert len(weights) == 3
        error_cols = [
            "nrmse_test" if ts_validation else "nrmse_train",
            "decomp.rssd",
            "mape",
        ]
        assert all(col in df.columns for col in error_cols)

        balance = np.array(weights) / sum(weights)

        scores = df[error_cols].copy()
        scores = scores.rename(columns={error_cols[0]: "nrmse"})

        scores["nrmse"] = np.where(
            np.isinf(scores["nrmse"]),
            np.max(scores["nrmse"][~np.isinf(scores["nrmse"])]),
            scores["nrmse"],
        )
        scores["decomp.rssd"] = np.where(
            np.isinf(scores["decomp.rssd"]),
            np.max(scores["decomp.rssd"][~np.isinf(scores["decomp.rssd"])]),
            scores["decomp.rssd"],
        )
        scores["mape"] = np.where(
            np.isinf(scores["mape"]),
            np.max(scores["mape"][~np.isinf(scores["mape"])]),
            scores["mape"],
        )

        scores["nrmse_n"] = ParetoUtils.min_max_norm(scores["nrmse"])
        scores["decomp.rssd_n"] = ParetoUtils.min_max_norm(scores["decomp.rssd"])
        scores["mape_n"] = ParetoUtils.min_max_norm(scores["mape"])

        scores.fillna(0, inplace=True)

        scores["nrmse_w"] = balance[0] * scores["nrmse_n"]
        scores["decomp.rssd_w"] = balance[1] * scores["decomp.rssd_n"]
        scores["mape_w"] = balance[2] * scores["mape_n"]

        scores["error_score"] = np.sqrt(
            scores["nrmse_w"] ** 2
            + scores["decomp.rssd_w"] ** 2
            + scores["mape_w"] ** 2
        )

        return scores["error_score"]
    def _select_top_solutions(
        self, clustered_df: pd.DataFrame, config: ClusteringConfig, limit: int = 1
    ) -> pd.DataFrame:
        """
        Selects the top models based on their distance to the origin.
        Generate a data frame with error scores and ranks for each cluster.
        Args:
            clustered_df (pd.DataFrame): Input data frame.
            config (ClusteringConfig): Used to get Balance weights for error scores. Defaults to [1, 1, 1].
            limit (int, optional): Number of rows to return per cluster. Defaults to 1.
            ts_validation (bool, optional): Whether to use time series validation. Defaults to True.
        Returns:
            pandas DataFrame: Data frame with error scores and ranks for each cluster.
        """
        ts_validation: bool = "nrmse_test" in clustered_df.columns
        df = clustered_df.copy()
        df["error_score"] = self._compute_error_scores(
            df, config.weights, ts_validation
        )
        df.fillna(0, inplace=True)
        df = df.sort_values(by=["cluster", "error_score"])
        df = df.groupby("cluster", observed=False).head(limit)
        df["rank"] = df.groupby("cluster", observed=False)["error_score"].rank(method="dense")

        return df[
            ["cluster", "rank"]
            + [col for col in df.columns if col not in ["cluster", "rank"]]
        ]

    def _bootstrap_sampling(
        self, sample: pd.Series, boot_n: int
    ) -> Dict[str, List[float]]:
        """
        Calculate the bootstrap confidence interval for a given sample.
        Args:
            sample (array-like): The input sample.
            boot_n (int, optional): The number of bootstrap iterations. Defaults to 1000.
        Returns:
            dict: A dictionary containing the bootstrap means, confidence interval, and standard error.
        """
        if len(sample[~sample.isna()]) > 1:
            sample_n = len(sample)
            sample_mean = np.mean(sample)
            boot_sample = np.random.choice(
                sample, size=(boot_n, sample_n), replace=True
            )
            boot_means = np.mean(boot_sample, axis=1)
            se = np.std(boot_means)
            me = stats.t.ppf(0.975, sample_n - 1) * se
            sample_me = np.sqrt(sample_n) * me
            ci = [sample_mean - sample_me, sample_mean + sample_me]
            return {
                "boot_means": boot_means,
                "ci": ci,
                "se": [float(se)],
            }
        else:
            ci = [np.nan, np.nan] if sample.isna().all() else [sample.iloc[0], sample.iloc[0]]
            return {
                "boot_means": sample.tolist(),
                "ci": ci,
                "se": [0.0],
            }
