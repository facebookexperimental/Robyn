# pyre-strict

import logging
from sys import maxsize
from typing import Dict, List, Optional, Union

import re
import numpy as np
import pandas as pd
from robyn.common.logger import RobynLogger
from robyn.modeling.clustering.clustering_config import ClusterBy, ClusteringConfig
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
        self.logger.info("Initializing ClusterBuilder")
        self.logger.debug(
            "Received ParetoResult with %d solutions",
            len(pareto_result.pareto_solutions),
        )

        self.pareto_result: ParetoResult = pareto_result
        self.cluster_visualizer = ClusterVisualizer(None, None, None)
        self.logger.info("ClusterBuilder initialization complete")

    def cluster_models(self, config: ClusteringConfig) -> ClusteredResult:
        """
        Clusters models based on specified criteria.
        """
        self.logger.info("Starting model clustering process")
        self.logger.debug("Clustering configuration: %s", config)

        allSolutions = self.pareto_result.pareto_solutions
        self.logger.debug("Processing %d Pareto solutions", len(allSolutions))

        x_decomp_agg = self.pareto_result.x_decomp_agg[
            self.pareto_result.x_decomp_agg["sol_id"].isin(allSolutions)
        ]
        result_hyp_param = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["sol_id"].isin(allSolutions)
        ]

        if config.all_media is None:
            aux = self.pareto_result.media_vec_collect.columns
            type_index = aux.get_loc("type")
            config.all_media = aux[1:type_index].tolist()
            self.logger.info("Auto-detected media channels: %s", config.all_media)

        self.logger.info("Clustering by: %s", config.cluster_by)
        if config.cluster_by == ClusterBy.HYPERPARAMETERS:
            self.logger.debug("Preparing hyperparameter data for clustering")
            df = self._prepare_hyperparameter_data_for_clustering(result_hyp_param)
        elif config.cluster_by == ClusterBy.PERFORMANCE:
            self.logger.debug("Preparing performance data for clustering")
            df = self._prepare_performance_data_for_clustering(x_decomp_agg, config)
        else:
            self.logger.error(f"Invalid clustering method: {config.cluster_by}")
            raise ValueError("Invalid clustering method")

        RobynLogger.log_df(self.logger, df)
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
        limit_clusters = min(len(df) - 1, 30)

        self.logger.debug(
            f"Determining optimal number of clusters (limit: {limit_clusters})"
        )
        config.k_clusters = self._select_optimal_clusters(
            df, config, limit_clusters, ignored_columns
        )
        self.logger.info(f"Selected {config.k_clusters} clusters")

        if (
            config.k_clusters < config.min_clusters
            or config.k_clusters > limit_clusters
        ):
            self.logger.error(
                f"Invalid number of clusters: {config.k_clusters} "
                f"(min: {config.min_clusters}, max: {limit_clusters})"
            )
            raise ValueError(
                f"Number of clusters {config.k_clusters} is outside the specified range "
                f"{config.min_clusters} to {config.max_clusters}."
            )

        self.logger.debug("Performing KMeans clustering")
        result = self.clusterKmeans(
            df=df,
            config=config,
            k=config.k_clusters,
            limit=limit_clusters,
            ignored_columns=ignored_columns,
        )

        self.logger.debug("Processing cluster results")
        grouped = result.cluster_data.groupby("cluster", observed=False)
        result.cluster_data["n"] = grouped["cluster"].transform("size")

        all_paid = [
            col
            for col in result.cluster_data.columns
            if col not in ignored_columns + ["cluster"]
        ]

        self.logger.debug("Selecting top solutions")
        top_solutions = self._select_top_solutions(result.cluster_data, config)
        self.logger.info(f"Selected {len(top_solutions)} top solutions")
        RobynLogger.log_df(self.logger, top_solutions, print_head=True)

        self.logger.debug("Calculating confidence intervals")
        ci_results: ConfidenceIntervalCollectionData = (
            self._calculate_confidence_intervals(
                x_decomp_agg, result.cluster_data, config, all_paid
            )
        )

        self.logger.debug("Preparing plot results")
        plots = ClusterPlotResults(
            top_solutions_errors_plot=self.cluster_visualizer.plot_top_solutions_errors(
                df, top_solutions, balance=config.weights
            ),
            top_solutions_rois_plot=self.cluster_visualizer.plot_topsols_rois(
                df, top_solutions, config.all_media
            ),
        )

        self.logger.debug("Creating confidence intervals object")
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
        clustered_data = result.cluster_data.assign(
            top_sol=result.cluster_data["sol_id"].isin(top_solutions["sol_id"]),
            cluster=result.cluster_data["cluster"].astype(int),
        )

        RobynLogger.log_df(self.logger, clustered_data)
        self.logger.info("Clustering process completed successfully")
        return ClusteredResult(
            cluster_data=clustered_data,
            top_solutions=top_solutions,
            wss=result.wss,
            cluster_ci=cluster_ci,
            n_clusters=result.n_clusters,
            errors_weights=config.weights,
            clusters_means=result.clusters_means,
            clusters_pca=None,
            clusters_tsne=None,
            correlations=None,
            plots=plots,
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
        """Calculate confidence intervals for clustered data"""
        self.logger.info("Starting confidence interval calculations")
        self.logger.debug(f"Parameters: boot_n={boot_n}, sim_n={sim_n}")

        self.logger.debug("Preparing cluster outcomes data")
        df_clusters_outcome = x_decomp_agg.dropna(subset=["total_spend"]).merge(
            clustered_data[["sol_id", "cluster"]], on="sol_id", how="left"
        )[["sol_id", "cluster", "rn", "roi_total", "cpa_total", "robynPareto"]]

        self.logger.debug("Grouping cluster outcomes")
        df_clusters_outcome = (
            df_clusters_outcome.groupby(["cluster", "rn"], observed=False)
            .apply(lambda x: x.assign(n=len(x)))
            .reset_index(drop=True)
            .dropna(subset=["cluster"])
            .sort_values(by=["cluster", "rn"])
        )

        cluster_collect = []

        self.logger.debug(f"Processing {config.k_clusters} clusters")
        for j in range(0, config.k_clusters):
            df_outcome = df_clusters_outcome[df_clusters_outcome["cluster"] == j]
            if len(df_outcome["sol_id"].unique()) < 3:
                self.logger.warning(
                    f"Cluster {j} has insufficient models ({len(df_outcome['sol_id'].unique())}) for CI calculation"
                )
                continue

            chn_collect = {}
            sim_collect = {}
            self.logger.debug(f"Processing cluster {j}")
            if config.cluster_by == ClusterBy.HYPERPARAMETERS:
                pattern = "|".join(["_" + name for name in HYPERPARAMETER_NAMES])
                replacement = ""
                all_paid = list(
                    set([re.sub(pattern, replacement, str(x)) for x in all_paid])
                )

            for i in all_paid:
                self.logger.debug(f"Processing channel {i} for cluster {j}")
                if config.dep_var_type == DependentVarType.CONVERSION:
                    df_chn = df_outcome[
                        (df_outcome["rn"] == i) & (df_outcome["cpa_total"] != np.inf)
                    ]
                    v_samp = df_chn["cpa_total"]
                else:
                    df_chn = df_outcome[df_outcome["rn"] == i]
                    v_samp = df_chn["roi_total"]

                boot_res = self._bootstrap_sampling(v_samp, boot_n)
                boot_mean = float(np.nanmean(boot_res["boot_means"]))
                boot_se = boot_res["se"]
                ci_low = max(0, boot_res["ci"][0])
                ci_up = boot_res["ci"][1]

                self.logger.debug(
                    f"Cluster {j}, Channel {i}: CI [{ci_low:.2f}, {ci_up:.2f}], "
                    f"SE={boot_se:.4f}, n={len(v_samp)}"
                )

                chn_collect[i] = df_chn.assign(
                    ci_low=ci_low,
                    ci_up=ci_up,
                    n=len(v_samp),
                    boot_se=boot_se,
                    boot_mean=boot_mean,
                    cluster=j,
                ).reset_index(drop=True)

                sim_collect[i] = (
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
                    )
                    .assign(
                        y_sim=lambda x: stats.norm.pdf(
                            x["x_sim"], loc=boot_mean, scale=boot_se
                        )
                    )
                    .reset_index(drop=True)
                )

            cluster_collect.append(
                {"chn_collect": chn_collect, "sim_collect": sim_collect}
            )

        self.logger.debug("Finalizing simulation results")
        sim_collect = pd.concat([pd.concat(x["sim_collect"]) for x in cluster_collect])
        sim_collect = sim_collect[sim_collect["n"] > 0]
        sim_collect["cluster_title"] = sim_collect.apply(
            lambda row: f"Cl.{row['cluster']} (n={row['n']})", axis=1
        )
        sim_collect = sim_collect.reset_index(drop=True)

        self.logger.debug("Adding cluster_title to confidence intervals")
        df_ci = pd.concat(
            [
                pd.concat(x["chn_collect"].values())
                for x in cluster_collect
                if x["chn_collect"]
            ]
        )
        df_ci = df_ci.assign(cluster_title=lambda x: f"Cl.{x['cluster']} (n={x['n']})")[
            [
                "rn",
                "cluster_title",
                "n",
                "cluster",
                "boot_mean",
                "boot_se",
                "ci_low",
                "ci_up",
            ]
        ].drop_duplicates()

        self.logger.debug("Computing final confidence intervals")
        df_ci = (
            df_ci.groupby(["rn", "cluster_title", "cluster"])
            .apply(
                lambda x: pd.Series(
                    {
                        "n": x["n"].iloc[0],
                        "boot_mean": x["boot_mean"].iloc[0],
                        "boot_se": x["boot_se"].iloc[0],
                        "boot_ci": f"[{x['ci_low'].iloc[0]:.2f}, {x['ci_up'].iloc[0]:.2f}]",
                        "ci_low": x["ci_low"].iloc[0],
                        "ci_up": x["ci_up"].iloc[0],
                        "sd": x["boot_se"].iloc[0] * np.sqrt(x["n"].iloc[0] - 1),
                        "dist100": (
                            x["ci_up"].iloc[0]
                            - x["ci_low"].iloc[0]
                            + 2 * x["boot_se"].iloc[0] * np.sqrt(x["n"].iloc[0] - 1)
                        )
                        / 99,
                    }
                )
            )
            .reset_index()
        )

        self.logger.info("Confidence interval calculations completed")
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
        limit=10,
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
        self.logger.debug(f"Starting clusterKmeans with input shape: {df.shape}")
        self.logger.debug(
            f"Initial parameters - k: {k}, limit: {limit}, wss_var: {wss_var}"
        )

        np.random.seed(config.seed)
        self.logger.debug(f"Random seed set to: {config.seed}")

        # Check if ignored_columns is not None and its first element is NA
        if ignored_columns is not None and pd.isna(ignored_columns[0]):
            self.logger.debug("First element of ignored_columns is NA, setting to None")
            ignored_columns = None

        # Ensure ignored_columns is a list of unique values
        if ignored_columns is not None:
            self.logger.debug(
                f"Processing ignored columns, initial count: {len(ignored_columns)}"
            )
            ignored_columns = list(set(ignored_columns))
            order = df.columns.tolist()
            aux = df[[col for col in df.columns if col in ignored_columns]]
            df = df[[col for col in df.columns if col not in ignored_columns]]
            self.logger.info(f"Ignored features: {', '.join(ignored_columns)}")
            self.logger.debug(
                f"DataFrame shape after removing ignored columns: {df.shape}"
            )

        self.logger.debug("Calculating initial WSS")
        wss_values = [np.sum(np.var(df, axis=0)) * (len(df) - 1)]
        limit = min(len(df) - 1, limit)
        self.logger.info(f"Starting WSS calculation for {limit} clusters")
        for i in range(2, limit + 1):
            self.logger.debug(f"Calculating WSS for {i} clusters")
            kmeans = KMeans(n_clusters=i).fit(df)
            wss_values.append(kmeans.inertia_)
        self.logger.debug("Creating nclusters DataFrame")
        nclusters = pd.DataFrame({"n": range(1, limit + 1), "wss": wss_values})

        result = ClusteredResult()
        result.cluster_data = nclusters
        result.wss = self.cluster_visualizer.create_wss_plot(nclusters, k=None)
        self.logger.debug("Initial WSS plot created")

        if wss_var > 0.0 and k is None:
            self.logger.info(
                f"Attempting automatic cluster selection with WSS variance: {wss_var}"
            )
            nclusters["pareto"] = nclusters["wss"] / nclusters["wss"].iloc[0]
            nclusters["dif"] = nclusters["pareto"].shift(1) - nclusters["pareto"]
            k_candidates = nclusters[nclusters["dif"] > wss_var]["n"]
            k = k_candidates.max() if not k_candidates.empty else None
            self.logger.info(
                f"Auto selected k = {k} (clusters) based on minimum WSS variance of {wss_var * 100}%"
            )

        if k is not None:
            self.logger.info(f"Proceeding with k={k} clusters")
            if ignored_columns is not None:
                self.logger.debug("Reconstructing DataFrame with ignored columns")
                result.cluster_data = pd.concat([df, aux], axis=1).reindex(
                    columns=order + [c for c in df.columns if c not in order]
                )
            else:
                result.cluster_data = df

            result.wss = self.cluster_visualizer.create_wss_plot(nclusters, k=k)
            result.n_clusters = k

            self.logger.debug("Performing KMeans clustering")
            df_copy = df.copy()
            try:
                kmeans = KMeans(n_clusters=k, algorithm="lloyd", max_iter=limit)
                df_copy.loc[:, "cluster"] = pd.Categorical(kmeans.fit_predict(df_copy))
                self.logger.debug("KMeans clustering completed successfully")
            except Exception as e:
                self.logger.error(f"KMeans clustering failed: {str(e)}")
                raise

            result.cluster_data = pd.concat(
                [result.cluster_data, df_copy["cluster"]], axis=1
            )
            self.logger.debug(f"Clustered data after KMeans:")
            RobynLogger.log_df(self.logger, result.cluster_data)

            df = df_copy

            self.logger.debug("Calculating cluster means and counts")
            try:
                # Group by 'cluster' and calculate the mean for each group
                cluster_means: pd.DataFrame = df.groupby(
                    "cluster", observed=False
                ).mean()
                # Calculate the count of each cluster
                cluster_counts = df["cluster"].value_counts()
                # Add the counts as a new column to the cluster_means DataFrame
                cluster_means["n"] = cluster_counts

                self.logger.debug(f"Cluster sizes: {dict(cluster_counts)}")
                self.logger.debug(f"Cluster means:")
                RobynLogger.log_df(self.logger, cluster_means)

                # Check for potentially problematic clusters
                min_cluster_size = cluster_counts.min()
                if min_cluster_size < 3:
                    self.logger.warning(
                        f"Some clusters have very few members. Minimum cluster size: {min_cluster_size}"
                    )
            except Exception as e:
                self.logger.error(f"Error calculating cluster statistics: {str(e)}")
                raise

            result.clusters_means = cluster_means.reset_index()
            result.correlations = None
            result.clusters_pca = None
            result.clusters_tsne = None

            self.logger.info("Clustering process completed successfully")
            self.logger.debug(f"Final result shape: {result.cluster_data.shape}")

        return result

    def _select_optimal_clusters(
        self,
        df: pd.DataFrame,
        config: ClusteringConfig,
        limit_clusters: int,
        ignored_columns: Optional[List[str]] = None,
        wss_var: float = 0.06,
    ) -> int:
        """Selects the optimal number of clusters based on WSS variance."""
        self.logger.info("Starting optimal cluster selection")
        self.logger.debug(
            f"Parameters: limit_clusters={limit_clusters}, wss_var={wss_var}"
        )

        if config.k_clusters == maxsize:
            try:
                self.logger.debug("Attempting automatic cluster selection")
                cls = self.clusterKmeans(
                    df,
                    config,
                    k=None,
                    limit=limit_clusters,
                    ignored_columns=ignored_columns,
                )

                if cls.cluster_data is None or cls.cluster_data.empty:
                    self.logger.error("Cluster data is None or empty")
                    raise ValueError("cls.cluster_data is None or empty")

                k = (
                    cls.cluster_data.assign(
                        pareto=lambda x: x["wss"] / x["wss"].iloc[0]
                    )
                    .assign(
                        dif=lambda x: x["pareto"].shift(1, fill_value=np.nan)
                        - x["pareto"]
                    )
                    .query(f"dif > {wss_var}")["n"]
                    .values
                )
                k = np.max(k)

                self.logger.debug(f"Initial optimal k value: {k}")

                if k < config.min_clusters:
                    self.logger.warning(
                        f"Adjusting cluster count up from {k} to minimum {config.min_clusters}"
                    )
                    k = config.min_clusters
                if k > limit_clusters:
                    self.logger.warning(
                        f"Adjusting cluster count down from {k} to maximum {limit_clusters}"
                    )
                    k = limit_clusters

                self.logger.info(f"Selected optimal number of clusters: {k}")
                return k
            except Exception as e:
                self.logger.error(
                    f"Failed to automatically determine clusters: {str(e)}"
                )
                raise e
        else:
            self.logger.info(f"Using pre-configured cluster count: {config.k_clusters}")
            return config.k_clusters

    def _prepare_performance_data_for_clustering(
        self, x_decomp_agg: pd.DataFrame, config: ClusteringConfig
    ) -> pd.DataFrame:
        """Prepares performance data for clustering"""
        self.logger.debug("Preparing performance data for clustering")
        self.logger.debug(f"Dependent variable type: {config.dep_var_type}")

        outcome: pd.DataFrame = pd.DataFrame()
        if config.dep_var_type == DependentVarType.REVENUE:
            self.logger.debug("Processing revenue-based metrics")
            outcome = x_decomp_agg[["sol_id", "rn", "roi_total"]].pivot(
                index="sol_id", columns="rn", values="roi_total"
            )
            outcome = outcome.dropna(axis=1, how="all")
            outcome = outcome.reset_index()[
                ["sol_id"] + [col for col in outcome.columns if col in config.all_media]
            ]
        elif config.dep_var_type == DependentVarType.CONVERSION:
            self.logger.debug("Processing conversion-based metrics")
            outcome = x_decomp_agg[["sol_id", "rn", "cpa_total"]].dropna(
                subset=["cpa_total"]
            )
            outcome = outcome.pivot(index="sol_id", columns="rn", values="cpa_total")
            outcome = outcome.dropna(axis=1, how="all")
            outcome = outcome.reset_index()[
                ["sol_id"] + [col for col in outcome.columns if col in config.all_media]
            ]

        self.logger.debug("Selecting error metrics")
        errors = x_decomp_agg[
            ["sol_id"]
            + [col for col in x_decomp_agg.columns if col.startswith("nrmse")]
            + ["decomp.rssd", "mape"]
        ].drop_duplicates()

        self.logger.debug("Merging outcome with errors")
        final_df = outcome.merge(errors, on="sol_id", how="left")
        self.logger.debug(f"Final dataset shape: {final_df.shape}")
        return final_df

    def _prepare_hyperparameter_data_for_clustering(
        self, result_hyp_param: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepares hyperparameter data for clustering"""
        self.logger.debug("Preparing hyperparameter data for clustering")

        # Select relevant columns
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

        self.logger.debug(f"Selected {len(selected_columns)} columns for clustering")

        # Create DataFrame with selected columns
        outcome: pd.DataFrame = result_hyp_param[selected_columns]

        # Remove columns with all NA values
        initial_cols = len(outcome.columns)
        outcome = outcome.dropna(axis=1, how="all")
        dropped_cols = initial_cols - len(outcome.columns)

        if dropped_cols > 0:
            self.logger.warning(f"Dropped {dropped_cols} columns with all NA values")

        self.logger.debug(f"Final dataset shape: {outcome.shape}")
        return outcome

    def _compute_error_scores(
        self,
        df: pd.DataFrame,
        weights: List[float] = [1, 1, 1],
        ts_validation: bool = True,
    ) -> pd.Series:
        """Calculate error scores for models"""
        self.logger.debug("Computing error scores")
        self.logger.debug(
            f"Parameters: weights={weights}, ts_validation={ts_validation}"
        )

        assert len(weights) == 3, "Weights must have exactly 3 values"

        error_cols = [
            "nrmse_test" if ts_validation else "nrmse_train",
            "decomp.rssd",
            "mape",
        ]

        missing_cols = [col for col in error_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            raise AssertionError(f"Missing columns: {missing_cols}")

        balance = np.array(weights) / sum(weights)
        self.logger.debug(f"Normalized weights: {balance}")

        scores = df[error_cols].copy()
        scores = scores.rename(columns={error_cols[0]: "nrmse"})

        # Handle infinite values
        for col in ["nrmse", "decomp.rssd", "mape"]:
            inf_count = np.sum(np.isinf(scores[col]))
            if inf_count > 0:
                self.logger.warning(f"Found {inf_count} infinite values in {col}")
                scores[col] = np.where(
                    np.isinf(scores[col]),
                    np.max(scores[col][~np.isinf(scores[col])]),
                    scores[col],
                )

        # Normalize scores
        scores["nrmse_n"] = ParetoUtils.min_max_norm(scores["nrmse"])
        scores["decomp.rssd_n"] = ParetoUtils.min_max_norm(scores["decomp.rssd"])
        scores["mape_n"] = ParetoUtils.min_max_norm(scores["mape"])

        scores.fillna(0, inplace=True)

        # Apply weights
        scores["nrmse_w"] = balance[0] * scores["nrmse_n"]
        scores["decomp.rssd_w"] = balance[1] * scores["decomp.rssd_n"]
        scores["mape_w"] = balance[2] * scores["mape_n"]

        # Calculate final error score
        scores["error_score"] = np.sqrt(
            scores["nrmse_w"] ** 2
            + scores["decomp.rssd_w"] ** 2
            + scores["mape_w"] ** 2
        )

        self.logger.debug(
            f"Error score statistics: min={scores['error_score'].min():.4f}, "
            f"max={scores['error_score'].max():.4f}, "
            f"mean={scores['error_score'].mean():.4f}"
        )

        return scores["error_score"]

    def _select_top_solutions(
        self, clustered_df: pd.DataFrame, config: ClusteringConfig, limit: int = 1
    ) -> pd.DataFrame:
        """Selects top solutions from each cluster"""
        self.logger.info(f"Selecting top {limit} solutions per cluster")
        self.logger.debug(f"Input dataset shape: {clustered_df.shape}")

        ts_validation: bool = "nrmse_test" in clustered_df.columns
        df = clustered_df.copy()

        self.logger.debug("Computing error scores")
        df["error_score"] = self._compute_error_scores(
            df, config.weights, ts_validation
        )

        df.fillna(0, inplace=True)

        self.logger.debug("Sorting and ranking solutions")
        df = df.sort_values(by=["cluster", "error_score"])
        initial_solutions = len(df)
        df = df.groupby("cluster", observed=False).head(limit)
        df["rank"] = df.groupby("cluster", observed=False)["error_score"].rank(
            method="dense"
        )
        df = df[
            ["cluster", "rank"]
            + [col for col in df.columns if col not in ["cluster", "rank"]]
        ]
        self.logger.info(f"Selected {len(df)} solutions from {initial_solutions} total")
        RobynLogger.log_df(self.logger, df)

        return df

    def _bootstrap_sampling(
        self, sample: pd.Series, boot_n: int, seed: int = 1
    ) -> Dict[str, Union[np.ndarray, List[float]]]:
        self.logger.debug(
            f"Starting bootstrap sampling with sample size: {len(sample)} and boot_n: {boot_n}"
        )

        # Log sample statistics at debug level
        self.logger.debug(
            f"Sample statistics - Mean: {sample.mean():.4f}, "
            f"Std: {sample.std():.4f}, "
            f"Non-null count: {sample.count()}"
        )
        np.random.seed(seed)

        valid_samples = sample[~sample.isna()]
        if len(valid_samples) > 1:
            # Calculate initial statistics
            sample_n = len(sample)
            sample_mean = float(
                np.nanmean(sample)
            )  # Equivalent to mean(samp, na.rm=TRUE)

            # Generate bootstrap samples
            boot_sample = np.random.choice(
                valid_samples,  # Use only valid samples for bootstrapping
                size=(boot_n, sample_n),
                replace=True,
            )

            # Calculate bootstrap means
            boot_means = np.mean(boot_sample, axis=1)

            # Calculate standard error
            se = float(np.std(boot_means))

            # Calculate margin of error and confidence interval
            me = stats.t.ppf(0.975, sample_n - 1) * se
            sample_me = np.sqrt(sample_n) * me
            ci = [sample_mean - sample_me, sample_mean + sample_me]

            return {
                "boot_means": boot_means,
                "ci": ci,
                "se": se,  # Return single value instead of list
            }
        else:
            # Handle case with insufficient samples
            if len(valid_samples) == 0:
                single_value = np.nan
            else:
                single_value = valid_samples.iloc[0]

            return {
                "boot_means": np.array([single_value]),
                "ci": [single_value, single_value],
                "se": 0.0,  # Return single value instead of list
            }
