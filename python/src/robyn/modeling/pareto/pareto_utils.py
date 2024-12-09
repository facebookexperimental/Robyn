# pyre-strict

from typing import List

import logging
import numpy as np
import pandas as pd
from robyn.modeling.entities.clustering_results import ClusteredResult
from robyn.modeling.entities.pareto_result import ParetoResult


class ParetoUtils:
    """
    Utility class for Pareto optimization in marketing mix models.

    This class provides various utility methods for Pareto front calculation,
    error scoring, and other helper functions used in the Pareto optimization process.
    It maintains state across operations, allowing for caching of intermediate results
    and configuration of optimization parameters.
    """

    def __init__(self):
        """
        Initialize the ParetoUtils instance.
        """
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_errors_scores(
        df: pd.DataFrame, balance: List[float] = [1, 1, 1], ts_validation: bool = True
    ) -> np.ndarray:
        """
        Calculate combined error scores based on NRMSE, DECOMP.RSSD, and MAPE.

        Args:
            df (pd.DataFrame): DataFrame containing error columns.
            balance (List[float]): Weights for NRMSE, DECOMP.RSSD, and MAPE. Defaults to [1, 1, 1].
            ts_validation (bool): If True, use 'nrmse_test', else use 'nrmse_train'. Defaults to True.

        Returns:
            np.ndarray: Array of calculated error scores.
        """
        assert len(balance) == 3, "Balance must be a list of 3 values"

        error_cols = [
            "nrmse_test" if ts_validation else "nrmse_train",
            "decomp.rssd",
            "mape",
        ]
        assert all(
            col in df.columns for col in error_cols
        ), f"Missing columns: {[col for col in error_cols if col not in df.columns]}"

        # Normalize balance weights
        balance = np.array(balance) / sum(balance)

        # Select and rename columns
        errors = df[error_cols].copy()
        errors.columns = ["nrmse", "decomp.rssd", "mape"]

        # Replace infinite values with the maximum finite value
        for col in errors.columns:
            max_val = errors[np.isfinite(errors[col])][col].max()
            errors[col] = errors[col].apply(lambda x: max_val if np.isinf(x) else x)

        # Normalize error values
        for col in errors.columns:
            errors[f"{col}_n"] = ParetoUtils.min_max_norm(errors[col])

        # Replace NaN with 0
        errors = errors.fillna(0)

        # Apply balance weights
        errors["nrmse_w"] = balance[0] * errors["nrmse_n"]
        errors["decomp.rssd_w"] = balance[1] * errors["decomp.rssd_n"]
        errors["mape_w"] = balance[2] * errors["mape_n"]

        # Calculate error score
        errors["error_score"] = np.sqrt(
            errors["nrmse_w"] ** 2
            + errors["decomp.rssd_w"] ** 2
            + errors["mape_w"] ** 2
        )

        return errors["error_score"].values

    @staticmethod
    def min_max_norm(x: pd.Series, min: float = 0, max: float = 1) -> pd.Series:
        x = x.replace([np.inf, -np.inf], np.max(np.isfinite(x)))
        if len(x) <= 1:
            return x
        a, b = x.min(), x.max()
        if b - a != 0:
            return (max - min) * (x - a) / (b - a) + min
        else:
            return x

    @staticmethod
    def calculate_fx_objective(
        x: float,
        coeff: float,
        alpha: float,
        inflexion: float,
        x_hist_carryover: float,
        get_sum: bool = True,
    ) -> float:
        # Adstock scales
        x_adstocked = x + np.mean(x_hist_carryover)

        # Hill transformation
        if get_sum:
            x_out = coeff * np.sum((1 + inflexion**alpha / x_adstocked**alpha) ** -1)
        else:
            x_out = coeff * ((1 + inflexion**alpha / x_adstocked**alpha) ** -1)

        return x_out

    def process_pareto_clustered_results(
        self,
        pareto_results: ParetoResult,
        clustered_result: ClusteredResult,
        ran_cluster: bool = True,
        ran_calibration: bool = False,
    ) -> ParetoResult:
        """
        Process Pareto optimization results and update the internal state.

        Args:
            pareto_results (ParetoResult): Pareto optimization results.
            clustered_result (ClusteredResult): Clustered results.
            ran_cluster (bool): Whether to run clustering.
            ran_calibration (bool): Whether calibration was run.
        Returns:
            ParetoResult: Updated Pareto optimization results.
        """
        all_solutions = pareto_results.pareto_solutions

        # Common logic for all cases
        x_decomp_agg = pareto_results.x_decomp_agg[
            pareto_results.x_decomp_agg["sol_id"].isin(all_solutions)
        ]
        result_hyp_param = pareto_results.result_hyp_param[
            pareto_results.result_hyp_param["sol_id"].isin(all_solutions)
        ]
        result_calibration = (
            pareto_results.result_calibration[
                pareto_results.result_calibration["sol_id"].isin(all_solutions)
            ]
            if ran_calibration and pareto_results.result_calibration is not None
            else None
        )

        if ran_cluster:
            # Select common columns from cluster data
            common_clustered_df = clustered_result.cluster_data[
                ["sol_id", "cluster", "top_sol"]
            ]

            result_hyp_param = pd.merge(
                result_hyp_param, common_clustered_df, on="sol_id", how="left"
            )

            x_decomp_agg = (
                pd.merge(x_decomp_agg, common_clustered_df, on="sol_id", how="left")
                .merge(
                    clustered_result.cluster_ci.cluster_confidence_interval_df[
                        ["rn", "cluster", "boot_mean", "boot_se", "ci_low", "ci_up"]
                    ],
                    on=["rn", "cluster"],
                    how="left",
                )
                .merge(
                    pareto_results.df_caov_pct_all[
                        pareto_results.df_caov_pct_all["type"] == "Carryover"
                    ][["sol_id", "rn", "carryover_pct"]],
                    on=["sol_id", "rn"],
                    how="left",
                )
            )

            media_vec_collect = pd.merge(
                pareto_results.media_vec_collect,
                common_clustered_df,
                on="sol_id",
                how="left",
            )

            # Join xDecompVecCollect
            x_decomp_vec_collect = pd.merge(
                pareto_results.x_decomp_vec_collect,
                common_clustered_df,
                on="sol_id",
                how="left",
            )

            if ran_calibration and pareto_results.result_calibration is not None:
                result_calibration = pd.merge(
                    result_calibration, common_clustered_df, on="sol_id", how="left"
                )

            return ParetoResult(
                pareto_solutions=all_solutions,
                x_decomp_agg=x_decomp_agg,
                result_hyp_param=result_hyp_param,
                result_calibration=result_calibration,
                media_vec_collect=media_vec_collect,
                x_decomp_vec_collect=x_decomp_vec_collect,
                pareto_fronts=pareto_results.pareto_fronts,
                df_caov_pct_all=pareto_results.df_caov_pct_all,
                plot_data_collect=pareto_results.plot_data_collect,
            )
        else:
            return ParetoResult(
                pareto_solutions=all_solutions,
                x_decomp_agg=x_decomp_agg,
                result_hyp_param=result_hyp_param,
                result_calibration=result_calibration,
                media_vec_collect=pareto_results.media_vec_collect,
                x_decomp_vec_collect=pareto_results.x_decomp_vec_collect,
                pareto_fronts=pareto_results.pareto_fronts,
                df_caov_pct_all=pareto_results.df_caov_pct_all,
                plot_data_collect=pareto_results.plot_data_collect,
            )
