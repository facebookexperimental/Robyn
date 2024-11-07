# pyre-strict

from concurrent.futures import as_completed, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.holidays_data import HolidaysData

from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.pareto.hill_calculator import HillCalculator
from robyn.modeling.pareto.immediate_carryover import ImmediateCarryoverCalculator
from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.modeling.pareto.response_curve import ResponseCurveCalculator, ResponseOutput
from robyn.modeling.transformations.transformations import Transformation
from tqdm import tqdm  # Import tqdm for progress bar  # Import tqdm for progress bar


@dataclass
class ParetoData:
    decomp_spend_dist: pd.DataFrame
    result_hyp_param: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    pareto_fronts: List[int]


class ParetoOptimizer:
    """
    Performs Pareto optimization on marketing mix models.

    This class orchestrates the Pareto optimization process, including data aggregation,
    Pareto front calculation, response curve calculation, and plot data preparation.

    Attributes:
        mmm_data (MMMData): Input data for the marketing mix model.
        model_outputs (ModelOutputs): Output data from the model runs.
        response_calculator (ResponseCurveCalculator): Calculator for response curves.
        carryover_calculator (ImmediateCarryoverCalculator): Calculator for immediate and carryover effects.
        pareto_utils (ParetoUtils): Utility functions for Pareto-related calculations.
    """

    def __init__(
        self,
        mmm_data: MMMData,
        model_outputs: ModelOutputs,
        hyper_parameter: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
        holidays_data: HolidaysData,
    ):
        """
        Initialize the ParetoOptimizer.

        Args:
            mmm_data (MMMData): Input data for the marketing mix model.
            model_outputs (ModelOutputs): Output data from the model runs.
            hyper_parameter (Hyperparameters): Hyperparameters for the model runs.
        """
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        self.hyper_parameter = hyper_parameter
        self.featurized_mmm_data = featurized_mmm_data
        self.holidays_data = holidays_data

        self.transformer = Transformation(mmm_data)

    def optimize(
        self,
        pareto_fronts: str = "auto",
        min_candidates: int = 100,
        calibration_constraint: float = 0.1,
        calibrated: bool = False,
    ) -> ParetoResult:
        """
        Perform Pareto optimization on the model results.

        This method orchestrates the entire Pareto optimization process, including data aggregation,
        Pareto front calculation, response curve calculation, and preparation of plot data.

        Args:
            pareto_fronts (str): Number of Pareto fronts to consider or "auto" for automatic selection.
            min_candidates (int): Minimum number of candidates to consider when using "auto" Pareto fronts.
            calibration_constraint (float): Constraint for calibration, used if models are calibrated.
            calibrated (bool): Whether the models have undergone calibration.

        Returns:
            ParetoResult: The results of the Pareto optimization process.
        """
        aggregated_data = self._aggregate_model_data(calibrated)
        aggregated_data["result_hyp_param"] = self._compute_pareto_fronts(
            aggregated_data, pareto_fronts, min_candidates, calibration_constraint
        )

        pareto_data = self.prepare_pareto_data(aggregated_data, pareto_fronts, min_candidates, calibrated)
        pareto_data = self._compute_response_curves(pareto_data, aggregated_data)
        plotting_data = self._generate_plot_data(aggregated_data, pareto_data)

        return ParetoResult(
            pareto_solutions=plotting_data["pareto_solutions"],
            pareto_fronts=pareto_fronts,
            result_hyp_param=aggregated_data["result_hyp_param"],
            result_calibration=aggregated_data["result_calibration"],
            x_decomp_agg=pareto_data.x_decomp_agg,
            media_vec_collect=plotting_data["mediaVecCollect"],
            x_decomp_vec_collect=plotting_data["xDecompVecCollect"],
            plot_data_collect=plotting_data["plotDataCollect"],
            df_caov_pct_all=plotting_data["df_caov_pct_all"],
        )

    def _aggregate_model_data(self, calibrated: bool) -> Dict[str, pd.DataFrame]:
        """
        Aggregate and prepare data from model outputs for Pareto optimization.

        This method combines hyperparameters, decomposition results, and calibration data (if applicable)
        from all model runs into a format suitable for Pareto optimization.

        Args:
            calibrated (bool): Whether the models have undergone calibration.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing aggregated data, including:
                - 'result_hyp_param': Hyperparameters for all model runs
                - 'x_decomp_agg': Aggregated decomposition results
                - 'result_calibration': Calibration results (if calibrated is True)
        """
        hyper_fixed = self.model_outputs.hyper_fixed
        # Extract resultCollect from self.model_outputs
        trials = [model for model in self.model_outputs.trials if hasattr(model, "resultCollect")]

        # Create lists of resultHypParam and xDecompAgg using list comprehension
        resultHypParam_list = [trial.result_hyp_param for trial in self.model_outputs.trials]
        xDecompAgg_list = [trial.x_decomp_agg for trial in self.model_outputs.trials]

        # Concatenate the lists into DataFrames using pd.concat
        resultHypParam = pd.concat(resultHypParam_list, ignore_index=True)
        xDecompAgg = pd.concat(xDecompAgg_list, ignore_index=True)

        if calibrated:
            resultCalibration = pd.concat([pd.DataFrame(trial.liftCalibration) for trial in trials])
            resultCalibration = resultCalibration.rename(columns={"liftMedia": "rn"})
        else:
            resultCalibration = None
        if not hyper_fixed:
            df_names = [resultHypParam, xDecompAgg]
            if calibrated:
                df_names.append(resultCalibration)
            for df in df_names:
                df["iterations"] = (df["iterNG"] - 1) * self.model_outputs.cores + df["iterPar"]
        elif hyper_fixed and calibrated:
            df_names = [resultCalibration]
            for df in df_names:
                df["iterations"] = (df["iterNG"] - 1) * self.model_outputs.cores + df["iterPar"]

        # Check if recreated model and bootstrap results are available
        if len(xDecompAgg["solID"].unique()) == 1 and "boot_mean" not in xDecompAgg.columns:
            # Get bootstrap results from model_outputs object
            bootstrap = getattr(self.model_outputs, "bootstrap", None)
            if bootstrap is not None:
                # Merge bootstrap results with xDecompAgg using left join
                xDecompAgg = pd.merge(xDecompAgg, bootstrap, left_on="rn", right_on="variable")

        return {
            "result_hyp_param": resultHypParam,
            "x_decomp_agg": xDecompAgg,
            "result_calibration": resultCalibration,
        }

    def _compute_pareto_fronts(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_fronts: str,
        min_candidates: int,
        calibration_constraint: float,
    ) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        This method identifies Pareto-optimal solutions based on NRMSE and DECOMP.RSSD
        optimization criteria and assigns them to Pareto fronts.

        Args:
            resultHypParamPareto (pd.DataFrame): DataFrame containing model results,
                                                including 'nrmse' and 'decomp.rssd' columns.
            pareto_fronts (str): Number of Pareto fronts to compute or "auto".

        Returns:
            pd.DataFrame: A dataframe of Pareto-optimal solutions with their corresponding front numbers.
        """
        resultHypParam = aggregated_data["result_hyp_param"]
        xDecompAgg = aggregated_data["x_decomp_agg"]
        resultCalibration = aggregated_data["result_calibration"]

        if not self.model_outputs.hyper_fixed:
            # Filter and group data to calculate coef0
            xDecompAggCoef0 = (
                xDecompAgg[xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends)]
                .groupby("solID")["coef"]
                .apply(lambda x: min(x.dropna()) == 0)
            )
            # calculate quantiles
            mape_lift_quantile10 = resultHypParam["mape"].quantile(calibration_constraint)
            nrmse_quantile90 = resultHypParam["nrmse"].quantile(0.9)
            decomprssd_quantile90 = resultHypParam["decomp.rssd"].quantile(0.9)
            # merge resultHypParam with xDecompAggCoef0
            resultHypParam = pd.merge(resultHypParam, xDecompAggCoef0, on="solID", how="left")
            # create a new column 'mape.qt10'
            resultHypParam["mape.qt10"] = (
                (resultHypParam["mape"] <= mape_lift_quantile10)
                & (resultHypParam["nrmse"] <= nrmse_quantile90)
                & (resultHypParam["decomp.rssd"] <= decomprssd_quantile90)
            )
            # filter resultHypParam
            resultHypParamPareto = resultHypParam[resultHypParam["mape.qt10"] == True]
            # calculate Pareto front
            pareto_fronts_df = ParetoOptimizer._pareto_fronts(resultHypParamPareto, pareto_fronts=pareto_fronts)
            # merge resultHypParamPareto with pareto_fronts_df
            resultHypParamPareto = pd.merge(
                resultHypParamPareto,
                pareto_fronts_df,
                left_on=["nrmse", "decomp.rssd"],
                right_on=["x", "y"],
            )
            resultHypParamPareto = resultHypParamPareto.rename(columns={"pareto_front": "robynPareto"})
            resultHypParamPareto = resultHypParamPareto.sort_values(["iterNG", "iterPar", "nrmse"])[
                ["solID", "robynPareto"]
            ]
            resultHypParamPareto = resultHypParamPareto.groupby("solID").first().reset_index()
            resultHypParam = pd.merge(resultHypParam, resultHypParamPareto, on="solID", how="left")
        else:
            resultHypParam = resultHypParam.assign(mape_qt10=True, robynPareto=1, coef0=np.nan)

        # Calculate combined weighted error scores
        resultHypParam["error_score"] = ParetoUtils.calculate_errors_scores(
            df=resultHypParam, ts_validation=self.model_outputs.ts_validation
        )
        return resultHypParam

    @staticmethod
    def _pareto_fronts(resultHypParamPareto: pd.DataFrame, pareto_fronts: str) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        This method identifies Pareto-optimal solutions based on NRMSE and DECOMP.RSSD
        optimization criteria and assigns them to Pareto fronts.

        Args:
            resultHypParamPareto (pd.DataFrame): DataFrame containing model results,
                                                including 'nrmse' and 'decomp.rssd' columns.
            pareto_fronts (Union[str, int]): Number of Pareto fronts to calculate or "auto".
        """
        # Ensure nrmse_values and decomp_rssd_values have the same length
        nrmse_values = resultHypParamPareto["nrmse"]
        decomp_rssd_values = resultHypParamPareto["decomp.rssd"]

        # Ensure nrmse_values and decomp_rssd_values have the same length
        if len(nrmse_values) != len(decomp_rssd_values):
            raise ValueError("Length of nrmse_values must be equal to length of decomp_rssd_values")

        # Create a DataFrame from nrmse and decomp_rssd
        data = pd.DataFrame({"nrmse": nrmse_values, "decomp_rssd": decomp_rssd_values})

        # Sort the DataFrame by nrmse and decomp_rssd
        sorted_data = data.sort_values(by=["nrmse", "decomp_rssd"]).reset_index(drop=True)
        pareto_fronts_df = pd.DataFrame()
        front_number = 1

        # Determine the maximum number of Pareto fronts
        max_fronts = float("inf") if "auto" in pareto_fronts else pareto_fronts

        # Loop to identify Pareto fronts
        while len(sorted_data) > 0 and front_number <= max_fronts:
            # Select non-duplicated cumulative minimums of decomp_rssd
            pareto_candidates: pd.DataFrame = sorted_data[
                sorted_data["decomp_rssd"] == sorted_data["decomp_rssd"].cummin()
            ]
            pareto_candidates = pareto_candidates.assign(pareto_front=front_number)

            # Append to the result DataFrame
            pareto_fronts_df = pd.concat([pareto_fronts_df, pareto_candidates], ignore_index=True)

            # Remove selected rows from sorted_data
            sorted_data = sorted_data[~sorted_data.index.isin(pareto_candidates.index)]

            # Increment the Pareto front counter
            front_number += 1

        # Merge the original DataFrame with the Pareto front DataFrame
        result = pd.merge(
            data,
            pareto_fronts_df[["nrmse", "decomp_rssd", "pareto_front"]],
            on=["nrmse", "decomp_rssd"],
            how="left",
        )
        result.columns = ["x", "y", "pareto_front"]
        return result.reset_index(drop=True)

    def prepare_pareto_data(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_fronts: Union[str, int],
        min_candidates: int,
        calibrated: bool,
    ) -> ParetoData:
        result_hyp_param = aggregated_data["result_hyp_param"]

        # Debug print for columns
        print("\nColumns in result_hyp_param:", result_hyp_param.columns.tolist())

        # Ensure consistent column naming
        if "solID" in result_hyp_param.columns:
            result_hyp_param = result_hyp_param.rename(columns={"solID": "sol_id"})

        # 1. Binding Pareto results
        if "solID" in aggregated_data["x_decomp_agg"].columns:
            aggregated_data["x_decomp_agg"] = aggregated_data["x_decomp_agg"].rename(columns={"solID": "sol_id"})

        aggregated_data["x_decomp_agg"] = pd.merge(
            aggregated_data["x_decomp_agg"],
            result_hyp_param[["robynPareto", "sol_id"]],
            on="sol_id",
            how="left",
        )

        # Step 1: Collect decomp_spend_dist from each trial
        trial_dfs = []
        for trial in self.model_outputs.trials:
            if hasattr(trial, "decomp_spend_dist") and trial.decomp_spend_dist is not None:
                print(f"Found decomp_spend_dist in trial with shape: {trial.decomp_spend_dist.shape}")
                trial_dfs.append(trial.decomp_spend_dist)
            else:
                print("Trial has no decomp_spend_dist or it is None")

        if not trial_dfs:
            raise ValueError("No valid decomp_spend_dist found in any trials")

        decomp_spend_dist = pd.concat(trial_dfs, ignore_index=True)
        print(f"\nConcatenated decomp_spend_dist shape: {decomp_spend_dist.shape}")

        # Step 2: Add sol_id if hyper_fixed is False
        if not self.model_outputs.hyper_fixed:
            decomp_spend_dist["sol_id"] = (
                decomp_spend_dist["trial"].astype(str)
                + "_"
                + decomp_spend_dist["iterNG"].astype(str)
                + "_"
                + decomp_spend_dist["iterPar"].astype(str)
            )

        # Step 3: Left join with resultHypParam
        decomp_spend_dist = pd.merge(
            decomp_spend_dist,
            result_hyp_param[["robynPareto", "sol_id"]],
            on="sol_id",
            how="left",
        )

        # Debug print for columns after processing
        print("\nColumns in decomp_spend_dist:", decomp_spend_dist.columns.tolist())

        # Handle pareto fronts determination
        if self.model_outputs.hyper_fixed or len(result_hyp_param) == 1:
            num_pareto_fronts = 1
        elif isinstance(pareto_fronts, str) and pareto_fronts.lower() == "auto":
            # 4. Handling automatic Pareto front selection
            n_pareto = result_hyp_param["robynPareto"].notna().sum()

            if n_pareto <= min_candidates and len(result_hyp_param) > 1 and not calibrated:
                raise ValueError(
                    f"Less than {min_candidates} candidates in pareto fronts. "
                    "Increase iterations to get more model candidates or decrease min_candidates."
                )

            # Group by 'robynPareto' and count distinct 'sol_id'
            grouped_data = (
                result_hyp_param[result_hyp_param["robynPareto"].notna()]
                .groupby("robynPareto", as_index=False)
                .agg(n=("sol_id", "nunique"))
            )
            # Calculate cumulative sum and create a new column 'n_cum'
            grouped_data["n_cum"] = grouped_data["n"].cumsum()

            # Filter where cumulative sum is greater than or equal to min_candidates
            auto_pareto = grouped_data[grouped_data["n_cum"] >= min_candidates].head(1)

            if auto_pareto.empty:
                num_pareto_fronts = 1
            else:
                num_pareto_fronts = int(auto_pareto["robynPareto"].iloc[0])

            print(
                f">> Automatically selected {num_pareto_fronts} Pareto-fronts ",
                f"to contain at least {min_candidates} pareto-optimal models",
            )
        else:
            num_pareto_fronts = int(pareto_fronts)

        # Create Pareto front vector
        pareto_fronts_vec = list(range(1, num_pareto_fronts + 1))

        # Filtering data for selected Pareto fronts
        decomp_spend_dist_pareto = decomp_spend_dist[decomp_spend_dist["robynPareto"].isin(pareto_fronts_vec)]
        result_hyp_param_pareto = result_hyp_param[result_hyp_param["robynPareto"].isin(pareto_fronts_vec)]
        x_decomp_agg_pareto = aggregated_data["x_decomp_agg"][
            aggregated_data["x_decomp_agg"]["robynPareto"].isin(pareto_fronts_vec)
        ]

        return ParetoData(
            decomp_spend_dist=decomp_spend_dist_pareto,
            result_hyp_param=result_hyp_param_pareto,
            x_decomp_agg=x_decomp_agg_pareto,
            pareto_fronts=pareto_fronts_vec,
        )

    def _compute_response_curves(
        self, pareto_data: ParetoData, aggregated_data: Dict[str, pd.DataFrame]
    ) -> ParetoData:
        """
        Calculate response curves for Pareto-optimal solutions.
        """
        print(
            f">>> Calculating response curves for all models' media variables ({len(pareto_data.decomp_spend_dist)})..."
        )

        # Debug print statements
        print("\nDecomp spend dist columns:", pareto_data.decomp_spend_dist.columns.tolist())

        # Ensure we have either solID or sol_id in decomp_spend_dist
        if "solID" in pareto_data.decomp_spend_dist.columns:
            pareto_data.decomp_spend_dist = pareto_data.decomp_spend_dist.rename(columns={"solID": "sol_id"})
        elif "sol_id" not in pareto_data.decomp_spend_dist.columns:
            raise ValueError("Neither 'solID' nor 'sol_id' found in decomp_spend_dist columns")

        # Parallel processing
        run_dt_resp_partial = partial(self.run_dt_resp, paretoData=pareto_data)

        if self.model_outputs.cores > 1:
            with ProcessPoolExecutor(max_workers=self.model_outputs.cores) as executor:
                futures = [
                    executor.submit(run_dt_resp_partial, row) for _, row in pareto_data.decomp_spend_dist.iterrows()
                ]
                resp_collect = pd.DataFrame([f.result() for f in as_completed(futures)])
        else:
            resp_collect = pareto_data.decomp_spend_dist.apply(run_dt_resp_partial, axis=1)

        # Debug print statements
        print("\nResp collect columns:", resp_collect.columns.tolist())
        print("\nResp collect first row:", resp_collect.iloc[0] if not resp_collect.empty else "Empty")

        # Ensure consistent column naming in response collection
        if "solID" in resp_collect.columns:
            resp_collect = resp_collect.rename(columns={"solID": "sol_id"})

        # Additional debugging
        print("\nColumns after renaming:")
        print("decomp_spend_dist columns:", pareto_data.decomp_spend_dist.columns.tolist())
        print("resp_collect columns:", resp_collect.columns.tolist())

        # Merge results
        try:
            pareto_data.decomp_spend_dist = pd.merge(
                pareto_data.decomp_spend_dist, resp_collect, on=["sol_id", "rn"], how="left"
            )
        except KeyError as e:
            print(f"\nError during merge: {str(e)}")
            raise

        # Calculate ROI and CPA metrics after merging
        pareto_data.decomp_spend_dist["roi_mean"] = (
            pareto_data.decomp_spend_dist["mean_response"] / pareto_data.decomp_spend_dist["mean_spend"]
        )
        pareto_data.decomp_spend_dist["roi_total"] = (
            pareto_data.decomp_spend_dist["xDecompAgg"] / pareto_data.decomp_spend_dist["total_spend"]
        )
        pareto_data.decomp_spend_dist["cpa_mean"] = (
            pareto_data.decomp_spend_dist["mean_spend"] / pareto_data.decomp_spend_dist["mean_response"]
        )
        pareto_data.decomp_spend_dist["cpa_total"] = (
            pareto_data.decomp_spend_dist["total_spend"] / pareto_data.decomp_spend_dist["xDecompAgg"]
        )

        # Ensure consistent column naming in x_decomp_agg
        if "solID" in aggregated_data["x_decomp_agg"].columns:
            aggregated_data["x_decomp_agg"] = aggregated_data["x_decomp_agg"].rename(columns={"solID": "sol_id"})

        pareto_data.x_decomp_agg = pd.merge(
            aggregated_data["x_decomp_agg"],
            pareto_data.decomp_spend_dist[
                [
                    "rn",
                    "sol_id",
                    "total_spend",
                    "mean_spend",
                    "mean_spend_adstocked",
                    "mean_carryover",
                    "mean_response",
                    "spend_share",
                    "effect_share",
                    "roi_mean",
                    "roi_total",
                    "cpa_total",
                ]
            ],
            on=["sol_id", "rn"],
            how="left",
        )

        return pareto_data

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up column names to ensure consistency.
        """
        if "solID" in df.columns and "sol_id" in df.columns:
            # If both exist, drop solID and keep sol_id
            df = df.drop(columns=["solID"])
        elif "solID" in df.columns:
            # If only solID exists, rename it
            df = df.rename(columns={"solID": "sol_id"})
        return df

    def _compute_response_curves(
        self, pareto_data: ParetoData, aggregated_data: Dict[str, pd.DataFrame]
    ) -> ParetoData:
        """
        Calculate response curves for Pareto-optimal solutions.
        """
        print(
            f">>> Calculating response curves for all models' media variables ({len(pareto_data.decomp_spend_dist)})..."
        )

        # Check if decomp_spend_dist is empty
        if pareto_data.decomp_spend_dist.empty:
            print("Warning: decomp_spend_dist is empty. No response curves to calculate.")
            return pareto_data

        # Clean up column names
        pareto_data.decomp_spend_dist = self._clean_column_names(pareto_data.decomp_spend_dist)
        pareto_data.result_hyp_param = self._clean_column_names(pareto_data.result_hyp_param)
        pareto_data.x_decomp_agg = self._clean_column_names(pareto_data.x_decomp_agg)

        # Debug print statements
        print("\nDecomp spend dist columns after cleanup:", pareto_data.decomp_spend_dist.columns.tolist())
        print("Number of rows in decomp_spend_dist:", len(pareto_data.decomp_spend_dist))
        print("Sample of decomp_spend_dist 'sol_id' values:", pareto_data.decomp_spend_dist["sol_id"].head().tolist())

        # If decomp_spend_dist has no valid sol_id values, we need to fix that
        if pareto_data.decomp_spend_dist["sol_id"].isna().all():
            print("Creating sol_id from trial, iterNG, and iterPar...")
            pareto_data.decomp_spend_dist["sol_id"] = (
                pareto_data.decomp_spend_dist["trial"].astype(str)
                + "_"
                + pareto_data.decomp_spend_dist["iterNG"].astype(str)
                + "_"
                + pareto_data.decomp_spend_dist["iterPar"].astype(str)
            )

        # Additional check for required columns
        required_cols = ["sol_id", "rn", "mean_spend"]
        if not all(col in pareto_data.decomp_spend_dist.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in pareto_data.decomp_spend_dist.columns]
            raise ValueError(f"Missing required columns in decomp_spend_dist: {missing_cols}")

        # Parallel processing
        run_dt_resp_partial = partial(self.run_dt_resp, paretoData=pareto_data)

        try:
            if self.model_outputs.cores > 1:
                with ProcessPoolExecutor(max_workers=self.model_outputs.cores) as executor:
                    futures = [
                        executor.submit(run_dt_resp_partial, row)
                        for _, row in pareto_data.decomp_spend_dist.iterrows()
                    ]
                    resp_collect = pd.DataFrame([f.result() for f in as_completed(futures)])
            else:
                resp_collect = pd.DataFrame(
                    [run_dt_resp_partial(row) for _, row in pareto_data.decomp_spend_dist.iterrows()]
                )
        except Exception as e:
            print(f"Error during response calculation: {str(e)}")
            raise

        # If resp_collect is empty after calculations, something went wrong
        if resp_collect.empty:
            print("Warning: No response curves were calculated. Check the input data and parameters.")
            return pareto_data

        # Debug print statements for response collection
        print("\nResp collect columns:", resp_collect.columns.tolist())
        print("Number of rows in resp_collect:", len(resp_collect))
        if not resp_collect.empty:
            print("Sample of resp_collect:", resp_collect.head())

        # Clean up response collection
        resp_collect = self._clean_column_names(resp_collect)

        # Additional debugging
        print("\nColumns after cleaning:")
        print("decomp_spend_dist columns:", pareto_data.decomp_spend_dist.columns.tolist())
        print("resp_collect columns:", resp_collect.columns.tolist())

        # Merge results
        try:
            print("\nAttempting merge with the following columns:")
            print("Left (decomp_spend_dist) unique sol_id values:", pareto_data.decomp_spend_dist["sol_id"].nunique())
            print("Left (decomp_spend_dist) sol_id sample:", pareto_data.decomp_spend_dist["sol_id"].head().tolist())
            print("Right (resp_collect) unique sol_id values:", resp_collect["sol_id"].nunique())
            print("Right (resp_collect) sol_id sample:", resp_collect["sol_id"].head().tolist())

            pareto_data.decomp_spend_dist = pd.merge(
                pareto_data.decomp_spend_dist,
                resp_collect,
                on=["sol_id", "rn"],
                how="left",
                validate="1:1",  # Ensure we're not duplicating rows
            )
        except KeyError as e:
            print(f"\nError during merge: {str(e)}")
            print("Left DataFrame columns:", pareto_data.decomp_spend_dist.columns.tolist())
            print("Right DataFrame columns:", resp_collect.columns.tolist())
            raise

        return pareto_data

    def run_dt_resp(self, row: pd.Series, paretoData: ParetoData) -> pd.Series:
        """
        Calculate response curves for a given row of Pareto data.
        """
        # Ensure we're using the correct column name
        get_solID = row["sol_id"] if "sol_id" in row else row.get("solID")
        if get_solID is None:
            raise ValueError("Neither 'sol_id' nor 'solID' found in row")

        get_spendname = row["rn"]
        startRW = self.mmm_data.mmmdata_spec.rolling_window_start_which
        endRW = self.mmm_data.mmmdata_spec.rolling_window_end_which

        response_calculator = ResponseCurveCalculator(
            mmm_data=self.mmm_data,
            model_outputs=self.model_outputs,
            hyperparameter=self.hyperparameter,
        )

        # Clean up DataFrames before using them
        dt_hyppar = self._clean_column_names(paretoData.result_hyp_param)
        dt_coef = self._clean_column_names(paretoData.x_decomp_agg)

        response_output: ResponseOutput = response_calculator.calculate_response(
            select_model=get_solID,
            metric_name=get_spendname,
            date_range="all",
            dt_hyppar=dt_hyppar,
            dt_coef=dt_coef,
            quiet=True,
        )

        mean_spend_adstocked = np.mean(response_output.input_total[startRW:endRW])
        mean_carryover = np.mean(response_output.input_carryover[startRW:endRW])

        dt_hyppar = dt_hyppar[dt_hyppar["sol_id"] == get_solID]
        chn_adstocked = pd.DataFrame({get_spendname: response_output.input_total[startRW:endRW]})
        dt_coef = dt_coef[(dt_coef["sol_id"] == get_solID) & (dt_coef["rn"] == get_spendname)][["rn", "coef"]]

        hill_calculator = HillCalculator(
            mmmdata=self.mmm_data,
            model_outputs=self.model_outputs,
            dt_hyppar=dt_hyppar,
            dt_coef=dt_coef,
            media_spend_sorted=[get_spendname],
            select_model=get_solID,
            chn_adstocked=chn_adstocked,
        )
        hills = hill_calculator.get_hill_params()

        mean_response = ParetoUtils.calculate_fx_objective(
            x=row["mean_spend"],
            coeff=hills["coefs_sorted"][0],
            alpha=hills["alphas"][0],
            inflexion=hills["inflexions"][0],
            x_hist_carryover=mean_carryover,
            get_sum=False,
        )

        return pd.Series(
            {
                "mean_response": mean_response,
                "mean_spend_adstocked": mean_spend_adstocked,
                "mean_carryover": mean_carryover,
                "rn": row["rn"],
                "sol_id": get_solID,  # Consistently use sol_id
            }
        )

    def _generate_plot_data(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_data: ParetoData,
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for various plots used in the Pareto analysis.
        """
        mediaVecCollect = pd.DataFrame()
        xDecompVecCollect = pd.DataFrame()
        plotDataCollect = {}
        df_caov_pct_all = pd.DataFrame()

        xDecompAgg = pareto_data.x_decomp_agg
        dt_mod = self.featurized_mmm_data.dt_mod
        dt_modRollWind = self.featurized_mmm_data.dt_modRollWind
        rw_start_loc = self.mmm_data.mmmdata_spec.rolling_window_start_which
        rw_end_loc = self.mmm_data.mmmdata_spec.rolling_window_end_which

        # Clean column names to ensure consistency
        xDecompAgg = self._clean_column_names(xDecompAgg)

        # Debug print
        print("\nColumns in xDecompAgg:", xDecompAgg.columns.tolist())
        print("\nShape of xDecompAgg:", xDecompAgg.shape)

        # Assuming pareto_fronts_vec is derived from pareto_data
        pareto_fronts_vec = pareto_data.pareto_fronts

        print(f"\nProcessing Pareto fronts: {pareto_fronts_vec}")

        for pf in pareto_fronts_vec:
            plotMediaShare = xDecompAgg[
                (xDecompAgg["robynPareto"] == pf)
                & (xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends))
            ]

            # Debug print
            print(f"\nShape of plotMediaShare for front {pf}:", plotMediaShare.shape)

            uniqueSol = plotMediaShare["sol_id"].unique()  # Changed from solID to sol_id
            print(f"Unique solutions found for front {pf}:", uniqueSol)

            plotWaterfall = xDecompAgg[xDecompAgg["robynPareto"] == pf]
            print(f">> Pareto-Front: {pf} [{len(uniqueSol)} models]")

            for sid in tqdm(uniqueSol, desc="Processing Solutions", unit="solution"):
                # Get data for the current solution
                current_solution = plotMediaShare[plotMediaShare["sol_id"] == sid]

                # Create empty dictionaries for plot data
                plot1data = {
                    "plotMediaShareLoopBar": pd.DataFrame(),
                    "plotMediaShareLoopLine": pd.DataFrame(),
                    "ySecScale": 1.0,
                }
                plot2data = {"plotWaterfallLoop": pd.DataFrame()}
                plot3data = {"dt_geometric": None, "weibullCollect": None, "wb_type": self.hyper_parameter.adstock}
                plot4data = {"dt_scurvePlot": pd.DataFrame(), "dt_scurvePlotMean": pd.DataFrame()}
                plot5data = {
                    "xDecompVecPlotMelted": pd.DataFrame(),
                    "rsq": current_solution["rsq_train"].iloc[0] if not current_solution.empty else 0,
                }
                plot6data = {"xDecompVecPlot": pd.DataFrame()}
                plot7data = pd.DataFrame()

                # Add model data to mediaVecCollect
                if not dt_mod.empty and "ds" in dt_mod.columns:
                    media_data = dt_mod[["ds"] + self.mmm_data.mmmdata_spec.all_media].copy()
                    media_data["sol_id"] = sid
                    media_data["type"] = "rawMedia"
                    mediaVecCollect = pd.concat([mediaVecCollect, media_data], ignore_index=True)

                # Store plot data for this solution
                plotDataCollect[sid] = {
                    "plot1data": plot1data,
                    "plot2data": plot2data,
                    "plot3data": plot3data,
                    "plot4data": plot4data,
                    "plot5data": plot5data,
                    "plot6data": plot6data,
                    "plot7data": plot7data,
                }

        pareto_solutions = set()
        if "sol_id" in mediaVecCollect.columns:
            pareto_solutions.update(mediaVecCollect["sol_id"].unique())

        print("\nNumber of Pareto solutions found:", len(pareto_solutions))

        return {
            "pareto_solutions": pareto_solutions,
            "mediaVecCollect": mediaVecCollect,
            "xDecompVecCollect": xDecompVecCollect,
            "plotDataCollect": plotDataCollect,
            "df_caov_pct_all": df_caov_pct_all,
        }

    def robyn_immcarr(
        self,
        pareto_data: ParetoData,
        result_hyp_param: pd.DataFrame,
        solID=None,
        start_date=None,
        end_date=None,
    ):
        # Define default values when not provided
        if solID is None:
            solID = result_hyp_param["solID"].iloc[0]
        if start_date is None:
            start_date = self.mmm_data.mmmdata_spec.window_start
        if end_date is None:
            end_date = self.mmm_data.mmmdata_spec.window_end

        # Assuming dt_modRollWind is a DataFrame with a 'ds' column
        dt_modRollWind = pd.to_datetime(self.featurized_mmm_data.dt_modRollWind["ds"])
        dt_modRollWind = dt_modRollWind.dropna()

        # Check if start_date is a single value
        if isinstance(start_date, (list, pd.Series)):
            start_date = start_date[0]

        # Find the closest start_date
        start_date_closest = dt_modRollWind.iloc[(dt_modRollWind - pd.to_datetime(start_date)).abs().idxmin()]

        # Check if end_date is a single value
        if isinstance(end_date, (list, pd.Series)):
            end_date = end_date[0]  # Take the first element if it's a list or Series

        # Find the closest end_date
        end_date_closest = dt_modRollWind.iloc[(dt_modRollWind - pd.to_datetime(end_date)).abs().idxmin()]

        # Filter for custom window
        rollingWindowStartWhich = dt_modRollWind[dt_modRollWind == start_date].index[0]
        rollingWindowEndWhich = dt_modRollWind[dt_modRollWind == end_date].index[0]
        rollingWindow = range(rollingWindowStartWhich, rollingWindowEndWhich + 1)

        # Calculate saturated dataframes with carryover and immediate parts
        hypParamSam = result_hyp_param[result_hyp_param["solID"] == solID]
        hyperparameter = self._extract_hyperparameter(hypParamSam)

        dt_saturated_dfs = self.transformer.run_transformations(
            self.featurized_mmm_data,
            hyperparameter,
            hyperparameter.adstock,
        )

        # Calculate decomposition
        coefs = pareto_data.x_decomp_agg.loc[pareto_data.x_decomp_agg["solID"] == solID, "coef"].values
        coefs_names = pareto_data.x_decomp_agg.loc[pareto_data.x_decomp_agg["solID"] == solID, "rn"].values

        # Create a DataFrame to hold coefficients and their names
        coefs_df = pd.DataFrame({"name": coefs_names, "coefficient": coefs})

        decompCollect = self._model_decomp(
            inputs={
                "coefs": coefs_df,
                "y_pred": dt_saturated_dfs.dt_modSaturated["dep_var"].iloc[rollingWindow],
                "dt_modSaturated": dt_saturated_dfs.dt_modSaturated.iloc[rollingWindow],
                "dt_saturatedImmediate": dt_saturated_dfs.dt_saturatedImmediate.iloc[rollingWindow],
                "dt_saturatedCarryover": dt_saturated_dfs.dt_saturatedCarryover.iloc[rollingWindow],
                "dt_modRollWind": self.featurized_mmm_data.dt_modRollWind.iloc[rollingWindow],
                "refreshAddedStart": start_date,
            }
        )

        # Media decomposition
        mediaDecompImmediate = decompCollect["mediaDecompImmediate"].drop(columns=["ds", "y"], errors="ignore")
        mediaDecompImmediate.columns = [f"{col}_MDI" for col in mediaDecompImmediate.columns]

        mediaDecompCarryover = decompCollect["mediaDecompCarryover"].drop(columns=["ds", "y"], errors="ignore")
        mediaDecompCarryover.columns = [f"{col}_MDC" for col in mediaDecompCarryover.columns]

        # Combine results
        temp = pd.concat(
            [decompCollect["xDecompVec"], mediaDecompImmediate, mediaDecompCarryover],
            axis=1,
        )
        temp["solID"] = solID

        # Create vector collections
        vec_collect = {
            "xDecompVec": temp.drop(
                columns=temp.columns[temp.columns.str.endswith("_MDI") | temp.columns.str.endswith("_MDC")]
            ),
            "xDecompVecImmediate": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDC") | temp.columns.isin(self.mmm_data.mmmdata_spec.all_media)
                ]
            ),
            "xDecompVecCarryover": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDI") | temp.columns.isin(self.mmm_data.mmmdata_spec.all_media)
                ]
            ),
        }

        # Rename columns
        this = vec_collect["xDecompVecImmediate"].columns.str.replace("_MDI", "", regex=False)
        vec_collect["xDecompVecImmediate"].columns = this
        vec_collect["xDecompVecCarryover"].columns = this

        # Calculate carryover percentages
        df_caov = (vec_collect["xDecompVecCarryover"].groupby("solID").sum().reset_index()).drop(columns="ds")
        df_total = vec_collect["xDecompVec"].groupby("solID").sum().reset_index().drop(columns="ds")

        df_caov_pct = df_caov.copy()
        df_caov_pct.loc[:, df_caov_pct.columns[1:]] = df_caov_pct.loc[:, df_caov_pct.columns[1:]].div(
            df_total.iloc[:, 1:].values
        )
        df_caov_pct = df_caov_pct.melt(id_vars="solID", var_name="rn", value_name="carryover_pct").fillna(0)

        # Gather everything in an aggregated format
        xDecompVecImmeCaov = (
            pd.concat(
                [
                    vec_collect["xDecompVecImmediate"].assign(type="Immediate"),
                    vec_collect["xDecompVecCarryover"].assign(type="Carryover"),
                ],
                axis=0,
            )
            .melt(
                id_vars=["solID", "type"],
                value_vars=self.mmm_data.mmmdata_spec.all_media,
                var_name="rn",
                value_name="value",
            )
            .assign(start_date=start_date, end_date=end_date)
        )

        # Grouping and aggregating the data
        xDecompVecImmeCaov = (
            xDecompVecImmeCaov.groupby(["solID", "start_date", "end_date", "rn", "type"])
            .agg(response=("value", "sum"))
            .reset_index()
        )

        xDecompVecImmeCaov["percentage"] = xDecompVecImmeCaov["response"] / xDecompVecImmeCaov.groupby(
            ["solID", "start_date", "end_date", "type"]
        )["response"].transform("sum")
        xDecompVecImmeCaov.fillna(0, inplace=True)

        # Join with carryover percentages
        xDecompVecImmeCaov = xDecompVecImmeCaov.merge(df_caov_pct, on=["solID", "rn"], how="left")

        return xDecompVecImmeCaov

    def _extract_hyperparameter(self, hypParamSam: pd.DataFrame) -> Hyperparameters:
        """
        This function extracts hyperparameters from a given DataFrame.

        Parameters:
        hypParamSam (DataFrame): A DataFrame containing hyperparameters.

        Returns:
        hyperparameter (dict): A dictionary of hyperparameters.
        """
        channelHyperparams: dict[str, ChannelHyperparameters] = {}
        for med in self.mmm_data.mmmdata_spec.all_media:
            alphas = hypParamSam[f"{med}_alphas"].values
            gammas = hypParamSam[f"{med}_gammas"].values
            if self.hyper_parameter.adstock == AdstockType.GEOMETRIC:
                thetas = hypParamSam[f"{med}_thetas"].values
                channelHyperparams[med] = ChannelHyperparameters(
                    thetas=thetas,
                    alphas=alphas,
                    gammas=gammas,
                )
            elif self.hyper_parameter.adstock in [
                AdstockType.WEIBULL_CDF,
                AdstockType.WEIBULL_PDF,
            ]:
                shapes = hypParamSam[f"{med}_shapes"].values
                scales = hypParamSam[f"{med}_scales"].values
                channelHyperparams[med] = ChannelHyperparameters(
                    shapes=shapes,
                    scales=scales,
                    alphas=alphas,
                    gammas=gammas,
                )

        return Hyperparameters(adstock=self.hyper_parameter.adstock, hyperparameters=channelHyperparams)

    def _model_decomp(self, inputs) -> Dict[str, pd.DataFrame]:
        # Extracting inputs from the dictionary
        coefs = inputs["coefs"]
        y_pred = inputs["y_pred"]
        dt_modSaturated = inputs["dt_modSaturated"]
        dt_saturatedImmediate = inputs["dt_saturatedImmediate"]
        dt_saturatedCarryover = inputs["dt_saturatedCarryover"]
        dt_modRollWind = inputs["dt_modRollWind"]
        refreshAddedStart = inputs["refreshAddedStart"]

        # Input for decomp
        y = dt_modSaturated["dep_var"]

        # Select all columns except 'dep_var'
        x = dt_modSaturated.drop(columns=["dep_var"])
        intercept = coefs["coefficient"].iloc[0]  # Assuming the first row contains the intercept
        x_name = x.columns
        x_factor = x_name[x.dtypes == "category"]  # Assuming factors are categorical

        # Decomp x
        # Create an empty DataFrame for xDecomp
        xDecomp = pd.DataFrame()

        # Multiply each regressor by its corresponding coefficient
        for name in x.columns:
            # Get the corresponding coefficient for the regressor
            coefficient_value = coefs.loc[coefs["name"] == name, "coefficient"].values
            xDecomp[name] = x[name] * (coefficient_value if len(coefficient_value) > 0 else 0)

        # Add intercept as the first column
        xDecomp.insert(0, "intercept", intercept)  # Assuming intercept is defined

        xDecompOut = pd.concat(
            [
                pd.DataFrame({"ds": dt_modRollWind["ds"], "y": y, "y_pred": y_pred}),
                xDecomp,
            ],
            axis=1,
        )

        # Decomp immediate & carryover response
        sel_coef = coefs["name"].isin(
            dt_saturatedImmediate.columns
        )  # Check if coefficient names are in the immediate DataFrame
        coefs_media = coefs[sel_coef].set_index("name")["coefficient"]  # Set names for coefs_media

        mediaDecompImmediate = pd.DataFrame(
            {name: dt_saturatedImmediate[name] * coefs_media[name] for name in coefs_media.index}
        )
        mediaDecompCarryover = pd.DataFrame(
            {name: dt_saturatedCarryover[name] * coefs_media[name] for name in coefs_media.index}
        )

        return {
            "xDecompVec": xDecompOut,
            "mediaDecompImmediate": mediaDecompImmediate,
            "mediaDecompCarryover": mediaDecompCarryover,
        }
