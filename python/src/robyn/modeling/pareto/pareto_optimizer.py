# pyre-strict

from concurrent.futures import as_completed, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.holidays_data import HolidaysData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.pareto.hill_calculator import HillCalculator
from robyn.modeling.pareto.immediate_carryover import ImmediateCarryoverCalculator
from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.modeling.pareto.response_curve import ResponseCurveCalculator, ResponseOutput
from robyn.modeling.transformations.transformations import Transformation
from robyn.modeling.entities.pareto_result import ParetoResult
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
    """

    def __init__(
        self,
        mmm_data: MMMData,
        model_outputs: ModelOutputs,
        hyper_parameter: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
        holidays_data: HolidaysData,
    ):
        """Initialize the ParetoOptimizer."""
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        self.hyper_parameter = hyper_parameter
        self.featurized_mmm_data = featurized_mmm_data
        self.holidays_data = holidays_data
        self.transformer = Transformation(mmm_data)

    def _ensure_trial_ids(self, trial: Trial) -> Trial:
        """
        Ensure trial has proper sol_id and all its dataframes have sol_id column.

        Args:
            trial: Trial object to process

        Returns:
            Processed trial with consistent sol_id
        """
        # Generate sol_id if not present
        if not trial.sol_id:
            trial.sol_id = f"{trial.trial}_{trial.iter_ng}_{trial.iter_par}"

        # Ensure result_hyp_param has sol_id
        if isinstance(trial.result_hyp_param, pd.DataFrame):
            if "sol_id" not in trial.result_hyp_param.columns:
                trial.result_hyp_param["sol_id"] = trial.sol_id

        # Ensure x_decomp_agg has sol_id
        if isinstance(trial.x_decomp_agg, pd.DataFrame):
            if "sol_id" not in trial.x_decomp_agg.columns:
                trial.x_decomp_agg["sol_id"] = trial.sol_id

        # Ensure decomp_spend_dist has sol_id if it exists
        if isinstance(trial.decomp_spend_dist, pd.DataFrame):
            if "sol_id" not in trial.decomp_spend_dist.columns:
                trial.decomp_spend_dist["sol_id"] = trial.sol_id

        return trial

    def _validate_model_outputs(self) -> None:
        """
        Validate model outputs data structure and ensure required fields are present.

        Raises:
            ValueError: If required data is missing or malformed
        """
        if not self.model_outputs.trials:
            raise ValueError("No trials found in model outputs")

        for trial in self.model_outputs.trials:
            if not isinstance(trial.result_hyp_param, pd.DataFrame):
                raise ValueError(f"Trial {trial.sol_id} has invalid result_hyp_param")
            if not isinstance(trial.x_decomp_agg, pd.DataFrame):
                raise ValueError(f"Trial {trial.sol_id} has invalid x_decomp_agg")

    def _aggregate_model_data(self, calibrated: bool) -> Dict[str, pd.DataFrame]:
        """
        Aggregate and prepare data from model outputs for Pareto optimization.

        Args:
            calibrated: Whether the models have undergone calibration

        Returns:
            Dictionary containing aggregated data
        """
        # Validate and ensure proper data structure
        self._validate_model_outputs()

        # Process all trials to ensure proper sol_id
        self.model_outputs.trials = [self._ensure_trial_ids(trial) for trial in self.model_outputs.trials]

        hyper_fixed = self.model_outputs.hyper_fixed
        trials = [model for model in self.model_outputs.trials if hasattr(model, "resultCollect")]

        # Create DataFrames with guaranteed sol_id column
        resultHypParam_list = [trial.result_hyp_param for trial in self.model_outputs.trials]
        xDecompAgg_list = [trial.x_decomp_agg for trial in self.model_outputs.trials]

        resultHypParam = pd.concat(resultHypParam_list, ignore_index=True)
        xDecompAgg = pd.concat(xDecompAgg_list, ignore_index=True)

        # Verify sol_id is present
        if "sol_id" not in resultHypParam.columns:
            raise ValueError("sol_id missing from resultHypParam after aggregation")
        if "sol_id" not in xDecompAgg.columns:
            raise ValueError("sol_id missing from xDecompAgg after aggregation")

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

        # Handle bootstrap results
        if len(xDecompAgg["sol_id"].unique()) == 1 and "boot_mean" not in xDecompAgg.columns:
            bootstrap = getattr(self.model_outputs, "bootstrap", None)
            if bootstrap is not None:
                xDecompAgg = pd.merge(xDecompAgg, bootstrap, left_on="rn", right_on="variable", how="left")

        return {
            "result_hyp_param": resultHypParam,
            "x_decomp_agg": xDecompAgg,
            "result_calibration": resultCalibration,
        }

    def optimize(
        self,
        pareto_fronts: str = "auto",
        min_candidates: int = 100,
        calibration_constraint: float = 0.1,
        calibrated: bool = False,
    ) -> ParetoResult:
        """
        Perform Pareto optimization on the model results.

        Args:
            pareto_fronts: Number of Pareto fronts to consider or "auto"
            min_candidates: Minimum number of candidates for auto selection
            calibration_constraint: Constraint for calibration
            calibrated: Whether models are calibrated

        Returns:
            ParetoResult object containing optimization results

        Raises:
            ValueError: If data validation fails
        """
        try:
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
        except Exception as e:
            print(f"Error during Pareto optimization: {str(e)}")
            print("\nDebugging information:")
            print(f"Number of trials: {len(self.model_outputs.trials)}")
            if self.model_outputs.trials:
                trial = self.model_outputs.trials[0]
                print(f"\nFirst trial sol_id: {trial.sol_id}")
                print("\nColumns in result_hyp_param:")
                print(trial.result_hyp_param.columns.tolist())
                print("\nColumns in x_decomp_agg:")
                print(trial.x_decomp_agg.columns.tolist())
            raise

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
                .groupby("sol_id")["coef"]
                .apply(lambda x: min(x.dropna()) == 0)
            )
            # calculate quantiles
            mape_lift_quantile10 = resultHypParam["mape"].quantile(calibration_constraint)
            nrmse_quantile90 = resultHypParam["nrmse"].quantile(0.9)
            decomprssd_quantile90 = resultHypParam["decomp.rssd"].quantile(0.9)
            # merge resultHypParam with xDecompAggCoef0
            resultHypParam = pd.merge(resultHypParam, xDecompAggCoef0, on="sol_id", how="left")
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
                ["sol_id", "robynPareto"]
            ]
            resultHypParamPareto = resultHypParamPareto.groupby("sol_id").first().reset_index()
            resultHypParam = pd.merge(resultHypParam, resultHypParamPareto, on="sol_id", how="left")
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
        pareto_fronts: str,
        min_candidates: int,
        calibrated: bool,
    ) -> ParetoData:
        """
        Prepare Pareto optimization data with adaptive candidate sizing.
        """
        result_hyp_param = aggregated_data["result_hyp_param"]
        total_models = len(result_hyp_param)

        # Scale min_candidates based on total models available
        scaled_min_candidates = min(
            min_candidates, max(5, int(total_models * 0.1))  # At least 5 candidates, maximum 10% of total models
        )

        print(f"\n>> Total models available: {total_models}")
        print(f">> Using scaled minimum candidates: {scaled_min_candidates}")

        # 1. Binding Pareto results
        aggregated_data["x_decomp_agg"] = pd.merge(
            aggregated_data["x_decomp_agg"],
            result_hyp_param[["robynPareto", "sol_id"]],
            on="sol_id",
            how="left",
        )

        # 2. Collect decomp_spend_dist with progress tracking
        print(">> Processing model decompositions...")
        decomp_spend_dist_list = []
        for trial in tqdm(self.model_outputs.trials, desc="Processing trials"):
            if trial.decomp_spend_dist is not None:
                decomp_spend_dist_list.append(trial.decomp_spend_dist)

        decomp_spend_dist = pd.concat(decomp_spend_dist_list, ignore_index=True)

        # Add sol_id if hyper_fixed is False
        if not self.model_outputs.hyper_fixed:
            decomp_spend_dist["sol_id"] = (
                decomp_spend_dist["trial"].astype(str)
                + "_"
                + decomp_spend_dist["iterNG"].astype(str)
                + "_"
                + decomp_spend_dist["iterPar"].astype(str)
            )

        # Left join with resultHypParam
        decomp_spend_dist = pd.merge(
            decomp_spend_dist,
            result_hyp_param[["robynPareto", "sol_id"]],
            on="sol_id",
            how="left",
        )

        # 3. Determining the number of Pareto fronts
        if self.model_outputs.hyper_fixed or len(result_hyp_param) == 1:
            pareto_fronts = 1
            print(">> Using single Pareto front due to fixed hyperparameters or single model")

        # 4. Handling automatic Pareto front selection
        if pareto_fronts == "auto":
            n_pareto = result_hyp_param["robynPareto"].notna().sum()
            print(f">> Number of Pareto-optimal solutions found: {n_pareto}")

            if n_pareto <= scaled_min_candidates and len(result_hyp_param) > 1 and not calibrated:
                # Instead of raising error, adjust the minimum candidates
                print(f"Warning: Found only {n_pareto} Pareto-optimal solutions.")
                print(f"Adjusting minimum candidates from {scaled_min_candidates} to {max(1, n_pareto)}")
                scaled_min_candidates = max(1, n_pareto)

            # Group by 'robynPareto' and count distinct 'sol_id'
            grouped_data = (
                result_hyp_param[result_hyp_param["robynPareto"].notna()]
                .groupby("robynPareto", as_index=False)
                .agg(n=("sol_id", "nunique"))
            )

            # Calculate cumulative sum
            grouped_data["n_cum"] = grouped_data["n"].cumsum()

            # Filter for candidates meeting minimum threshold
            auto_pareto = grouped_data[grouped_data["n_cum"] >= scaled_min_candidates]

            if len(auto_pareto) == 0:
                # Use all available Pareto fronts instead of raising error
                auto_pareto = grouped_data.iloc[[-1]]
                print(
                    f"Warning: Using all available Pareto fronts ({len(grouped_data)}) with {int(auto_pareto['n_cum'].iloc[0])} total candidates"
                )
            else:
                auto_pareto = auto_pareto.iloc[0]
                print(
                    f">> Selected {int(auto_pareto['robynPareto'])} Pareto-fronts ",
                    f"containing {int(auto_pareto['n_cum'])} candidates",
                )

            pareto_fronts = int(auto_pareto["robynPareto"])

        # 5. Creating Pareto front vector
        pareto_fronts_vec = list(range(1, pareto_fronts + 1))

        # 6. Filtering data for selected Pareto fronts
        print(">> Filtering data for selected Pareto fronts...")
        decomp_spend_dist_pareto = decomp_spend_dist[decomp_spend_dist["robynPareto"].isin(pareto_fronts_vec)]
        result_hyp_param_pareto = result_hyp_param[result_hyp_param["robynPareto"].isin(pareto_fronts_vec)]
        x_decomp_agg_pareto = aggregated_data["x_decomp_agg"][
            aggregated_data["x_decomp_agg"]["robynPareto"].isin(pareto_fronts_vec)
        ]

        print(f">> Final number of models selected: {len(result_hyp_param_pareto)}")

        return ParetoData(
            decomp_spend_dist=decomp_spend_dist_pareto,
            result_hyp_param=result_hyp_param_pareto,
            x_decomp_agg=x_decomp_agg_pareto,
            pareto_fronts=pareto_fronts_vec,
        )

    def run_dt_resp(self, row: pd.Series, paretoData: ParetoData) -> Optional[dict]:
        """
        Calculate response curves for a given row of Pareto data.
        Added error handling and validation.

        Args:
            row: A row of Pareto data
            paretoData: Pareto data object

        Returns:
            Dictionary with response curve calculations or None if calculation fails
        """
        try:
            get_sol_id = row["sol_id"]
            get_spendname = row["rn"]
            startRW = self.mmm_data.mmmdata_spec.rolling_window_start_which
            endRW = self.mmm_data.mmmdata_spec.rolling_window_end_which

            response_calculator = ResponseCurveCalculator(
                mmm_data=self.mmm_data,
                model_outputs=self.model_outputs,
                hyperparameter=self.hyper_parameter,
            )

            response_output: ResponseOutput = response_calculator.calculate_response(
                select_model=get_sol_id,
                metric_name=get_spendname,
                date_range="all",
                dt_hyppar=paretoData.result_hyp_param,
                dt_coef=paretoData.x_decomp_agg,
                quiet=True,
            )

            mean_spend_adstocked = np.mean(response_output.input_total[startRW:endRW])
            mean_carryover = np.mean(response_output.input_carryover[startRW:endRW])

            dt_hyppar = paretoData.result_hyp_param[paretoData.result_hyp_param["sol_id"] == get_sol_id]
            chn_adstocked = pd.DataFrame({get_spendname: response_output.input_total[startRW:endRW]})
            dt_coef = paretoData.x_decomp_agg[
                (paretoData.x_decomp_agg["sol_id"] == get_sol_id) & (paretoData.x_decomp_agg["rn"] == get_spendname)
            ][["rn", "coef"]]

            hill_calculator = HillCalculator(
                mmmdata=self.mmm_data,
                model_outputs=self.model_outputs,
                dt_hyppar=dt_hyppar,
                dt_coef=dt_coef,
                media_spend_sorted=[get_spendname],
                select_model=get_sol_id,
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

            return {
                "mean_response": mean_response,
                "mean_spend_adstocked": mean_spend_adstocked,
                "mean_carryover": mean_carryover,
                "rn": row["rn"],
                "sol_id": row["sol_id"],
            }
        except Exception as e:
            print(
                f"Error processing row for sol_id {row.get('sol_id', 'unknown')}, rn {row.get('rn', 'unknown')}: {str(e)}"
            )
            return None

    def _compute_response_curves(
        self, pareto_data: ParetoData, aggregated_data: Dict[str, pd.DataFrame]
    ) -> ParetoData:
        """
        Calculate response curves with improved error handling and validation.
        """
        if pareto_data.decomp_spend_dist.empty:
            print("Warning: No data in decomp_spend_dist. Skipping response curves calculation.")
            return pareto_data

        print(f"\n>>> Calculating response curves for {len(pareto_data.decomp_spend_dist)} models' media variables...")
        print(f"Available columns: {pareto_data.decomp_spend_dist.columns.tolist()}")

        # Validate required columns
        required_columns = ["rn", "sol_id", "mean_spend", "total_spend"]
        missing_columns = [col for col in required_columns if col not in pareto_data.decomp_spend_dist.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in decomp_spend_dist: {missing_columns}")

        # Process in batches for better memory management
        batch_size = min(100, len(pareto_data.decomp_spend_dist))
        resp_collect_list = []

        try:
            for i in range(0, len(pareto_data.decomp_spend_dist), batch_size):
                batch = pareto_data.decomp_spend_dist.iloc[i : i + batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}, size: {len(batch)}")

                if self.model_outputs.cores > 1 and len(batch) > 1:
                    with ProcessPoolExecutor(max_workers=self.model_outputs.cores) as executor:
                        run_dt_resp_partial = partial(self.run_dt_resp, paretoData=pareto_data)
                        futures = []
                        for _, row in batch.iterrows():
                            try:
                                futures.append(executor.submit(run_dt_resp_partial, row))
                            except Exception as e:
                                print(f"Error submitting row to executor: {str(e)}")
                                continue

                        if not futures:
                            print("Warning: No futures created for this batch")
                            continue

                        results = []
                        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
                            try:
                                result = future.result()
                                if result is not None:
                                    results.append(result)
                            except Exception as e:
                                print(f"Error processing future: {str(e)}")
                                continue

                        if results:
                            resp_collect_batch = pd.DataFrame(results)
                            resp_collect_list.append(resp_collect_batch)
                else:
                    # Serial processing for small batches or single core
                    results = []
                    for _, row in tqdm(batch.iterrows(), total=len(batch), desc="Processing rows"):
                        try:
                            result = self.run_dt_resp(row, paretoData=pareto_data)
                            if result is not None:
                                results.append(result)
                        except Exception as e:
                            print(f"Error processing row: {str(e)}")
                            continue

                    if results:
                        resp_collect_batch = pd.DataFrame(results)
                        resp_collect_list.append(resp_collect_batch)

            if not resp_collect_list:
                print("Warning: No response curves were calculated successfully")
                return pareto_data

            # Combine results
            resp_collect = pd.concat(resp_collect_list, ignore_index=True)
            print(f"\nSuccessfully processed {len(resp_collect)} response curves")

            # Merge results and calculate metrics
            print(">> Computing final metrics...")
            pareto_data.decomp_spend_dist = pd.merge(
                pareto_data.decomp_spend_dist, resp_collect, on=["sol_id", "rn"], how="left"
            )

            # Calculate ROI and CPA metrics
            print(">> Calculating ROI and CPA metrics...")
            metrics_df = pd.DataFrame()
            try:
                metrics_df = pareto_data.decomp_spend_dist.assign(
                    roi_mean=lambda x: x["mean_response"] / x["mean_spend"],
                    roi_total=lambda x: x["xDecompAgg"] / x["total_spend"],
                    cpa_mean=lambda x: x["mean_spend"] / x["mean_response"],
                    cpa_total=lambda x: x["total_spend"] / x["xDecompAgg"],
                )
            except Exception as e:
                print(f"Warning: Error calculating metrics: {str(e)}")
                print("Available columns:", pareto_data.decomp_spend_dist.columns.tolist())

            if not metrics_df.empty:
                pareto_data.decomp_spend_dist = metrics_df

            return pareto_data

        except Exception as e:
            print(f"Error in response curves calculation: {str(e)}")
            print("\nDebugging information:")
            print(f"decomp_spend_dist shape: {pareto_data.decomp_spend_dist.shape}")
            print(f"decomp_spend_dist columns: {pareto_data.decomp_spend_dist.columns.tolist()}")
            print(f"Number of response collect batches: {len(resp_collect_list)}")
            raise

    def _generate_plot_data(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_data: ParetoData,
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for various plots with metric calculation.
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

        print("\nStarting plot data generation...")
        print(f"Available columns in xDecompAgg: {xDecompAgg.columns.tolist()}")

        # Get Pareto fronts vector
        pareto_fronts_vec = pareto_data.pareto_fronts

        for pf in pareto_fronts_vec:
            print(f"\nProcessing Pareto front {pf}")

            # Filter media share data
            plotMediaShare = xDecompAgg[
                (xDecompAgg["robynPareto"] == pf)
                & (xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends))
            ].copy()  # Create a copy to avoid SettingWithCopyWarning

            print(f"Shape of plotMediaShare: {plotMediaShare.shape}")

            # Calculate necessary metrics
            try:
                # Calculate media total spends
                if "mean_spend" in plotMediaShare.columns:
                    window_length = rw_end_loc - rw_start_loc
                    plotMediaShare["total_spend"] = plotMediaShare["mean_spend"] * window_length
                elif "xDecompAgg" in plotMediaShare.columns:
                    # Fallback: estimate from decomposition
                    plotMediaShare["total_spend"] = plotMediaShare["xDecompAgg"]

                # Calculate spend shares
                spend_sums = plotMediaShare.groupby("sol_id")["total_spend"].transform("sum")
                plotMediaShare["spend_share"] = plotMediaShare["total_spend"] / spend_sums

                # Calculate effect shares
                effect_sums = plotMediaShare.groupby("sol_id")["xDecompAgg"].transform("sum")
                plotMediaShare["effect_share"] = plotMediaShare["xDecompAgg"] / effect_sums

                # Calculate ROI and CPA
                plotMediaShare["roi_total"] = plotMediaShare["xDecompAgg"] / plotMediaShare["total_spend"]
                plotMediaShare["cpa_total"] = plotMediaShare["total_spend"] / plotMediaShare["xDecompAgg"]

                print("Successfully calculated all required metrics")

            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                print("Available columns:", plotMediaShare.columns.tolist())
                continue

            uniqueSol = plotMediaShare["sol_id"].unique()
            plotWaterfall = xDecompAgg[xDecompAgg["robynPareto"] == pf]

            print(f">> Pareto-Front: {pf} [{len(uniqueSol)} models]")

            for sid in tqdm(uniqueSol, desc="Processing Solutions", unit="solution"):
                try:
                    # 1. Spend x effect share comparison
                    temp = plotMediaShare[plotMediaShare["sol_id"] == sid]

                    # Select available columns for id_vars
                    id_columns = ["rn"]
                    for col in ["nrmse", "decomp.rssd", "rsq_train"]:
                        if col in temp.columns:
                            id_columns.append(col)

                    temp = temp.melt(
                        id_vars=id_columns,
                        value_vars=[
                            "spend_share",
                            "effect_share",
                            "roi_total",
                            "cpa_total",
                        ],
                        var_name="variable",
                        value_name="value",
                    )

                    temp["rn"] = pd.Categorical(
                        temp["rn"],
                        categories=sorted(self.mmm_data.mmmdata_spec.paid_media_spends),
                        ordered=True,
                    )

                    plotMediaShareLoopBar = temp[temp["variable"].isin(["spend_share", "effect_share"])]
                    metric_type = (
                        "cpa_total" if self.mmm_data.mmmdata_spec.dep_var_type == "conversion" else "roi_total"
                    )
                    plotMediaShareLoopLine = temp[temp["variable"] == metric_type]

                    # Calculate scale while handling infinite values
                    valid_line_values = plotMediaShareLoopLine["value"][~np.isinf(plotMediaShareLoopLine["value"])]
                    valid_bar_values = plotMediaShareLoopBar["value"][~np.isinf(plotMediaShareLoopBar["value"])]

                    if not valid_line_values.empty and not valid_bar_values.empty:
                        ySecScale = max(valid_line_values) / max(valid_bar_values) * 1.1
                    else:
                        ySecScale = 1.0

                    plot1data = {
                        "plotMediaShareLoopBar": plotMediaShareLoopBar,
                        "plotMediaShareLoopLine": plotMediaShareLoopLine,
                        "ySecScale": ySecScale,
                    }

                    # 2. Waterfall plot data
                    if "xDecompPerc" not in plotWaterfall.columns and "xDecompAgg" in plotWaterfall.columns:
                        total_decomp = plotWaterfall.groupby("sol_id")["xDecompAgg"].transform("sum")
                        plotWaterfall["xDecompPerc"] = plotWaterfall["xDecompAgg"] / total_decomp

                    plotWaterfallLoop = (
                        plotWaterfall[plotWaterfall["sol_id"] == sid]
                        .sort_values("xDecompPerc")
                        .assign(
                            end=lambda x: 1 - x["xDecompPerc"].cumsum(),
                            start=lambda x: x["end"].shift(1).fillna(1),
                            id=range(1, len(plotWaterfall[plotWaterfall["sol_id"] == sid]) + 1),
                            rn=lambda x: pd.Categorical(x["rn"]),
                            sign=lambda x: pd.Categorical(np.where(x["xDecompPerc"] >= 0, "Positive", "Negative")),
                        )
                    )

                    plot2data = {"plotWaterfallLoop": plotWaterfallLoop}

                    # Store results
                    plotDataCollect[sid] = {
                        "plot1data": plot1data,
                        "plot2data": plot2data,
                    }

                except Exception as e:
                    print(f"Error processing solution {sid}: {str(e)}")
                    continue

        return {
            "pareto_solutions": set(plotDataCollect.keys()),
            "mediaVecCollect": mediaVecCollect,
            "xDecompVecCollect": xDecompVecCollect,
            "plotDataCollect": plotDataCollect,
            "df_caov_pct_all": df_caov_pct_all,
        }

    def robyn_immcarr(
        self,
        pareto_data: ParetoData,
        result_hyp_param: pd.DataFrame,
        sol_id=None,
        start_date=None,
        end_date=None,
    ):
        # Define default values when not provided
        if sol_id is None:
            sol_id = result_hyp_param["sol_id"].iloc[0]
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
        hypParamSam = result_hyp_param[result_hyp_param["sol_id"] == sol_id]
        hyperparameter = self._extract_hyperparameter(hypParamSam)

        dt_saturated_dfs = self.transformer.run_transformations(
            self.featurized_mmm_data,
            hyperparameter,
            hyperparameter.adstock,
        )

        # Calculate decomposition
        coefs = pareto_data.x_decomp_agg.loc[pareto_data.x_decomp_agg["sol_id"] == sol_id, "coef"].values
        coefs_names = pareto_data.x_decomp_agg.loc[pareto_data.x_decomp_agg["sol_id"] == sol_id, "rn"].values

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
        temp["sol_id"] = sol_id

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
        df_caov = (vec_collect["xDecompVecCarryover"].groupby("sol_id").sum().reset_index()).drop(columns="ds")
        df_total = vec_collect["xDecompVec"].groupby("sol_id").sum().reset_index().drop(columns="ds")

        df_caov_pct = df_caov.copy()
        df_caov_pct.loc[:, df_caov_pct.columns[1:]] = df_caov_pct.loc[:, df_caov_pct.columns[1:]].div(
            df_total.iloc[:, 1:].values
        )
        df_caov_pct = df_caov_pct.melt(id_vars="sol_id", var_name="rn", value_name="carryover_pct").fillna(0)

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
                id_vars=["sol_id", "type"],
                value_vars=self.mmm_data.mmmdata_spec.all_media,
                var_name="rn",
                value_name="value",
            )
            .assign(start_date=start_date, end_date=end_date)
        )

        # Grouping and aggregating the data
        xDecompVecImmeCaov = (
            xDecompVecImmeCaov.groupby(["sol_id", "start_date", "end_date", "rn", "type"])
            .agg(response=("value", "sum"))
            .reset_index()
        )

        xDecompVecImmeCaov["percentage"] = xDecompVecImmeCaov["response"] / xDecompVecImmeCaov.groupby(
            ["sol_id", "start_date", "end_date", "type"]
        )["response"].transform("sum")
        xDecompVecImmeCaov.fillna(0, inplace=True)

        # Join with carryover percentages
        xDecompVecImmeCaov = xDecompVecImmeCaov.merge(df_caov_pct, on=["sol_id", "rn"], how="left")

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
