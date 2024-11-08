# pyre-strict

from concurrent.futures import as_completed, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional
import logging

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
from tqdm import tqdm


@dataclass
class ParetoResult:
    """
    Holds the results of Pareto optimization for marketing mix models.

    Attributes:
        pareto_solutions (List[str]): List of solution IDs that are Pareto-optimal.
        pareto_fronts (int): Number of Pareto fronts considered in the optimization.
        result_hyp_param (pd.DataFrame): Hyperparameters of Pareto-optimal solutions.
        x_decomp_agg (pd.DataFrame): Aggregated decomposition results for Pareto-optimal solutions.
        result_calibration (Optional[pd.DataFrame]): Calibration results, if calibration was performed.
        media_vec_collect (pd.DataFrame): Collected media vectors for all Pareto-optimal solutions.
        x_decomp_vec_collect (pd.DataFrame): Collected decomposition vectors for all Pareto-optimal solutions.
        plot_data_collect (Dict[str, pd.DataFrame]): Data for various plots, keyed by plot type.
        df_caov_pct_all (pd.DataFrame): Carryover percentage data for all channels and Pareto-optimal solutions.
    """

    pareto_solutions: List[str]
    pareto_fronts: int
    result_hyp_param: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    result_calibration: Optional[pd.DataFrame]
    media_vec_collect: pd.DataFrame
    x_decomp_vec_collect: pd.DataFrame
    plot_data_collect: Dict[str, pd.DataFrame]
    df_caov_pct_all: pd.DataFrame


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

        # Setup logger with a single handler
        self.logger = logging.getLogger("robyn.pareto_optimizer")
        # Remove any existing handlers to prevent duplicates
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        # Create a single handler with custom formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Prevent logger from propagating to root logger
        self.logger.propagate = False

    def _ensure_trial_ids(self, trial: Trial) -> Trial:
        """Ensure trial has proper sol_id and all its dataframes have sol_id column."""
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
        """Validate model outputs data structure and ensure required fields are present."""
        if not self.model_outputs.trials:
            raise ValueError("No trials found in model outputs")

        for trial in self.model_outputs.trials:
            if not isinstance(trial.result_hyp_param, pd.DataFrame):
                raise ValueError(f"Trial {trial.sol_id} has invalid result_hyp_param")
            if not isinstance(trial.x_decomp_agg, pd.DataFrame):
                raise ValueError(f"Trial {trial.sol_id} has invalid x_decomp_agg")

    def _aggregate_model_data(self, calibrated: bool) -> Dict[str, pd.DataFrame]:
        """Aggregate and prepare data from model outputs for Pareto optimization."""
        self._validate_model_outputs()
        self.logger.info("Starting model data aggregation")

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
            self.logger.info("Processing calibration data")
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
                self.logger.info("Merging bootstrap results")
                xDecompAgg = pd.merge(xDecompAgg, bootstrap, left_on="rn", right_on="variable", how="left")

        self.logger.info("Model data aggregation completed successfully")
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
        """Perform Pareto optimization on the model results."""
        try:
            self.logger.info("Starting Pareto optimization")
            aggregated_data = self._aggregate_model_data(calibrated)
            aggregated_data["result_hyp_param"] = self._compute_pareto_fronts(
                aggregated_data, pareto_fronts, min_candidates, calibration_constraint
            )

            pareto_data = self.prepare_pareto_data(aggregated_data, pareto_fronts, min_candidates, calibrated)
            pareto_data = self._compute_response_curves(pareto_data, aggregated_data)
            plotting_data = self._generate_plot_data(aggregated_data, pareto_data)

            self.logger.info("Pareto optimization completed successfully")
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
            self.logger.error(f"Error during Pareto optimization: {e}")
            raise

    def prepare_pareto_data(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_fronts: str,
        min_candidates: int,
        calibrated: bool,
    ) -> ParetoData:
        """
        Prepare Pareto optimization data with memory-efficient processing.

        Args:
            aggregated_data: Dictionary containing model results
            pareto_fronts: Number of Pareto fronts to consider or "auto"
            min_candidates: Minimum number of candidates to consider
            calibrated: Whether models are calibrated

        Returns:
            ParetoData: Processed Pareto data
        """
        self.logger.info("Preparing Pareto data")
        result_hyp_param = aggregated_data["result_hyp_param"]
        total_models = len(result_hyp_param)

        # Scale minimum candidates based on total models
        scaled_min_candidates = min(min_candidates, max(5, int(total_models * 0.1)))
        self.logger.info(f"Total models: {total_models} | Scaled minimum candidates: {scaled_min_candidates}")

        # Memory-efficient merge using only required columns
        aggregated_data["x_decomp_agg"] = pd.merge(
            aggregated_data["x_decomp_agg"], result_hyp_param[["robynPareto", "sol_id"]], on="sol_id", how="left"
        )

        # Process decomp_spend_dist in chunks
        self.logger.info("Processing model decompositions")
        chunk_size = 1000  # Adjust based on available memory
        decomp_spend_dist_list = []

        for i in range(0, len(self.model_outputs.trials), chunk_size):
            chunk_trials = self.model_outputs.trials[i : i + chunk_size]
            chunk_data = []

            for trial in chunk_trials:
                if trial.decomp_spend_dist is not None:
                    # Select only necessary columns
                    required_cols = ["trial", "iterNG", "iterPar", "rn", "mean_spend", "total_spend", "xDecompAgg", "sol_id"]
                    trial_data = trial.decomp_spend_dist[
                        [col for col in required_cols if col in trial.decomp_spend_dist.columns]
                    ]
                    chunk_data.append(trial_data)

            if chunk_data:
                chunk_df = pd.concat(chunk_data, ignore_index=True)
                decomp_spend_dist_list.append(chunk_df)

            # Clear memory
            del chunk_data

        decomp_spend_dist = pd.concat(decomp_spend_dist_list, ignore_index=True)
        del decomp_spend_dist_list  # Free memory

        # Add sol_id if not fixed hyperparameters
        if not self.model_outputs.hyper_fixed:
            decomp_spend_dist["sol_id"] = (
                decomp_spend_dist["trial"].astype(str)
                + "_"
                + decomp_spend_dist["iterNG"].astype(str)
                + "_"
                + decomp_spend_dist["iterPar"].astype(str)
            )

        # Efficient merge with only necessary columns
        decomp_spend_dist = pd.merge(
            decomp_spend_dist, result_hyp_param[["robynPareto", "sol_id"]], on="sol_id", how="left"
        )

        # Handle single model or fixed hyperparameters case
        if self.model_outputs.hyper_fixed or len(result_hyp_param) == 1:
            pareto_fronts = 1
            self.logger.info("Using single Pareto front due to fixed hyperparameters or single model")

        # Automatic Pareto front selection with memory optimization
        grouped_data = None
        if pareto_fronts == "auto":
            n_pareto = result_hyp_param["robynPareto"].notna().sum()
            self.logger.info(f"Number of Pareto-optimal solutions found: {n_pareto}")

            if n_pareto <= scaled_min_candidates and len(result_hyp_param) > 1 and not calibrated:
                self.logger.warning(f"Found only {n_pareto} Pareto-optimal solutions.")
                self.logger.info(f"Adjusting minimum candidates from {scaled_min_candidates} to {max(1, n_pareto)}")
                scaled_min_candidates = max(1, n_pareto)

            # Efficient grouping and calculation
            grouped_data = (
                result_hyp_param[result_hyp_param["robynPareto"].notna()]
                .groupby("robynPareto")
                .agg(n=("sol_id", "nunique"))
                .reset_index()
            )
            grouped_data["n_cum"] = grouped_data["n"].cumsum()
            auto_pareto = grouped_data[grouped_data["n_cum"] >= scaled_min_candidates]

            if len(auto_pareto) == 0:
                auto_pareto = grouped_data.iloc[[-1]]
                self.logger.warning(
                    f"Using all available Pareto fronts ({len(grouped_data)}) "
                    f"with {int(auto_pareto['n_cum'].iloc[0])} total candidates"
                )
            else:
                auto_pareto = auto_pareto.iloc[0]
                self.logger.info(
                    f"Selected {int(auto_pareto['robynPareto'])} Pareto-fronts "
                    f"containing {int(auto_pareto['n_cum'])} candidates"
                )

            pareto_fronts = int(auto_pareto["robynPareto"])

        pareto_fronts_vec = list(range(1, pareto_fronts + 1))

        # Filter data efficiently
        self.logger.info("Filtering data for selected Pareto fronts...")
        mask = decomp_spend_dist["robynPareto"].isin(pareto_fronts_vec)
        decomp_spend_dist_pareto = decomp_spend_dist[mask].copy()

        mask = result_hyp_param["robynPareto"].isin(pareto_fronts_vec)
        result_hyp_param_pareto = result_hyp_param[mask].copy()

        mask = aggregated_data["x_decomp_agg"]["robynPareto"].isin(pareto_fronts_vec)
        x_decomp_agg_pareto = aggregated_data["x_decomp_agg"][mask].copy()

        self.logger.info(f"Final number of models selected: {len(result_hyp_param_pareto)}")

        # Clear any remaining temporary variables
        del mask, grouped_data

        return ParetoData(
            decomp_spend_dist=decomp_spend_dist_pareto,
            result_hyp_param=result_hyp_param_pareto,
            x_decomp_agg=x_decomp_agg_pareto,
            pareto_fronts=pareto_fronts_vec,
        )

    def run_dt_resp(self, row: pd.Series, paretoData: ParetoData) -> Optional[dict]:
        """Calculate response curves for a given row of Pareto data."""
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
            self.logger.error(
                f"Error processing row for sol_id {row.get('sol_id', 'unknown')}, "
                f"rn {row.get('rn', 'unknown')}: {str(e)}"
            )
            return None

    def _compute_response_curves(
        self, pareto_data: ParetoData, aggregated_data: Dict[str, pd.DataFrame]
    ) -> ParetoData:
        """Calculate response curves with improved error handling and validation."""
        if pareto_data.decomp_spend_dist.empty:
            self.logger.warning("No data in decomp_spend_dist. Skipping response curves calculation.")
            return pareto_data

        self.logger.info(
            f"Calculating response curves for {len(pareto_data.decomp_spend_dist)} models' media variables..."
        )
        self.logger.debug(f"Available columns: {pareto_data.decomp_spend_dist.columns.tolist()}")

        required_columns = ["rn", "sol_id", "mean_spend", "total_spend"]
        missing_columns = [col for col in required_columns if col not in pareto_data.decomp_spend_dist.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in decomp_spend_dist: {missing_columns}")

        batch_size = min(100, len(pareto_data.decomp_spend_dist))
        resp_collect_list = []

        try:
            for i in range(0, len(pareto_data.decomp_spend_dist), batch_size):
                batch = pareto_data.decomp_spend_dist.iloc[i : i + batch_size]
                self.logger.debug(f"Processing batch {i//batch_size + 1}, size: {len(batch)}")

                if self.model_outputs.cores > 1 and len(batch) > 1:
                    with ProcessPoolExecutor(max_workers=self.model_outputs.cores) as executor:
                        run_dt_resp_partial = partial(self.run_dt_resp, paretoData=pareto_data)
                        futures = []
                        for _, row in batch.iterrows():
                            try:
                                futures.append(executor.submit(run_dt_resp_partial, row))
                            except Exception as e:
                                self.logger.error(f"Error submitting row to executor: {str(e)}")
                                continue

                        if not futures:
                            self.logger.warning("No futures created for this batch")
                            continue

                        results = []
                        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
                            try:
                                result = future.result()
                                if result is not None:
                                    results.append(result)
                            except Exception as e:
                                self.logger.error(f"Error processing future: {str(e)}")
                                continue

                        if results:
                            resp_collect_batch = pd.DataFrame(results)
                            resp_collect_list.append(resp_collect_batch)
                else:
                    results = []
                    for _, row in tqdm(batch.iterrows(), total=len(batch), desc="Processing rows"):
                        try:
                            result = self.run_dt_resp(row, paretoData=pareto_data)
                            if result is not None:
                                results.append(result)
                        except Exception as e:
                            self.logger.error(f"Error processing row: {str(e)}")
                            continue

                    if results:
                        resp_collect_batch = pd.DataFrame(results)
                        resp_collect_list.append(resp_collect_batch)

            if not resp_collect_list:
                self.logger.warning("No response curves were calculated successfully")
                return pareto_data

            resp_collect = pd.concat(resp_collect_list, ignore_index=True)
            self.logger.info(f"Successfully processed {len(resp_collect)} response curves")

            self.logger.info("Computing final metrics...")
            pareto_data.decomp_spend_dist = pd.merge(
                pareto_data.decomp_spend_dist, resp_collect, on=["sol_id", "rn"], how="left"
            )

            self.logger.info("Calculating ROI and CPA metrics...")
            metrics_df = pd.DataFrame()
            try:
                metrics_df = pareto_data.decomp_spend_dist.assign(
                    roi_mean=lambda x: x["mean_response"] / x["mean_spend"],
                    roi_total=lambda x: x["xDecompAgg"] / x["total_spend"],
                    cpa_mean=lambda x: x["mean_spend"] / x["mean_response"],
                    cpa_total=lambda x: x["total_spend"] / x["xDecompAgg"],
                )
            except Exception as e:
                self.logger.warning(f"Error calculating metrics: {str(e)}")
                self.logger.debug(f"Available columns: {pareto_data.decomp_spend_dist.columns.tolist()}")

            if not metrics_df.empty:
                pareto_data.decomp_spend_dist = metrics_df

            return pareto_data

        except Exception as e:
            self.logger.error(f"Error in response curves calculation: {str(e)}")
            self.logger.debug(f"decomp_spend_dist shape: {pareto_data.decomp_spend_dist.shape}")
            self.logger.debug(f"decomp_spend_dist columns: {pareto_data.decomp_spend_dist.columns.tolist()}")
            self.logger.debug(f"Number of response collect batches: {len(resp_collect_list)}")
            raise

    def _generate_plot_data(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_data: ParetoData,
    ) -> Dict[str, pd.DataFrame]:
        """Prepare data for various plots with metric calculation."""
        mediaVecCollect = pd.DataFrame()
        xDecompVecCollect = pd.DataFrame()
        plotDataCollect = {}
        df_caov_pct_all = pd.DataFrame()

        xDecompAgg = pareto_data.x_decomp_agg
        dt_mod = self.featurized_mmm_data.dt_mod
        dt_modRollWind = self.featurized_mmm_data.dt_modRollWind
        rw_start_loc = self.mmm_data.mmmdata_spec.rolling_window_start_which
        rw_end_loc = self.mmm_data.mmmdata_spec.rolling_window_end_which

        self.logger.info("Starting plot data generation...")
        self.logger.debug(f"Available columns in xDecompAgg: {xDecompAgg.columns.tolist()}")

        pareto_fronts_vec = pareto_data.pareto_fronts

        for pf in pareto_fronts_vec:
            self.logger.info(f"Processing Pareto front {pf}")

            plotMediaShare = xDecompAgg[
                (xDecompAgg["robynPareto"] == pf)
                & (xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends))
            ].copy()

            self.logger.debug(f"Shape of plotMediaShare: {plotMediaShare.shape}")

            try:
                if "mean_spend" in plotMediaShare.columns:
                    window_length = rw_end_loc - rw_start_loc
                    plotMediaShare["total_spend"] = plotMediaShare["mean_spend"] * window_length
                elif "xDecompAgg" in plotMediaShare.columns:
                    plotMediaShare["total_spend"] = plotMediaShare["xDecompAgg"]

                spend_sums = plotMediaShare.groupby("sol_id")["total_spend"].transform("sum")
                plotMediaShare["spend_share"] = plotMediaShare["total_spend"] / spend_sums

                effect_sums = plotMediaShare.groupby("sol_id")["xDecompAgg"].transform("sum")
                plotMediaShare["effect_share"] = plotMediaShare["xDecompAgg"] / effect_sums

                plotMediaShare["roi_total"] = plotMediaShare["xDecompAgg"] / plotMediaShare["total_spend"]
                plotMediaShare["cpa_total"] = plotMediaShare["total_spend"] / plotMediaShare["xDecompAgg"]

                self.logger.debug("Successfully calculated all required metrics")

            except Exception as e:
                self.logger.error(f"Error calculating metrics: {str(e)}")
                self.logger.debug(f"Available columns: {plotMediaShare.columns.tolist()}")
                continue

            uniqueSol = plotMediaShare["sol_id"].unique()
            plotWaterfall = xDecompAgg[xDecompAgg["robynPareto"] == pf]

            self.logger.info(f"Pareto-Front: {pf} [{len(uniqueSol)} models]")

            for sid in tqdm(uniqueSol, desc="Processing Solutions", unit="solution"):
                try:
                    temp = plotMediaShare[plotMediaShare["sol_id"] == sid]

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
                    plotDataCollect[sid] = {
                        "plot1data": plot1data,
                        "plot2data": plot2data,
                    }

                except Exception as e:
                    self.logger.error(f"Error processing solution {sid}: {str(e)}")
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
        """Calculate immediate and carryover effects for media channels."""
        if sol_id is None:
            sol_id = result_hyp_param["sol_id"].iloc[0]
        if start_date is None:
            start_date = self.mmm_data.mmmdata_spec.window_start
        if end_date is None:
            end_date = self.mmm_data.mmmdata_spec.window_end

        dt_modRollWind = pd.to_datetime(self.featurized_mmm_data.dt_modRollWind["ds"])
        dt_modRollWind = dt_modRollWind.dropna()

        if isinstance(start_date, (list, pd.Series)):
            start_date = start_date[0]

        start_date_closest = dt_modRollWind.iloc[(dt_modRollWind - pd.to_datetime(start_date)).abs().idxmin()]

        if isinstance(end_date, (list, pd.Series)):
            end_date = end_date[0]

        end_date_closest = dt_modRollWind.iloc[(dt_modRollWind - pd.to_datetime(end_date)).abs().idxmin()]

        rollingWindowStartWhich = dt_modRollWind[dt_modRollWind == start_date].index[0]
        rollingWindowEndWhich = dt_modRollWind[dt_modRollWind == end_date].index[0]
        rollingWindow = range(rollingWindowStartWhich, rollingWindowEndWhich + 1)

        self.logger.info("Calculating saturated dataframes with carryover and immediate parts")
        hypParamSam = result_hyp_param[result_hyp_param["sol_id"] == sol_id]
        hyperparameter = self._extract_hyperparameter(hypParamSam)

        dt_saturated_dfs = self.transformer.run_transformations(
            self.featurized_mmm_data,
            hyperparameter,
            hyperparameter.adstock,
        )

        coefs = pareto_data.x_decomp_agg.loc[pareto_data.x_decomp_agg["sol_id"] == sol_id, "coef"].values
        coefs_names = pareto_data.x_decomp_agg.loc[pareto_data.x_decomp_agg["sol_id"] == sol_id, "rn"].values

        coefs_df = pd.DataFrame({"name": coefs_names, "coefficient": coefs})

        self.logger.debug("Computing decomposition")
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

        mediaDecompImmediate = decompCollect["mediaDecompImmediate"].drop(columns=["ds", "y"], errors="ignore")
        mediaDecompImmediate.columns = [f"{col}_MDI" for col in mediaDecompImmediate.columns]

        mediaDecompCarryover = decompCollect["mediaDecompCarryover"].drop(columns=["ds", "y"], errors="ignore")
        mediaDecompCarryover.columns = [f"{col}_MDC" for col in mediaDecompCarryover.columns]

        temp = pd.concat(
            [decompCollect["xDecompVec"], mediaDecompImmediate, mediaDecompCarryover],
            axis=1,
        )
        temp["sol_id"] = sol_id
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
        this = vec_collect["xDecompVecImmediate"].columns.str.replace("_MDI", "", regex=False)
        vec_collect["xDecompVecImmediate"].columns = this
        vec_collect["xDecompVecCarryover"].columns = this

        df_caov = (vec_collect["xDecompVecCarryover"].drop(columns="ds").groupby("sol_id").sum().reset_index())
        df_total = vec_collect["xDecompVec"].drop(columns="ds").groupby("sol_id").sum().reset_index()
        df_caov_pct = df_caov.copy()
        df_caov_pct.loc[:, df_caov_pct.columns[1:]] = df_caov_pct.loc[:, df_caov_pct.columns[1:]].div(
            df_total.iloc[:, 1:].values
        )
        df_caov_pct = df_caov_pct.melt(id_vars="sol_id", var_name="rn", value_name="carryover_pct").fillna(0)

        self.logger.info("Aggregating final results")
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

        xDecompVecImmeCaov = (
            xDecompVecImmeCaov.groupby(["sol_id", "start_date", "end_date", "rn", "type"])
            .agg(response=("value", "sum"))
            .reset_index()
        )

        xDecompVecImmeCaov["percentage"] = xDecompVecImmeCaov["response"] / xDecompVecImmeCaov.groupby(
            ["sol_id", "start_date", "end_date", "type"]
        )["response"].transform("sum")
        xDecompVecImmeCaov.fillna(0, inplace=True)
        xDecompVecImmeCaov = xDecompVecImmeCaov.merge(df_caov_pct, on=["sol_id", "rn"], how="left")

        return xDecompVecImmeCaov

    def _extract_hyperparameter(self, hypParamSam: pd.DataFrame) -> Hyperparameters:
        """Extract hyperparameters from a given DataFrame."""
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
        """Decompose model inputs and calculate immediate and carryover effects."""
        self.logger.debug("Starting model decomposition")
        coefs = inputs["coefs"]
        y_pred = inputs["y_pred"]
        dt_modSaturated = inputs["dt_modSaturated"]
        dt_saturatedImmediate = inputs["dt_saturatedImmediate"]
        dt_saturatedCarryover = inputs["dt_saturatedCarryover"]
        dt_modRollWind = inputs["dt_modRollWind"]
        refreshAddedStart = inputs["refreshAddedStart"]

        y = dt_modSaturated["dep_var"]

        x = dt_modSaturated.drop(columns=["dep_var"])
        intercept = coefs["coefficient"].iloc[0]
        x_name = x.columns
        x_factor = x_name[x.dtypes == "category"]

        xDecomp = pd.DataFrame()

        for name in x.columns:
            coefficient_value = coefs.loc[coefs["name"] == name, "coefficient"].values
            xDecomp[name] = x[name] * (coefficient_value if len(coefficient_value) > 0 else 0)

        xDecomp.insert(0, "intercept", intercept)

        xDecompOut = pd.concat(
            [
                pd.DataFrame({"ds": dt_modRollWind["ds"], "y": y, "y_pred": y_pred}),
                xDecomp,
            ],
            axis=1,
        )

        sel_coef = coefs["name"].isin(dt_saturatedImmediate.columns)
        coefs_media = coefs[sel_coef].set_index("name")["coefficient"]

        mediaDecompImmediate = pd.DataFrame(
            {name: dt_saturatedImmediate[name] * coefs_media[name] for name in coefs_media.index}
        )
        mediaDecompCarryover = pd.DataFrame(
            {name: dt_saturatedCarryover[name] * coefs_media[name] for name in coefs_media.index}
        )

        self.logger.debug("Model decomposition completed")
        return {
            "xDecompVec": xDecompOut,
            "mediaDecompImmediate": mediaDecompImmediate,
            "mediaDecompCarryover": mediaDecompCarryover,
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
            aggregated_data: Dictionary containing model results
            pareto_fronts: Number of Pareto fronts to compute or "auto"
            min_candidates: Minimum number of candidates to consider
            calibration_constraint: Constraint for calibration

        Returns:
            pd.DataFrame: A dataframe of Pareto-optimal solutions with their corresponding front numbers
        """
        self.logger.info("Computing Pareto fronts")
        resultHypParam = aggregated_data["result_hyp_param"]
        xDecompAgg = aggregated_data["x_decomp_agg"]
        resultCalibration = aggregated_data["result_calibration"]

        if not self.model_outputs.hyper_fixed:
            self.logger.debug("Processing non-fixed hyperparameters")
            # Filter and group data to calculate coef0
            xDecompAggCoef0 = (
                xDecompAgg[xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends)]
                .groupby("sol_id")["coef"]
                .apply(lambda x: min(x.dropna()) == 0)
            )

            # Calculate quantiles
            mape_lift_quantile10 = resultHypParam["mape"].quantile(calibration_constraint)
            nrmse_quantile90 = resultHypParam["nrmse"].quantile(0.9)
            decomprssd_quantile90 = resultHypParam["decomp.rssd"].quantile(0.9)

            self.logger.debug(f"MAPE lift quantile (10%): {mape_lift_quantile10}")
            self.logger.debug(f"NRMSE quantile (90%): {nrmse_quantile90}")
            self.logger.debug(f"DECOMP.RSSD quantile (90%): {decomprssd_quantile90}")

            # Merge resultHypParam with xDecompAggCoef0
            resultHypParam = pd.merge(resultHypParam, xDecompAggCoef0, on="sol_id", how="left")

            # Create mape.qt10 column
            resultHypParam["mape.qt10"] = (
                (resultHypParam["mape"] <= mape_lift_quantile10)
                & (resultHypParam["nrmse"] <= nrmse_quantile90)
                & (resultHypParam["decomp.rssd"] <= decomprssd_quantile90)
            )

            # Filter resultHypParam
            resultHypParamPareto = resultHypParam[resultHypParam["mape.qt10"] == True]

            self.logger.info(f"Number of solutions passing constraints: {len(resultHypParamPareto)}")

            # Calculate Pareto front
            self.logger.debug("Calculating Pareto fronts")
            pareto_fronts_df = ParetoOptimizer._pareto_fronts(resultHypParamPareto, pareto_fronts=pareto_fronts)

            # Merge resultHypParamPareto with pareto_fronts_df
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
            self.logger.info("Using fixed hyperparameters")
            resultHypParam = resultHypParam.assign(mape_qt10=True, robynPareto=1, coef0=np.nan)

        # Calculate combined weighted error scores
        self.logger.debug("Calculating error scores")
        resultHypParam["error_score"] = ParetoUtils.calculate_errors_scores(
            df=resultHypParam, ts_validation=self.model_outputs.ts_validation
        )

        self.logger.info("Pareto front computation completed")
        return resultHypParam

    def _pareto_fronts(resultHypParamPareto: pd.DataFrame, pareto_fronts: str) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        Args:
            resultHypParamPareto: DataFrame containing model results with nrmse and decomp.rssd
            pareto_fronts: Number of Pareto fronts to calculate or "auto"

        Returns:
            DataFrame with Pareto front assignments
        """
        nrmse_values = resultHypParamPareto["nrmse"]
        decomp_rssd_values = resultHypParamPareto["decomp.rssd"]

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
