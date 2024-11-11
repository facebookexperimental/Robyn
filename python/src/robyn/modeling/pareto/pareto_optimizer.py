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

from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.pareto.hill_calculator import HillCalculator
from robyn.modeling.pareto.immediate_carryover import ImmediateCarryoverCalculator
from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.modeling.pareto.response_curve import ResponseCurveCalculator, ResponseOutput
from robyn.modeling.transformations.transformations import Transformation
from tqdm import tqdm  # Import tqdm for progress bar 


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
        hyperparameter: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
        holidays_data: HolidaysData,
    ):
        """
        Initialize the ParetoOptimizer.

        Args:
            mmm_data (MMMData): Input data for the marketing mix model.
            model_outputs (ModelOutputs): Output data from the model runs.
            hyperparameter (Hyperparameters): Hyperparameters for the model runs.
        """
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        self.hyperparameter = hyperparameter
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
        self._validate_model_outputs()
        self.logger.info("Starting model data aggregation")

        # Process all trials to ensure proper sol_id
        self.model_outputs.trials = [self._ensure_trial_ids(trial) for trial in self.model_outputs.trials]
        
        hyper_fixed = self.model_outputs.hyper_fixed
        # Extract resultCollect from self.model_outputs
        trials = [
            model
            for model in self.model_outputs.trials
            if hasattr(model, "resultCollect")
        ]

        # Create lists of resultHypParam and xDecompAgg using list comprehension
        resultHypParam_list = [
            trial.result_hyp_param for trial in self.model_outputs.trials
        ]
        xDecompAgg_list = [trial.x_decomp_agg for trial in self.model_outputs.trials]

        # Concatenate the lists into DataFrames using pd.concat
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

        # Check if bootstrap results are available
        if (len(xDecompAgg["sol_id"].unique()) == 1 and "boot_mean" not in xDecompAgg.columns):
            # Get bootstrap results from model_outputs object
            bootstrap = getattr(self.model_outputs, "bootstrap", None)
            if bootstrap is not None:
                self.logger.info("Merging bootstrap results")
                xDecompAgg = pd.merge(
                    xDecompAgg, bootstrap, left_on="rn", right_on="variable"
                )

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
        try:
            self.logger.info("Starting Pareto optimization")
            aggregated_data = self._aggregate_model_data(calibrated)
            aggregated_data["result_hyp_param"] = self._compute_pareto_fronts(
                aggregated_data, pareto_fronts, calibration_constraint
            )

            pareto_data = self.prepare_pareto_data(
                aggregated_data, pareto_fronts, min_candidates, calibrated
            )
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

        # 1. Binding Pareto results
        aggregated_data["x_decomp_agg"] = pd.merge(
            aggregated_data["x_decomp_agg"],
            result_hyp_param[["robynPareto", "sol_id"]],
            on="sol_id",
            how="left",
        )

        # Collect decomp_spend_dist from each trial and add the trial number
        decomp_spend_dist = pd.concat(
            [
                trial.decomp_spend_dist
                for trial in self.model_outputs.trials
                if trial.decomp_spend_dist is not None
            ],
            ignore_index=True,
        )

        # Add sol_id if hyper_fixed is False
        if not self.model_outputs.hyper_fixed:
            decomp_spend_dist["sol_id"] = (
                decomp_spend_dist["trial"].astype(str)
                + "_"
                + decomp_spend_dist["iterNG"].astype(str)
                + "_"
                + decomp_spend_dist["iterPar"].astype(str)
            )

        decomp_spend_dist = pd.merge(
            decomp_spend_dist,
            result_hyp_param[["robynPareto", "sol_id"]],
            on="sol_id",
            how="left",
        )

        # Determining the number of Pareto fronts
        if self.model_outputs.hyper_fixed or len(result_hyp_param) == 1:
            pareto_fronts = 1
            self.logger.info("Using single Pareto front due to fixed hyperparameters or single model")

        # Handling automatic Pareto front selection
        if pareto_fronts == "auto":
            n_pareto = result_hyp_param["robynPareto"].notna().sum()
            self.logger.info(f"Number of Pareto-optimal solutions found: {n_pareto}")

            if (
                n_pareto <= min_candidates
                and len(result_hyp_param) > 1
                and not calibrated
            ):
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
            auto_pareto = grouped_data[grouped_data["n_cum"] >= min_candidates]
            
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
        # Creating Pareto front vector
        pareto_fronts_vec = list(range(1, pareto_fronts + 1))

        # Filtering data for selected Pareto fronts
        self.logger.info("Filtering data for selected Pareto fronts...")
        decomp_spend_dist_pareto = decomp_spend_dist[
            decomp_spend_dist["robynPareto"].isin(pareto_fronts_vec)
        ]
        result_hyp_param_pareto = result_hyp_param[
            result_hyp_param["robynPareto"].isin(pareto_fronts_vec)
        ]
        x_decomp_agg_pareto = aggregated_data["x_decomp_agg"][
            aggregated_data["x_decomp_agg"]["robynPareto"].isin(pareto_fronts_vec)
        ]

        return ParetoData(
            decomp_spend_dist=decomp_spend_dist_pareto,
            result_hyp_param=result_hyp_param_pareto,
            x_decomp_agg=x_decomp_agg_pareto,
            pareto_fronts=pareto_fronts_vec,
        )
    
    def run_dt_resp(self, row: pd.Series, paretoData: ParetoData) -> pd.Series:
        """
        Calculate response curves for a given row of Pareto data.
        This method is used for parallel processing.

        Args:
            row (pd.Series): A row of Pareto data.
            paretoData (ParetoData): Pareto data.

        Returns:
            pd.Series: A pandas series containing the row of response curves.
        """
        try:
            get_sol_id = row["sol_id"]
            get_spendname = row["rn"]
            startRW = self.mmm_data.mmmdata_spec.rolling_window_start_which
            endRW = self.mmm_data.mmmdata_spec.rolling_window_end_which

            response_curve_calculator = ResponseCurveCalculator(
                mmm_data=self.mmm_data,
                model_outputs=self.model_outputs,
                hyperparameter=self.hyperparameter,
            )

            response_output: ResponseOutput = response_curve_calculator.calculate_response(
                select_model=get_sol_id,
                metric_name=get_spendname,
                date_range="all",
                dt_hyppar=paretoData.result_hyp_param,
                dt_coef=paretoData.x_decomp_agg,
                quiet=True,
            )

            mean_spend_adstocked = np.mean(response_output.input_total[startRW:endRW])
            mean_carryover = np.mean(response_output.input_carryover[startRW:endRW])

            dt_hyppar = paretoData.result_hyp_param[
                paretoData.result_hyp_param["sol_id"] == get_sol_id
            ]
            chn_adstocked = pd.DataFrame(
                {get_spendname: response_output.input_total[startRW:endRW]}
            )
            dt_coef = paretoData.x_decomp_agg[
                (paretoData.x_decomp_agg["sol_id"] == get_sol_id)
                & (paretoData.x_decomp_agg["rn"] == get_spendname)
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

            return pd.Series({
                "mean_response": mean_response,
                "mean_spend_adstocked": mean_spend_adstocked,
                "mean_carryover": mean_carryover,
                "rn": row["rn"],
                "sol_id": row["sol_id"],
            })
        except Exception as e:
            self.logger.error(
                f"Error processing row for sol_id {row.get('sol_id', 'unknown')}, "
                f"rn {row.get('rn', 'unknown')}: {str(e)}"
            )
            return None
    
    def _compute_response_curves(
        self, pareto_data: ParetoData, aggregated_data: Dict[str, pd.DataFrame]
    ) -> ParetoData:
        """
        Calculate response curves for Pareto-optimal solutions.

        This method computes response curves for each media channel in each Pareto-optimal solution,
        providing insights into the relationship between media spend and response.

        Args:
            pareto_data (ParetoData): Pareto data.

        Returns:
            ParetoData: Pareto data with updated decomp_spend_dist and x_decomp_agg.
        """
        if pareto_data.decomp_spend_dist.empty:
            self.logger.warning("No data in decomp_spend_dist. Skipping response curves calculation.")
            return pareto_data
        self.logger.info(
            f"Calculating response curves for {len(pareto_data.decomp_spend_dist)} models' media variables..."
        )
        self.logger.debug(f"Available columns: {pareto_data.decomp_spend_dist.columns.tolist()}")

        # Parallel processing
        batch_size = min(1000, len(pareto_data.decomp_spend_dist))
        resp_collect_list = []

        try:
            for i in range(0, len(pareto_data.decomp_spend_dist), batch_size):
                batch = pareto_data.decomp_spend_dist.iloc[i : i + batch_size]
                self.logger.debug(f"Processing batch {i//batch_size + 1}, size: {len(batch)}, cores: {self.model_outputs.cores}")

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
            # Merge results
            self.logger.info(f"Successfully processed {len(resp_collect)} response curves")
            self.logger.info("Computing final metrics...")
            pareto_data.decomp_spend_dist = pd.merge(
                pareto_data.decomp_spend_dist, resp_collect, on=["sol_id", "rn"], how="left"
            )

            # Calculate ROI and CPA metrics after merging
            self.logger.info("Calculating ROI and CPA metrics...")
            pareto_data.decomp_spend_dist["roi_mean"] = (
                pareto_data.decomp_spend_dist["mean_response"]
                / pareto_data.decomp_spend_dist["mean_spend"]
            )
            pareto_data.decomp_spend_dist["roi_total"] = (
                pareto_data.decomp_spend_dist["xDecompAgg"]
                / pareto_data.decomp_spend_dist["total_spend"]
            )
            pareto_data.decomp_spend_dist["cpa_mean"] = (
                pareto_data.decomp_spend_dist["mean_spend"]
                / pareto_data.decomp_spend_dist["mean_response"]
            )
            pareto_data.decomp_spend_dist["cpa_total"] = (
                pareto_data.decomp_spend_dist["total_spend"]
                / pareto_data.decomp_spend_dist["xDecompAgg"]
            )

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

        self.logger.info("Starting plot data generation...")
        self.logger.debug(f"Available columns in xDecompAgg: {xDecompAgg.columns.tolist()}")

        pareto_fronts_vec = pareto_data.pareto_fronts

        for pf in pareto_fronts_vec:
            self.logger.info(f"Processing Pareto front {pf}")
            plotMediaShare = xDecompAgg[
                (xDecompAgg["robynPareto"] == pf)
                & (xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends))
            ]
            self.logger.debug(f"Shape of plotMediaShare: {plotMediaShare.shape}")
            
            uniqueSol = plotMediaShare["sol_id"].unique()

            plotWaterfall = xDecompAgg[xDecompAgg["robynPareto"] == pf]

            self.logger.info(f"Pareto-Front: {pf} [{len(uniqueSol)} models]")

            for sid in tqdm(uniqueSol, desc="Processing Solutions", unit="solution"):
                try:
                    # 1. Spend x effect share comparison
                    temp = plotMediaShare[plotMediaShare["sol_id"] == sid].melt(
                        id_vars=["rn", "nrmse", "decomp.rssd", "rsq_train"],
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

                    plotMediaShareLoopBar = temp[
                        temp["variable"].isin(["spend_share", "effect_share"])
                    ]
                    plotMediaShareLoopLine = temp[
                        temp["variable"]
                        == (
                            "cpa_total"
                            if self.mmm_data.mmmdata_spec.dep_var_type == "conversion"
                            else "roi_total"
                        )
                    ]

                    line_rm_inf = ~np.isinf(plotMediaShareLoopLine["value"])
                    ySecScale = (
                        max(plotMediaShareLoopLine["value"][line_rm_inf])
                        / max(plotMediaShareLoopBar["value"])
                        * 1.1
                    )

                    plot1data = {
                        "plotMediaShareLoopBar": plotMediaShareLoopBar,
                        "plotMediaShareLoopLine": plotMediaShareLoopLine,
                        "ySecScale": ySecScale,
                    }

                    self.logger.debug(f"Generated plot1data for sid: {sid}")
                    
                    # 2. Waterfall
                    plotWaterfallLoop = plotWaterfall[
                        plotWaterfall["sol_id"] == sid
                    ].sort_values("xDecompPerc")
                    plotWaterfallLoop["end"] = 1 - plotWaterfallLoop["xDecompPerc"].cumsum()
                    plotWaterfallLoop["start"] = plotWaterfallLoop["end"].shift(1).fillna(1)
                    plotWaterfallLoop["id"] = range(1, len(plotWaterfallLoop) + 1)
                    plotWaterfallLoop["rn"] = pd.Categorical(plotWaterfallLoop["rn"])
                    plotWaterfallLoop["sign"] = pd.Categorical(
                        np.where(
                            plotWaterfallLoop["xDecompPerc"] >= 0, "Positive", "Negative"
                        )
                    )

                    plotWaterfallLoop = plotWaterfallLoop[
                        [
                            "id",
                            "rn",
                            "coef",
                            "xDecompAgg",
                            "xDecompPerc",
                            "start",
                            "end",
                            "sign",
                        ]
                    ]

                    plot2data = {"plotWaterfallLoop": plotWaterfallLoop}
                    self.logger.debug(f"Generated plot2data for sid: {sid}")

                    # 3. Adstock rate
                    dt_geometric = None
                    weibullCollect = None
                    resultHypParamLoop = pareto_data.result_hyp_param[
                        pareto_data.result_hyp_param["sol_id"] == sid
                    ]
                    get_hp_names = [
                        name
                        for name in self.hyperparameter.hyperparameters.keys()
                        if not name.endswith("_penalty")
                    ]
                    hypParam = resultHypParamLoop[get_hp_names]

                    wb_type = self.hyperparameter.adstock
                    if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
                        hypParam_thetas = [
                            hypParam[f"{media}_thetas"].iloc[0]
                            for media in self.mmm_data.mmmdata_spec.all_media
                        ]
                        dt_geometric = pd.DataFrame(
                            {
                                "channels": self.mmm_data.mmmdata_spec.all_media,
                                "thetas": hypParam_thetas,
                            }
                        )
                    elif self.hyperparameter.adstock in [
                        AdstockType.WEIBULL_CDF,
                        AdstockType.WEIBULL_PDF,
                    ]:
                        shapeVec = [
                            hypParam[f"{media}_shapes"].iloc[0]
                            for media in self.mmm_data.mmmdata_spec.all_media
                        ]
                        scaleVec = [
                            hypParam[f"{media}_scales"].iloc[0]
                            for media in self.mmm_data.mmmdata_spec.all_media
                        ]
                        weibullCollect = []
                        for v1 in range(len(self.mmm_data.mmmdata_spec.all_media)):
                            dt_weibull = pd.DataFrame(
                                {
                                    "x": range(
                                        1,
                                        self.mmm_data.mmmdata_spec.rolling_window_length
                                        + 1,
                                    ),
                                    "decay_accumulated": self.transformer.adstock_weibull(
                                        range(
                                            1,
                                            self.mmm_data.mmmdata_spec.rolling_window_length
                                            + 1,
                                        ),
                                        shape=shapeVec[v1],
                                        scale=scaleVec[v1],
                                        adstockType=wb_type,
                                    )["thetaVecCum"],
                                    "adstockType": wb_type,
                                    "channel": self.mmm_data.mmmdata_spec.all_media[v1],
                                }
                            )
                            dt_weibull["halflife"] = (
                                (dt_weibull["decay_accumulated"] - 0.5).abs().idxmin()
                            )
                            max_non0 = (dt_weibull["decay_accumulated"] > 0.001).argmax()
                            dt_weibull["cut_time"] = (
                                max_non0 * 2
                                if max_non0 <= 5
                                else int(max_non0 + max_non0 / 3)
                            )
                            weibullCollect.append(dt_weibull)

                        weibullCollect = pd.concat(weibullCollect)
                        weibullCollect = weibullCollect[
                            weibullCollect["x"] <= weibullCollect["cut_time"].max()
                        ]

                    plot3data = {
                        "dt_geometric": dt_geometric,
                        "weibullCollect": weibullCollect,
                        "wb_type": wb_type,
                    }
                    self.logger.debug(f"Generated plot3data for sid: {sid}")

                    # 4. Spend response curve
                    dt_transformPlot = dt_mod[["ds"] + self.mmm_data.mmmdata_spec.all_media]
                    dt_transformSpend = pd.concat(
                        [
                            dt_transformPlot[["ds"]],
                            self.mmm_data.data[
                                self.mmm_data.mmmdata_spec.paid_media_spends
                            ],
                        ],
                        axis=1,
                    )
                    dt_transformSpendMod = dt_transformPlot.iloc[rw_start_loc:rw_end_loc]
                    dt_transformAdstock = dt_transformPlot.copy()
                    dt_transformSaturation = dt_transformPlot.iloc[rw_start_loc:rw_end_loc]

                    all_media_channels = self.mmm_data.mmmdata_spec.all_media
                    for med in range(len(all_media_channels)):
                        med_select = all_media_channels[med]
                        m = pd.Series(dt_transformPlot[med_select].values)
                        adstock = self.hyperparameter.adstock
                        if adstock == AdstockType.GEOMETRIC:
                            thetas = hypParam[f"{all_media_channels[med]}_thetas"].values
                            channelHyperparam = ChannelHyperparameters(thetas=thetas)
                        elif adstock in [
                            AdstockType.WEIBULL_PDF,
                            AdstockType.WEIBULL_CDF,
                        ]:
                            shapes = hypParam[f"{all_media_channels[med]}_shapes"].values
                            scales = hypParam[f"{all_media_channels[med]}_scales"].values
                            channelHyperparam = ChannelHyperparameters(
                                shapes=shapes, scales=scales
                            )

                        x_list = self.transformer.transform_adstock(
                            m, adstock, channelHyperparam
                        )
                        m_adstocked = x_list.x_decayed
                        dt_transformAdstock[med_select] = m_adstocked
                        m_adstockedRollWind = m_adstocked[rw_start_loc:rw_end_loc]

                        # Saturation
                        alpha = hypParam[f"{all_media_channels[med]}_alphas"].iloc[0]
                        gamma = hypParam[f"{all_media_channels[med]}_gammas"].iloc[0]
                        dt_transformSaturation.loc[:, med_select] = (
                            self.transformer.saturation_hill(
                                x=m_adstockedRollWind, alpha=alpha, gamma=gamma
                            )
                        )

                    dt_transformSaturationDecomp = dt_transformSaturation.copy()
                    for i in range(len(all_media_channels)):
                        coef = plotWaterfallLoop["coef"][
                            plotWaterfallLoop["rn"] == all_media_channels[i]
                        ].values[0]
                        dt_transformSaturationDecomp[all_media_channels[i]] *= coef

                    dt_transformSaturationSpendReverse = dt_transformAdstock.iloc[
                        rw_start_loc:rw_end_loc
                    ]

                    # Spend response curve
                    dt_scurvePlot = dt_transformSaturationDecomp.melt(
                        id_vars=["ds"],
                        value_vars=all_media_channels,
                        var_name="channel",
                        value_name="response",
                    )

                    # Gather spend data and merge it into dt_scurvePlot
                    spend_data = dt_transformSaturationSpendReverse.melt(
                        id_vars=["ds"],
                        value_vars=all_media_channels,
                        var_name="channel",
                        value_name="spend",
                    )

                    # Merge spend data into dt_scurvePlot based on the 'channel' and 'ds' columns
                    dt_scurvePlot = dt_scurvePlot.merge(
                        spend_data[["ds", "channel", "spend"]],
                        on=["ds", "channel"],
                        how="left",
                    )

                    # Remove outlier introduced by MM nls fitting
                    dt_scurvePlot = dt_scurvePlot[dt_scurvePlot["spend"] >= 0]

                    # Calculate dt_scurvePlotMean
                    dt_scurvePlotMean = plotWaterfall[
                        (plotWaterfall["sol_id"] == sid)
                        & (~plotWaterfall["mean_spend"].isna())
                    ][
                        [
                            "rn",
                            "mean_spend",
                            "mean_spend_adstocked",
                            "mean_carryover",
                            "mean_response",
                            "sol_id",
                        ]
                    ].rename(
                        columns={"rn": "channel"}
                    )

                    plot4data = {
                        "dt_scurvePlot": dt_scurvePlot,
                        "dt_scurvePlotMean": dt_scurvePlotMean,
                    }

                    self.logger.debug(f"Generated plot4data for sid: {sid}")
                    
                    # 5. Fitted vs actual
                    col_order = (
                        ["ds", "dep_var"]
                        + self.mmm_data.mmmdata_spec.all_media
                        + [var.value for var in self.holidays_data.prophet_vars]
                        + self.mmm_data.mmmdata_spec.context_vars
                    )
                    selected_columns = (
                        ["ds", "dep_var"]
                        + [var.value for var in self.holidays_data.prophet_vars]
                        + self.mmm_data.mmmdata_spec.context_vars
                    )

                    # Create a DataFrame with the selected columns
                    dt_transformDecomp = dt_modRollWind[selected_columns]
                    # Bind columns from dt_transformSaturation
                    dt_transformDecomp = pd.concat(
                        [
                            dt_transformDecomp,
                            dt_transformSaturation[self.mmm_data.mmmdata_spec.all_media],
                        ],
                        axis=1,
                    )
                    dt_transformDecomp = dt_transformDecomp[col_order]

                    # Create xDecompVec by filtering and pivoting xDecompAgg
                    xDecompVec = (
                        xDecompAgg[xDecompAgg["sol_id"] == sid][["sol_id", "rn", "coef"]]
                        .pivot(index="sol_id", columns="rn", values="coef")
                        .reset_index()
                    )

                    if "(Intercept)" not in xDecompVec.columns:
                        xDecompVec["(Intercept)"] = 0

                    xDecompVec = xDecompVec[
                        ["sol_id", "(Intercept)"]
                        + [col for col in col_order if col not in ["ds", "dep_var"]]
                    ]
                    intercept = xDecompVec["(Intercept)"].values[0]

                    # Multiply scurved and coefs
                    scurved = dt_transformDecomp.drop(columns=["ds", "dep_var"])
                    coefs = xDecompVec.drop(columns=["sol_id", "(Intercept)"])
                    xDecompVec = pd.DataFrame(
                        np.multiply(
                            scurved.values,
                            coefs.values,
                        ),
                        columns=coefs.columns,  # Use the columns from coefs
                    )

                    # Add intercept and calculate depVarHat
                    xDecompVec["intercept"] = intercept
                    xDecompVec["depVarHat"] = xDecompVec.sum(axis=1) + intercept

                    # Add sol_id back to xDecompVec
                    xDecompVec["sol_id"] = sid

                    xDecompVec = pd.concat(
                        [dt_transformDecomp[["ds", "dep_var"]], xDecompVec], axis=1
                    )

                    # Prepare xDecompVecPlot
                    xDecompVecPlot = xDecompVec[["ds", "dep_var", "depVarHat"]].rename(
                        columns={"dep_var": "actual", "depVarHat": "predicted"}
                    )
                    xDecompVecPlotMelted = xDecompVecPlot.melt(
                        id_vars="ds", var_name="variable", value_name="value"
                    )

                    # Extract R-squared value
                    rsq = xDecompAgg[xDecompAgg["sol_id"] == sid]["rsq_train"].values[0]
                    plot5data = {"xDecompVecPlotMelted": xDecompVecPlotMelted, "rsq": rsq}
                    self.logger.debug(f"Generated plot5data for sid: {sid}")

                    # 6. Diagnostic: fitted vs residual
                    plot6data = {"xDecompVecPlot": xDecompVecPlot}
                    self.logger.debug(f"Generated plot6data for sid: {sid}")

                    # 7. Immediate vs carryover response
                    plot7data = self.robyn_immcarr(
                        pareto_data, aggregated_data["result_hyp_param"], sid
                    )
                    self.logger.debug(f"Generated plot7data for sid: {sid}")

                    df_caov_pct_all = pd.concat([df_caov_pct_all, plot7data])

                    # Gather all results
                    mediaVecCollect = pd.concat(
                        [ 
                            mediaVecCollect,
                            dt_transformPlot.assign(type="rawMedia", sol_id=sid),
                            dt_transformSpend.assign(type="rawSpend", sol_id=sid),
                            dt_transformSpendMod.assign(
                                type="predictedExposure", sol_id=sid
                            ),
                            dt_transformAdstock.assign(type="adstockedMedia", sol_id=sid),
                            dt_transformSaturation.assign(type="saturatedMedia", sol_id=sid),
                            dt_transformSaturationSpendReverse.assign(
                                type="saturatedSpendReversed", sol_id=sid
                            ),
                            dt_transformSaturationDecomp.assign(
                                type="decompMedia", sol_id=sid
                            ),
                        ],
                        ignore_index=True,
                    )

                    xDecompVecCollect = pd.concat(
                        [xDecompVecCollect, xDecompVec], ignore_index=True
                    )
                    plotDataCollect[sid] = {
                        "plot1data": plot1data,
                        "plot2data": plot2data,
                        "plot3data": plot3data,
                        "plot4data": plot4data,
                        "plot5data": plot5data,
                        "plot6data": plot6data,
                        "plot7data": plot7data,
                    }
                except Exception as e:
                    self.logger.error(f"Error processing solution {sid}: {str(e)}")
                    raise e

        pareto_solutions = set()
        if "sol_id" in mediaVecCollect.columns:
            # Update the set with unique sol_id values from the DataFrame
            pareto_solutions.update(mediaVecCollect["sol_id"].unique())

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

        dt_modRollWind = pd.to_datetime(self.featurized_mmm_data.dt_modRollWind["ds"])
        dt_modRollWind = dt_modRollWind.dropna()

        # Check if start_date is a single value
        if isinstance(start_date, (list, pd.Series)):
            start_date = start_date[0]

        # Find the closest start_date
        start_date = dt_modRollWind.iloc[
            (dt_modRollWind - pd.to_datetime(start_date)).abs().idxmin()
        ]

        # Check if end_date is a single value
        if isinstance(end_date, (list, pd.Series)):
            end_date = end_date[0]  # Take the first element if it's a list or Series

        # Find the closest end_date
        end_date = dt_modRollWind.iloc[
            (dt_modRollWind - pd.to_datetime(end_date)).abs().idxmin()
        ]

        # Filter for custom window
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

        # Calculate decomposition
        coefs = pareto_data.x_decomp_agg.loc[
            pareto_data.x_decomp_agg["sol_id"] == sol_id, "coef"
        ].values
        coefs_names = pareto_data.x_decomp_agg.loc[
            pareto_data.x_decomp_agg["sol_id"] == sol_id, "rn"
        ].values

        # Create a DataFrame to hold coefficients and their names
        coefs_df = pd.DataFrame({"name": coefs_names, "coefficient": coefs})

        self.logger.debug("Computing decomposition")
        decompCollect = self._model_decomp(
            inputs={
                "coefs": coefs_df,
                "y_pred": dt_saturated_dfs.dt_modSaturated["dep_var"].iloc[
                    rollingWindow
                ],
                "dt_modSaturated": dt_saturated_dfs.dt_modSaturated.iloc[rollingWindow],
                "dt_saturatedImmediate": dt_saturated_dfs.dt_saturatedImmediate.iloc[
                    rollingWindow
                ],
                "dt_saturatedCarryover": dt_saturated_dfs.dt_saturatedCarryover.iloc[
                    rollingWindow
                ],
                "dt_modRollWind": self.featurized_mmm_data.dt_modRollWind.iloc[
                    rollingWindow
                ],
                "refreshAddedStart": start_date,
            }
        )

        # Media decomposition
        mediaDecompImmediate = decompCollect["mediaDecompImmediate"].drop(
            columns=["ds", "y"], errors="ignore"
        )
        mediaDecompImmediate.columns = [
            f"{col}_MDI" for col in mediaDecompImmediate.columns
        ]

        mediaDecompCarryover = decompCollect["mediaDecompCarryover"].drop(
            columns=["ds", "y"], errors="ignore"
        )
        mediaDecompCarryover.columns = [
            f"{col}_MDC" for col in mediaDecompCarryover.columns
        ]

        # Combine results
        temp = pd.concat(
            [decompCollect["xDecompVec"], mediaDecompImmediate, mediaDecompCarryover],
            axis=1,
        )
        temp["sol_id"] = sol_id

        # Create vector collections
        vec_collect = {
            "xDecompVec": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDI")
                    | temp.columns.str.endswith("_MDC")
                ]
            ),
            "xDecompVecImmediate": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDC")
                    | temp.columns.isin(self.mmm_data.mmmdata_spec.all_media)
                ]
            ),
            "xDecompVecCarryover": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDI")
                    | temp.columns.isin(self.mmm_data.mmmdata_spec.all_media)
                ]
            ),
        }

        # Rename columns
        this = vec_collect["xDecompVecImmediate"].columns.str.replace(
            "_MDI", "", regex=False
        )
        vec_collect["xDecompVecImmediate"].columns = this
        vec_collect["xDecompVecCarryover"].columns = this

        # Calculate carryover percentages
        df_caov = (
            vec_collect["xDecompVecCarryover"].groupby("sol_id").sum().reset_index()
        ).drop(columns="ds")
        df_total = (
            vec_collect["xDecompVec"]
            .groupby("sol_id")
            .sum()
            .reset_index()
            .drop(columns="ds")
        )

        df_caov_pct = df_caov.copy()
        df_caov_pct.loc[:, df_caov_pct.columns[1:]] = df_caov_pct.loc[
            :, df_caov_pct.columns[1:]
        ].div(df_total.iloc[:, 1:].values).astype('float64')
        df_caov_pct = df_caov_pct.melt(
            id_vars="sol_id", var_name="rn", value_name="carryover_pct"
        ).fillna(0)

        # Gather everything in an aggregated format
        self.logger.info("Aggregating final results from decomposition carryover and immediate parts")
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
            xDecompVecImmeCaov.groupby(
                ["sol_id", "start_date", "end_date", "rn", "type"]
            )
            .agg(response=("value", "sum"))
            .reset_index()
        )

        xDecompVecImmeCaov["percentage"] = xDecompVecImmeCaov[
            "response"
        ] / xDecompVecImmeCaov.groupby(["sol_id", "start_date", "end_date", "type"])[
            "response"
        ].transform(
            "sum"
        )
        xDecompVecImmeCaov.fillna(0, inplace=True)

        # Join with carryover percentages
        xDecompVecImmeCaov = xDecompVecImmeCaov.merge(
            df_caov_pct, on=["sol_id", "rn"], how="left"
        )

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
            if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
                thetas = hypParamSam[f"{med}_thetas"].values
                channelHyperparams[med] = ChannelHyperparameters(
                    thetas=thetas,
                    alphas=alphas,
                    gammas=gammas,
                )
            elif self.hyperparameter.adstock in [
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

        return Hyperparameters(
            adstock=self.hyperparameter.adstock, hyperparameters=channelHyperparams
        )

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
        intercept = coefs["coefficient"].iloc[
            0
        ] 

        # Decomp x
        # Create an empty DataFrame for xDecomp
        xDecomp = pd.DataFrame()

        # Multiply each regressor by its corresponding coefficient
        for name in x.columns:
            # Get the corresponding coefficient for the regressor
            coefficient_value = coefs.loc[coefs["name"] == name, "coefficient"].values
            xDecomp[name] = x[name] * (
                coefficient_value if len(coefficient_value) > 0 else 0
            )

        # Add intercept as the first column
        xDecomp.insert(0, "intercept", intercept)

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
        coefs_media = coefs[sel_coef].set_index("name")[
            "coefficient"
        ]  # Set names for coefs_media

        mediaDecompImmediate = pd.DataFrame(
            {
                name: dt_saturatedImmediate[name] * coefs_media[name]
                for name in coefs_media.index
            }
        )
        mediaDecompCarryover = pd.DataFrame(
            {
                name: dt_saturatedCarryover[name] * coefs_media[name]
                for name in coefs_media.index
            }
        )

        return {
            "xDecompVec": xDecompOut,
            "mediaDecompImmediate": mediaDecompImmediate,
            "mediaDecompCarryover": mediaDecompCarryover,
        }
    
    def _compute_pareto_fronts(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_fronts: str,
        calibration_constraint: float,
    ) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        This method identifies Pareto-optimal solutions based on NRMSE and DECOMP.RSSD
        optimization criteria and assigns them to Pareto fronts.

        Args:
            aggregated_data: Dictionary containing aggregated model results
            pareto_fronts: Number of Pareto fronts to compute or "auto"
            calibration_constraint: Constraint for calibration

        Returns:
            pd.DataFrame: A dataframe of Pareto-optimal solutions with their corresponding front numbers.
        """
        self.logger.info("Computing Pareto fronts")
        resultHypParam = aggregated_data["result_hyp_param"]
        xDecompAgg = aggregated_data["x_decomp_agg"]

        if not self.model_outputs.hyper_fixed:
            self.logger.debug("Processing non-fixed hyperparameters")
            # Filter and group data to calculate coef0
            xDecompAggCoef0 = (
                xDecompAgg[
                    xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends)
                ]
                .groupby("sol_id")["coef"]
                .apply(lambda x: min(x.dropna()) == 0)
            )
            # calculate quantiles
            mape_lift_quantile10 = resultHypParam["mape"].quantile(
                calibration_constraint
            )
            nrmse_quantile90 = resultHypParam["nrmse"].quantile(0.9)
            decomprssd_quantile90 = resultHypParam["decomp.rssd"].quantile(0.9)
            self.logger.debug(f"MAPE lift quantile (10%): {mape_lift_quantile10}")
            self.logger.debug(f"NRMSE quantile (90%): {nrmse_quantile90}")
            self.logger.debug(f"DECOMP.RSSD quantile (90%): {decomprssd_quantile90}")

            # merge resultHypParam with xDecompAggCoef0
            resultHypParam = pd.merge(
                resultHypParam, xDecompAggCoef0, on="sol_id", how="left"
            )
            # create a new column 'mape.qt10'
            resultHypParam["mape.qt10"] = (
                (resultHypParam["mape"] <= mape_lift_quantile10)
                & (resultHypParam["nrmse"] <= nrmse_quantile90)
                & (resultHypParam["decomp.rssd"] <= decomprssd_quantile90)
            )
            # filter resultHypParam
            resultHypParamPareto = resultHypParam[resultHypParam["mape.qt10"] == True]
            self.logger.debug(f"Number of solutions passing constraints: {len(resultHypParamPareto)}")

            # Calculate Pareto front
            self.logger.debug("Calculating Pareto fronts")
            pareto_fronts_df = ParetoOptimizer._pareto_fronts(
                resultHypParamPareto, pareto_fronts=pareto_fronts
            )
            # Merge resultHypParamPareto with pareto_fronts_df
            resultHypParamPareto = pd.merge(
                resultHypParamPareto,
                pareto_fronts_df,
                left_on=["nrmse", "decomp.rssd"],
                right_on=["x", "y"],
            )
            resultHypParamPareto = resultHypParamPareto.rename(
                columns={"pareto_front": "robynPareto"}
            )
            resultHypParamPareto = resultHypParamPareto.sort_values(
                ["iterNG", "iterPar", "nrmse"]
            )[["sol_id", "robynPareto"]]
            resultHypParamPareto = (
                resultHypParamPareto.groupby("sol_id").first().reset_index()
            )
            resultHypParam = pd.merge(
                resultHypParam, resultHypParamPareto, on="sol_id", how="left"
            )
        else:
            self.logger.info("Using fixed hyperparameters")
            resultHypParam = resultHypParam.assign(
                mape_qt10=True, robynPareto=1, coef0=np.nan
            )

        # Calculate combined weighted error scores
        self.logger.debug("Calculating error scores")
        resultHypParam["error_score"] = ParetoUtils.calculate_errors_scores(
            df=resultHypParam, ts_validation=self.model_outputs.ts_validation
        )
        self.logger.info("Pareto front computation completed")
        return resultHypParam
    
    @staticmethod
    def _pareto_fronts(
        resultHypParamPareto: pd.DataFrame, pareto_fronts: str
    ) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        This method identifies Pareto-optimal solutions based on NRMSE and DECOMP.RSSD
        optimization criteria and assigns them to Pareto fronts.

        Args:
            resultHypParamPareto (pd.DataFrame): DataFrame containing model results,
                                                including 'nrmse' and 'decomp.rssd' columns.
            pareto_fronts (Union[str, int]): Number of Pareto fronts to calculate or "auto".
        """
        # Extract vectors like in R
        nrmse = resultHypParamPareto["nrmse"].values
        decomp_rssd = resultHypParamPareto["decomp.rssd"].values

        # Ensure nrmse_values and decomp_rssd_values have the same length
        if len(nrmse) != len(decomp_rssd):
            raise ValueError("Length of nrmse_values must be equal to length of decomp_rssd")

        # Create initial dataframe and sort (equivalent to R's order())
        data = pd.DataFrame({"nrmse": nrmse, "decomp_rssd": decomp_rssd})
        sorted_data = data.sort_values(["nrmse", "decomp_rssd"], ascending=[True, True]).copy()
        
        # Initialize empty dataframe for results
        pareto_fronts_df = pd.DataFrame()
        i = 1
        
        # Convert pareto_fronts to match R's logic
        max_fronts = float('inf') if isinstance(pareto_fronts, str) and "auto" in pareto_fronts else pareto_fronts
        
        # Main loop matching R's while condition
        while len(sorted_data) >= 1 and i <= max_fronts:
            # Calculate cummin (matches R's behavior)
            cummin_mask = ~sorted_data['decomp_rssd'].cummin().duplicated()
            pareto_candidates = sorted_data[cummin_mask].copy()
            pareto_candidates['pareto_front'] = i
            
            # Append to results (equivalent to R's rbind)
            pareto_fronts_df = pd.concat([pareto_fronts_df, pareto_candidates], ignore_index=True)
            
            # Remove processed rows (equivalent to R's row.names logic)
            sorted_data = sorted_data.loc[~sorted_data.index.isin(pareto_candidates.index)].copy()
            i += 1
        
        # Merge results back with original data (equivalent to R's merge)
        result = pd.merge(
            left=data,
            right=pareto_fronts_df[['nrmse', 'decomp_rssd', 'pareto_front']],
            on=['nrmse', 'decomp_rssd'],
            how='left'
        )
        
        # Rename columns to match R output
        result.columns = ['x', 'y', 'pareto_front']
        
        return result.reset_index(drop=True)
