# pyre-strict

from dataclasses import dataclass
from typing import Dict, List
import logging

import numpy as np
import pandas as pd
from robyn.common.logger import RobynLogger
from robyn.data.entities.holidays_data import HolidaysData

from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.pareto.data_aggregator import DataAggregator
from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.modeling.pareto.plot_data_generator import PlotDataGenerator
from robyn.modeling.pareto.response_curve import ResponseCurveCalculator
from robyn.modeling.transformations.transformations import Transformation
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


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
        self.data_aggregator = DataAggregator(model_outputs)

        self.transformer = Transformation(mmm_data)

        self.response_curve_calculator = ResponseCurveCalculator(
            mmm_data=self.mmm_data,
            model_outputs=self.model_outputs,
            hyperparameter=self.hyperparameter,
        )

        self.plot_data_generator = PlotDataGenerator(
            mmm_data=mmm_data,
            hyperparameter=hyperparameter,
            featurized_mmm_data=featurized_mmm_data,
            holidays_data=holidays_data,
        )

        # Setup logger with a single handler
        self.logger = logging.getLogger(__name__)
        # Remove any existing handlers to prevent duplicates
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        # Create a single handler with custom formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Prevent logger from propagating to root logger
        self.logger.propagate = False

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
            aggregated_data = self.data_aggregator.aggregate_model_data(calibrated)
            aggregated_data["result_hyp_param"] = self._compute_pareto_fronts(
                aggregated_data, pareto_fronts, calibration_constraint
            )

            pareto_data = self.prepare_pareto_data(
                aggregated_data, pareto_fronts, min_candidates, calibrated
            )
            pareto_data = self.response_curve_calculator.compute_response_curves(
                pareto_data, aggregated_data
            )
            plotting_data = self.plot_data_generator.generate_plot_data(
                aggregated_data, pareto_data
            )

            self.logger.info("Pareto optimization completed successfully")
            return ParetoResult(
                pareto_solutions=plotting_data["pareto_solutions"],
                pareto_fronts=max(pareto_data.pareto_fronts),
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

    def _determine_pareto_fronts(
        self,
        result_hyp_param: pd.DataFrame,
        pareto_fronts: str,
        min_candidates: int,
        calibrated: bool,
    ) -> int:
        if self.model_outputs.hyper_fixed or len(result_hyp_param) == 1:
            self.logger.info(
                "Using single Pareto front due to fixed hyperparameters or single model"
            )
            return 1

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

            grouped_data = (
                result_hyp_param[result_hyp_param["robynPareto"].notna()]
                .groupby("robynPareto", as_index=False)
                .agg(n=("sol_id", "nunique"))
            )
            grouped_data["n_cum"] = grouped_data["n"].cumsum()
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

            return int(auto_pareto["robynPareto"])

        return int(pareto_fronts)

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

        pareto_fronts = self._determine_pareto_fronts(
            result_hyp_param, pareto_fronts, min_candidates, calibrated
        )
        pareto_fronts_vec = list(range(1, pareto_fronts + 1))
        self.logger.info(f"Selected Pareto fronts: {pareto_fronts+1}")

        # Filtering data for selected Pareto fronts
        self.logger.info("Filtering data for selected Pareto fronts...")
        decomp_spend_dist_pareto = decomp_spend_dist[
            decomp_spend_dist["robynPareto"].isin(pareto_fronts_vec)
        ]
        RobynLogger.log_df(self.logger, decomp_spend_dist_pareto)

        result_hyp_param_pareto = result_hyp_param[
            result_hyp_param["robynPareto"].isin(pareto_fronts_vec)
        ]
        RobynLogger.log_df(self.logger, result_hyp_param_pareto)

        x_decomp_agg_pareto = aggregated_data["x_decomp_agg"][
            aggregated_data["x_decomp_agg"]["robynPareto"].isin(pareto_fronts_vec)
        ]
        RobynLogger.log_df(self.logger, x_decomp_agg_pareto)

        self.logger.info("Pareto data preparation completed")
        return ParetoData(
            decomp_spend_dist=decomp_spend_dist_pareto,
            result_hyp_param=result_hyp_param_pareto,
            x_decomp_agg=x_decomp_agg_pareto,
            pareto_fronts=pareto_fronts_vec,
        )

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
            self.logger.debug(
                f"Number of solutions passing constraints: {len(resultHypParamPareto)}"
            )

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
            raise ValueError(
                "Length of nrmse_values must be equal to length of decomp_rssd"
            )

        # Create initial dataframe and sort (equivalent to R's order())
        data = pd.DataFrame({"nrmse": nrmse, "decomp_rssd": decomp_rssd})
        sorted_data = data.sort_values(
            ["nrmse", "decomp_rssd"], ascending=[True, True]
        ).copy()

        # Initialize empty dataframe for results
        pareto_fronts_df = pd.DataFrame()
        i = 1

        # Convert pareto_fronts to match R's logic
        max_fronts = (
            float("inf")
            if isinstance(pareto_fronts, str) and "auto" in pareto_fronts
            else pareto_fronts
        )

        # Main loop matching R's while condition
        while len(sorted_data) >= 1 and i <= max_fronts:
            # Calculate cummin (matches R's behavior)
            cummin_mask = ~sorted_data["decomp_rssd"].cummin().duplicated()
            pareto_candidates = sorted_data[cummin_mask].copy()
            pareto_candidates["pareto_front"] = i

            # Append to results (equivalent to R's rbind)
            pareto_fronts_df = pd.concat(
                [pareto_fronts_df, pareto_candidates], ignore_index=True
            )

            # Remove processed rows (equivalent to R's row.names logic)
            sorted_data = sorted_data.loc[
                ~sorted_data.index.isin(pareto_candidates.index)
            ].copy()
            i += 1

        # Merge results back with original data (equivalent to R's merge)
        result = pd.merge(
            left=data,
            right=pareto_fronts_df[["nrmse", "decomp_rssd", "pareto_front"]],
            on=["nrmse", "decomp_rssd"],
            how="left",
        )

        # Rename columns to match R output
        result.columns = ["x", "y", "pareto_front"]

        return result.reset_index(drop=True)
