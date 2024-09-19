# pyre-strict

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.pareto.response_curve import ResponseCurveCalculator
from robyn.modeling.pareto.immediate_carryover import ImmediateCarryoverCalculator
from robyn.modeling.pareto.pareto_utils import ParetoUtils


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

    def __init__(self, mmm_data: MMMData, model_outputs: ModelOutputs):
        """
        Initialize the ParetoOptimizer.

        Args:
            mmm_data (MMMData): Input data for the marketing mix model.
            model_outputs (ModelOutputs): Output data from the model runs.
        """
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        self.response_calculator = ResponseCurveCalculator(mmm_data, model_outputs)
        self.carryover_calculator = ImmediateCarryoverCalculator(mmm_data, model_outputs)
        self.pareto_utils = ParetoUtils()

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
        pareto_fronts_df = self._compute_pareto_fronts(aggregated_data, pareto_fronts, min_candidates)
        response_curves = self._compute_response_curves(pareto_fronts_df)
        plot_data = self._generate_plot_data(pareto_fronts_df, response_curves)

        return ParetoResult(
            pareto_solutions=pareto_fronts_df["solID"].tolist(),
            pareto_fronts=len(pareto_fronts_df["pareto_front"].unique()),
            result_hyp_param=aggregated_data["result_hyp_param"],
            x_decomp_agg=aggregated_data["x_decomp_agg"],
            result_calibration=aggregated_data.get("result_calibration"),
            media_vec_collect=response_curves["media_vec_collect"],
            x_decomp_vec_collect=response_curves["x_decomp_vec_collect"],
            plot_data_collect=plot_data,
            df_caov_pct_all=self.carryover_calculator.calculate_all(),
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
        pass

    def _compute_pareto_fronts(
        self, data: Dict[str, pd.DataFrame], pareto_fronts: str, min_candidates: int
    ) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        This method identifies Pareto-optimal solutions based on specified optimization criteria
        (typically NRMSE and DECOMP.RSSD) and assigns them to Pareto fronts.

        Args:
            data (Dict[str, pd.DataFrame]): Aggregated model data from _aggregate_model_data.
            pareto_fronts (str): Number of Pareto fronts to compute or "auto".
            min_candidates (int): Minimum number of candidates when using "auto" Pareto fronts.

        Returns:
            pd.DataFrame: A dataframe of Pareto-optimal solutions with their corresponding front numbers.
        """
        pass

    def _compute_response_curves(self, pareto_fronts_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate response curves for Pareto-optimal solutions.

        This method computes response curves for each media channel in each Pareto-optimal solution,
        providing insights into the relationship between media spend and response.

        Args:
            pareto_fronts_df (pd.DataFrame): Dataframe of Pareto-optimal solutions.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing:
                - 'media_vec_collect': Collected media vectors for all Pareto-optimal solutions
                - 'x_decomp_vec_collect': Collected decomposition vectors for all Pareto-optimal solutions
        """
        pass

    def _generate_plot_data(
        self, pareto_fronts_df: pd.DataFrame, response_curves: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for various plots used in the Pareto analysis.

        This method generates data for different types of plots used to visualize and analyze
        the Pareto-optimal solutions, including spend vs. effect comparisons, waterfalls, and more.

        Args:
            pareto_fronts_df (pd.DataFrame): Dataframe of Pareto-optimal solutions.
            response_curves (Dict[str, pd.DataFrame]): Response curves data from _compute_response_curves.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of dataframes, each containing data for a specific plot type.
        """
        pass
