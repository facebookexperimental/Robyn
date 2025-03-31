from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
import json
import logging
from .entities.allocation_params import AllocatorParams
from .entities.allocation_result import AllocationResult, OptimOutData, MainPoints
from .entities.optimization_result import OptimizationResult
from .entities.constraints import Constraints
from .constants import (
    SCENARIO_MAX_RESPONSE,
    ALGO_SLSQP_AUGLAG,
    CONSTRAINT_MODE_EQ,
    DEFAULT_CONSTRAINT_MULTIPLIER,
    DATE_RANGE_ALL,
)
from datetime import datetime
from .calculate_response import robyn_response, which_usecase
from .checks import check_metric_dates, check_daterange

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.model_refit_output import ModelRefitOutput
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.pareto_result import ParetoResult
from .optimizer import run_optimization, eval_f, eval_g_eq, eval_g_ineq, eval_g_eq_effi


class BudgetAllocator:
    """Budget allocation optimizer for MMM models."""

    def __init__(
        self,
        mmm_data: MMMData,
        featurized_mmm_data: FeaturizedMMMData,
        hyperparameters: Hyperparameters,
        pareto_result: ParetoResult,
        select_model: str,
        params: AllocatorParams,
    ):
        """
        Initialize BudgetAllocator with MMM data and parameters.

        Args:
            mmm_data: MMM data object containing model inputs
            featurized_mmm_data: Processed MMM data
            hyperparameters: Model hyperparameters
            pareto_result: Pareto optimization results
            select_model: Selected model ID
            params: Allocation parameters
        """
        self.logger = logging.getLogger(__name__)
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.hyperparameters = hyperparameters
        self.pareto_result = pareto_result
        self.select_model = select_model
        self.params = params
        self.params.quiet = False

        # Initialize optimization components
        self._setup_local_data_and_params()
        self._setup_spend_values()
        self._calculate_response_values()
        self._set_initial_values_and_bounds()
        self._setup_optimization_bounds_and_channels()
        self._setup_optimization_parameters()

        # Run optimization and get results
        self.dt_optim_out = self._optimize()

        # log json
        self.logger.debug(
            json.dumps(
                {
                    "step": "dt_optimOut",
                    "data": self.dt_optim_out.to_dict("index"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        # Calculate curves and main points
        curves_data = self._calculate_curves(self.dt_optim_out)
        scurve_data = self._prepare_scurve_data(
            curves_data["dt_optim_out_scurve"], curves_data["levs1"]
        )

        # Create and store allocation result
        self.result = AllocationResult(
            dt_optimOut=self.dt_optim_out,
            mainPoints=scurve_data["main_points"],
            scenario=self.params.scenario,
            usecase=self.usecase,
            total_budget=(
                self.params.total_budget
                if self.params.total_budget is not None
                else self.total_budget_window
            ),
            skipped_coef0=self.zero_coef_channel,
            skipped_constr=self.zero_constraint_channel,
            no_spend=self.zero_spend_channel,
        )

    def optimize(self) -> AllocationResult:
        """
        Run the budget allocation optimization.

        Returns:
            AllocationResult object containing optimization results
        """
        # TODO: Implement optimization logic
        raise NotImplementedError("Optimization not implemented yet")

    def _setup_local_data_and_params(self) -> None:
        """Set up local data and parameters for the allocator"""
        # Get paid media spends and sort them
        paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends
        media_order = np.argsort(paid_media_spends)  # equivalent to R's order()
        self.media_spend_sorted = [paid_media_spends[i] for i in media_order]

        # Get dependent variable type
        self.dep_var_type = self.mmm_data.mmmdata_spec.dep_var_type

        # Set up channel constraints
        if self.params.channel_constr_low is None:
            if self.params.scenario == "max_response":
                self.params.channel_constr_low = 0.5
            elif self.params.scenario == "target_efficiency":
                self.params.channel_constr_low = 0.1

        if self.params.channel_constr_up is None:
            if self.params.scenario == "max_response":
                self.params.channel_constr_up = 2
            elif self.params.scenario == "target_efficiency":
                self.params.channel_constr_up = 10

        # Replicate single values if needed
        if isinstance(self.params.channel_constr_low, (int, float)):
            self.params.channel_constr_low = [self.params.channel_constr_low] * len(
                paid_media_spends
            )
        if isinstance(self.params.channel_constr_up, (int, float)):
            self.params.channel_constr_up = [self.params.channel_constr_up] * len(
                paid_media_spends
            )

        # Create Series with media names as index
        self.channel_constr_low = pd.Series(
            self.params.channel_constr_low, index=paid_media_spends
        ).reindex(self.media_spend_sorted)

        self.channel_constr_up = pd.Series(
            self.params.channel_constr_up, index=paid_media_spends
        ).reindex(self.media_spend_sorted)

        # Get model parameters for selected model
        self.dt_hyppar = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["solID"] == self.select_model
        ]

        self.dt_best_coef = self.pareto_result.x_decomp_agg[
            (self.pareto_result.x_decomp_agg["solID"] == self.select_model)
            & (self.pareto_result.x_decomp_agg["rn"].isin(paid_media_spends))
        ]

        # Sort media coefficients
        dt_coef = self.dt_best_coef[["rn", "coef"]].copy()
        get_rn_order = dt_coef["rn"].argsort()
        self.dt_coef_sorted = dt_coef.iloc[get_rn_order].reset_index(drop=True)
        self.dt_best_coef = self.dt_best_coef.iloc[get_rn_order].reset_index(drop=True)

        # Create coefficient selector (coef > 0)
        self.coef_selector_sorted = pd.Series(
            self.dt_coef_sorted["coef"] > 0, index=self.dt_coef_sorted["rn"]
        )

        # Filter hyperparameters for media channels
        hyper_columns = self._get_hyper_names(
            self.hyperparameters.adstock, self.media_spend_sorted
        )
        self.dt_hyppar = self.dt_hyppar[sorted(hyper_columns)]

        # Filter best coefficients for media spend sorted
        self.dt_best_coef = self.dt_best_coef[
            self.dt_best_coef["rn"].isin(self.media_spend_sorted)
        ]

        # Sort channel constraints to match media order
        self.channel_constr_low_sorted = self.channel_constr_low[
            self.media_spend_sorted
        ]
        self.channel_constr_up_sorted = self.channel_constr_up[self.media_spend_sorted]

        # Get hill parameters
        hill_params = self._get_hill_params()
        self.alphas = hill_params["alphas"]
        self.inflexions = hill_params["inflexions"]
        self.coefs_sorted = hill_params["coefs_sorted"]

        # Log all hill parameters
        self.logger.debug(
            json.dumps(
                {
                    "step": "hill_parameters",
                    "data": {
                        "alphas": self.alphas.to_dict(),
                        "inflexions": self.inflexions.to_dict(),
                        "coefs_sorted": self.coefs_sorted.to_dict(),
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    def _setup_spend_values(self) -> None:
        """Set up spend values based on date range"""
        # Get window indices
        window_start = self.mmm_data.mmmdata_spec.rolling_window_start_which
        window_end = self.mmm_data.mmmdata_spec.rolling_window_end_which
        self.window_loc = slice(window_start, window_end + 1)

        # Get optimized cost data for window
        self.dt_optim_cost = self.featurized_mmm_data.dt_mod.iloc[
            window_start : window_end + 1
        ].copy()

        # Check and update date range
        date_range = check_metric_dates(
            date_range=self.params.date_range,
            all_dates=pd.Series(self.dt_optim_cost["ds"]),  # Convert to Series
            day_interval=self.mmm_data.mmmdata_spec.interval_type,
        )
        self.date_min = date_range["date_range_updated"][0]
        self.date_max = date_range["date_range_updated"][-1]

        # Validate date range
        check_daterange(self.date_min, self.date_max, self.dt_optim_cost["ds"])

        # Adjust date range if needed
        if self.date_min is None:
            self.date_min = self.dt_optim_cost["ds"].min()
        if self.date_max is None:
            self.date_max = self.dt_optim_cost["ds"].max()
        if self.date_min < self.dt_optim_cost["ds"].min():
            self.date_min = self.dt_optim_cost["ds"].min()
        if self.date_max > self.dt_optim_cost["ds"].max():
            self.date_max = self.dt_optim_cost["ds"].max()

        # Filter historical data by date range
        self.hist_filtered = self.dt_optim_cost[
            (self.dt_optim_cost["ds"] >= self.date_min)
            & (self.dt_optim_cost["ds"] <= self.date_max)
        ]

        # Calculate historical spend metrics for all data
        spend_cols = self.media_spend_sorted
        self.hist_spend_all = self.dt_optim_cost[spend_cols].sum()
        self.hist_spend_all_total = self.hist_spend_all.sum()
        self.hist_spend_all_unit = self.dt_optim_cost[spend_cols].mean()
        self.hist_spend_all_unit_total = self.hist_spend_all_unit.sum()
        self.hist_spend_all_share = (
            self.hist_spend_all_unit / self.hist_spend_all_unit_total
        )

        # Calculate historical spend metrics for window
        self.hist_spend_window = self.hist_filtered[spend_cols].sum()
        self.hist_spend_window_total = self.hist_spend_window.sum()
        self.hist_spend_window_unit = self.init_spend_unit = self.hist_filtered[
            spend_cols
        ].mean()
        self.hist_spend_window_unit_total = self.hist_spend_window_unit.sum()
        self.hist_spend_window_share = (
            self.hist_spend_window_unit / self.hist_spend_window_unit_total
        )

        # Calculate simulation period
        self.simulation_period = self.initial_mean_period = self.hist_filtered[
            spend_cols
        ].count()
        self.n_dates = {
            media: self.hist_filtered["ds"].values for media in self.media_spend_sorted
        }

        # Print date window info if not quiet
        if not self.params.quiet:
            print(
                f"Date Window: {self.date_min}:{self.date_max} "
                f"({self.initial_mean_period.iloc[0]} {self.mmm_data.mmmdata_spec.interval_type}s)"
            )

        # Identify zero spend channels
        self.zero_spend_channel = self.hist_spend_window[
            self.hist_spend_window == 0
        ].index.tolist()

        # Calculate initial spend metrics
        self.init_spend_unit_total = self.init_spend_unit.sum()
        self.init_spend_share = self.init_spend_unit / self.init_spend_unit_total

        # Calculate total budget
        period_length = self.initial_mean_period.iloc[0]
        self.total_budget_unit = (
            self.init_spend_unit_total
            if self.params.total_budget is None
            else self.params.total_budget / period_length
        )
        self.total_budget_window = self.total_budget_unit * period_length

        # Get use case based on inputs
        self.usecase = which_usecase(
            self.init_spend_unit.iloc[0], self.params.date_range
        )

        # Set ndates_loc based on use case
        if self.usecase == "all_historical_vec":
            self.ndates_loc = self.featurized_mmm_data.dt_mod[
                self.featurized_mmm_data.dt_mod["ds"].isin(self.hist_filtered["ds"])
            ].index.tolist()
        else:
            self.ndates_loc = list(range(len(self.hist_filtered["ds"])))

        # Add budget type to usecase
        budget_type = (
            "+ defined_budget"
            if self.params.total_budget is not None
            else "+ historical_budget"
        )
        self.usecase = f"{self.usecase} {budget_type}"

        if not self.params.quiet:
            print(f"\nUse case: {self.usecase}")

    def _get_hyper_names(self, adstock: str, media_list: List[str]) -> List[str]:
        """Get hyperparameter column names based on adstock type and media list"""
        base_params = ["alphas", "gammas"]
        if adstock == "geometric":
            base_params.append("thetas")
        elif "weibull" in adstock:
            base_params.extend(["shapes", "scales"])

        hyper_names = []
        for media in media_list:
            for param in base_params:
                hyper_names.append(f"{media}_{param}")

        return hyper_names

    def _get_hill_params(self) -> Dict[str, pd.Series]:
        """Get hill parameters for each channel"""
        # Get alpha and gamma parameters
        hill_params_cols = [
            col
            for col in self.dt_hyppar.columns
            if col.endswith("_alphas") or col.endswith("_gammas")
        ]
        hill_hyp_par = self.dt_hyppar[hill_params_cols].iloc[0]  # Get first row

        # Extract alphas and gammas for sorted media WITH suffixes
        alphas = pd.Series(
            [hill_hyp_par[f"{media}_alphas"] for media in self.media_spend_sorted],
            index=[
                f"{media}_alphas" for media in self.media_spend_sorted
            ],  # Keep suffixes
        )
        gammas = pd.Series(
            [hill_hyp_par[f"{media}_gammas"] for media in self.media_spend_sorted],
            index=[
                f"{media}_gammas" for media in self.media_spend_sorted
            ],  # Keep suffixes
        )

        # Get adstocked media data
        chn_adstocked = (
            self.pareto_result.media_vec_collect[
                (self.pareto_result.media_vec_collect["type"] == "adstockedMedia")
                & (self.pareto_result.media_vec_collect["solID"] == self.select_model)
            ][self.media_spend_sorted]
        ).iloc[
            self.mmm_data.mmmdata_spec.rolling_window_start_which : self.mmm_data.mmmdata_spec.rolling_window_end_which
            + 1
        ]

        # Calculate inflexions (with suffixes to match gammas)
        inflexions = pd.Series(
            index=[f"{media}_gammas" for media in self.media_spend_sorted], dtype=float
        )
        for i, media in enumerate(self.media_spend_sorted):
            media_range = chn_adstocked[media].agg(["min", "max"]).values
            gamma = gammas[f"{media}_gammas"]  # Access with suffix
            inflexions[f"{media}_gammas"] = np.dot(media_range, [(1 - gamma), gamma])

        # Get sorted coefficients (no suffix needed here)
        coefs_sorted = pd.Series(
            self.dt_coef_sorted["coef"].values, index=self.dt_coef_sorted["rn"]
        ).reindex(self.media_spend_sorted)

        # Debug prints
        if not self.params.quiet:
            print("\nHill Parameters:")
            for media in self.media_spend_sorted:
                print(f"\n{media}:")
                print(f"  Alpha: {alphas[f'{media}_alphas']:.6f}")
                print(f"  Gamma: {gammas[f'{media}_gammas']:.6f}")
                print(f"  Inflexion: {inflexions[f'{media}_gammas']:.6f}")
                print(f"  Coefficient: {coefs_sorted[media]:.6f}")

        return {
            "alphas": alphas,
            "inflexions": inflexions,
            "coefs_sorted": coefs_sorted,
        }

    def _calculate_response_values(self) -> None:
        """Calculate response values based on date range."""
        # Initialize response values
        self.init_response_unit = pd.Series(index=self.media_spend_sorted)
        self.init_response_marg_unit = pd.Series(index=self.media_spend_sorted)
        self.hist_carryover = {}
        self.qa_carryover = {}

        # Calculate response for each media channel
        for media in self.media_spend_sorted:
            response = robyn_response(
                mmm_data=self.mmm_data,
                featurized_mmm_data=self.featurized_mmm_data,
                pareto_result=self.pareto_result,
                hyperparameters=self.hyperparameters,
                select_model=self.select_model,
                metric_name=media,
                dt_hyppar=self.pareto_result.result_hyp_param,
                dt_coef=self.pareto_result.x_decomp_agg,
            )

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Log response values
            def convert_to_serializable(obj):
                if isinstance(obj, pd.Series):
                    return obj.tolist()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return str(obj)
                if isinstance(obj, (list, tuple)):
                    return [
                        str(x) if isinstance(x, (pd.Timestamp, datetime)) else x
                        for x in obj
                    ]
                if isinstance(obj, (dict_keys, dict_values)):
                    return list(obj)
                return obj

            # Get keys and types info
            keys_info = {
                key: {
                    "type": str(type(value).__name__),
                    "shape": (
                        value.shape
                        if hasattr(value, "shape")
                        else len(value) if hasattr(value, "__len__") else "scalar"
                    ),
                }
                for key, value in response.items()
            }

            # self.logger.debug(
            #     json.dumps(
            #         {
            #             "step": "response_values",
            #             "keys_info": keys_info,
            #             "data": {
            #                 "response_cols": list(response.keys()),  # Convert to list
            #                 "metric_name": response["metric_name"],
            #                 "date": convert_to_serializable(response["date"]),
            #                 "input_total": convert_to_serializable(
            #                     response["input_total"]
            #                 ),
            #                 "input_carryover": convert_to_serializable(
            #                     response["input_carryover"]
            #                 ),
            #                 "input_immediate": convert_to_serializable(
            #                     response["input_immediate"]
            #                 ),
            #                 "response_total": convert_to_serializable(
            #                     response["response_total"]
            #                 ),
            #                 "response_carryover": convert_to_serializable(
            #                     response["response_carryover"]
            #                 ),
            #                 "response_immediate": convert_to_serializable(
            #                     response["response_immediate"]
            #                 ),
            #                 "usecase": response["usecase"],
            #             },
            #             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #         },
            #         indent=2,
            #     )
            # )
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # Store carryover values with dates as index
            hist_carryover_temp = response["input_carryover"][self.window_loc]
            if isinstance(hist_carryover_temp, np.ndarray):
                hist_carryover_temp = pd.Series(
                    hist_carryover_temp, index=response["date"][self.window_loc]
                )
            self.hist_carryover[media] = hist_carryover_temp

            # Store QA carryover values (rounded total input for the window)
            self.qa_carryover[media] = np.round(
                response["input_total"][self.window_loc]
            )

            # Use initial spend unit for x_input
            x_input = self.init_spend_unit[media]

            resp_simulate = self._fx_objective(
                x=x_input,
                coeff=self.coefs_sorted[media],
                alpha=self.alphas[f"{media}_alphas"],
                inflexion=self.inflexions[f"{media}_gammas"],
                x_hist_carryover=hist_carryover_temp.mean(),
                get_sum=False,
            )

            resp_simulate_plus1 = self._fx_objective(
                x=x_input + 1,
                coeff=self.coefs_sorted[media],
                alpha=self.alphas[f"{media}_alphas"],
                inflexion=self.inflexions[f"{media}_gammas"],
                x_hist_carryover=hist_carryover_temp.mean(),
                get_sum=False,
            )

            # Store response values
            self.init_response_unit[media] = resp_simulate
            self.init_response_marg_unit[media] = resp_simulate_plus1 - resp_simulate

            if not self.params.quiet:
                print(f"\nResponse values for {media}:")
                print(f"  Response Unit: {self.init_response_unit[media]:.6f}")
                print(
                    f"  Response Marginal Unit: {self.init_response_marg_unit[media]:.6f}"
                )

        # Convert qa_carryover to DataFrame
        self.qa_carryover = pd.DataFrame(self.qa_carryover)

        # Handle zero spend channels
        zero_spend_channels = [
            media
            for media in self.media_spend_sorted
            if self.init_spend_unit[media] == 0
        ]
        if zero_spend_channels and not self.params.quiet:
            print(
                f"\nMedia variables with 0 spending during date range: {', '.join(zero_spend_channels)}"
            )

    def _fx_objective(
        self,
        x: Union[float, np.ndarray],
        coeff: float,
        alpha: float,
        inflexion: float,
        x_hist_carryover: Union[float, np.ndarray],
        get_sum: bool = True,
    ) -> float:
        """Calculate objective function value using Hill transformation.

        Args:
            x: Input value(s) for optimization
            coeff: Coefficient for scaling
            alpha: Hill function alpha parameter
            inflexion: Hill function inflexion point
            x_hist_carryover: Historical carryover values
            get_sum: Whether to sum the transformed values

        Returns:
            Transformed value(s) after Hill transformation
        """
        # Commented out Michaelis-Menten transformation as in R code
        # if criteria:
        #     x_scaled = mic_men(x=x, Vmax=vmax, Km=km)  # vmax * x / (km + x)
        # elif chn_name in mm_lm_coefs:
        #     x_scaled = x * mm_lm_coefs[chn_name]
        # else:
        #     x_scaled = x

        # Adstock scales
        x_adstocked = x + np.mean(x_hist_carryover)

        # Hill transformation
        if get_sum:
            x_out = coeff * np.sum((1 + inflexion**alpha / x_adstocked**alpha) ** -1)
        else:
            x_out = coeff * ((1 + inflexion**alpha / x_adstocked**alpha) ** -1)

        return x_out

    def _set_initial_values_and_bounds(self):
        """Set initial values and bounds for optimization."""

        # Calculate extended constraints
        channel_constr_low_sorted_ext = np.where(
            1
            - (1 - self.channel_constr_low_sorted)
            * self.params.channel_constr_multiplier
            < 0,
            0,
            1
            - (1 - self.channel_constr_low_sorted)
            * self.params.channel_constr_multiplier,
        )

        channel_constr_up_sorted_ext = np.where(
            1
            + (self.channel_constr_up_sorted - 1)
            * self.params.channel_constr_multiplier
            < 0,
            self.channel_constr_up_sorted * self.params.channel_constr_multiplier,
            1
            + (self.channel_constr_up_sorted - 1)
            * self.params.channel_constr_multiplier,
        )

        # Store extended constraints
        self.channel_constr_low_sorted_ext = channel_constr_low_sorted_ext
        self.channel_constr_up_sorted_ext = channel_constr_up_sorted_ext

        # Set target value extension
        self.target_value_ext = self.params.target_value
        if self.params.scenario == "target_efficiency":
            # Reset extended constraints for target efficiency
            self.channel_constr_low_sorted_ext = self.channel_constr_low_sorted
            self.channel_constr_up_sorted_ext = self.channel_constr_up_sorted

            if self.mmm_data.mmmdata_spec.dep_var_type == "conversion":
                if self.params.target_value is None:
                    self.params.target_value = (
                        np.sum(self.init_spend_unit) / np.sum(self.init_response_unit)
                    ) * 1.2
                self.target_value_ext = self.params.target_value * 1.5
            else:
                if self.params.target_value is None:
                    self.params.target_value = (
                        np.sum(self.init_response_unit) / np.sum(self.init_spend_unit)
                    ) * 0.8
                self.target_value_ext = 1

        # Set initial values
        self.temp_init = self.init_spend_unit.copy()
        self.temp_init_all = self.init_spend_unit.copy()

        # If no spend within window as initial spend, use historical average
        if len(self.zero_spend_channel) > 0:
            for channel in self.zero_spend_channel:
                self.temp_init_all[channel] = self.hist_spend_all_unit[channel]

    def _setup_optimization_bounds_and_channels(self):
        """Setup optimization bounds and handle channel exclusions."""

        # Convert to numpy arrays if needed
        self.media_spend_sorted = np.array(self.media_spend_sorted)
        self.channel_constr_low_sorted = np.array(self.channel_constr_low_sorted)
        self.channel_constr_up_sorted = np.array(self.channel_constr_up_sorted)

        # Set initial bounds
        self.temp_ub = self.temp_ub_all = self.channel_constr_up_sorted.copy()
        self.temp_lb = self.temp_lb_all = self.channel_constr_low_sorted.copy()
        self.temp_ub_ext = self.temp_ub_ext_all = (
            self.channel_constr_up_sorted_ext.copy()
        )
        self.temp_lb_ext = self.temp_lb_ext_all = (
            self.channel_constr_low_sorted_ext.copy()
        )

        # Log initial bounds
        self.logger.debug(
            json.dumps(
                {
                    "step": "initial_bounds",
                    "data": {
                        "temp_ub": self.temp_ub.tolist(),
                        "temp_lb": self.temp_lb.tolist(),
                        "temp_ub_ext": self.temp_ub_ext.tolist(),
                        "temp_lb_ext": self.temp_lb_ext.tolist(),
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

        # Calculate initial bounds
        self.x0 = self.x0_all = self.lb = self.lb_all = (
            self.temp_init_all * self.temp_lb_all
        )
        self.ub = self.ub_all = self.temp_init_all * self.temp_ub_all
        self.x0_ext = self.x0_ext_all = self.lb_ext = self.lb_ext_all = (
            self.temp_init_all * self.temp_lb_ext_all
        )
        self.ub_ext = self.ub_ext_all = self.temp_init_all * self.temp_ub_ext_all
        # Log calculated bounds
        self.logger.debug(
            json.dumps(
                {
                    "step": "calculated_bounds",
                    "data": {
                        "x0": self.x0.tolist(),
                        "lb": self.lb.tolist(),
                        "ub": self.ub.tolist(),
                        "x0_ext": self.x0_ext.tolist(),
                        "lb_ext": self.lb_ext.tolist(),
                        "ub_ext": self.ub_ext.tolist(),
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

        # Identify channels to exclude
        skip_these = (self.channel_constr_low_sorted == 0) & (
            self.channel_constr_up_sorted == 0
        )

        self.zero_constraint_channel = self.media_spend_sorted[skip_these]

        # Handle zero coefficient channels
        if not all(self.coef_selector_sorted) and not self.params.keep_zero_coefs:
            self.zero_coef_channel = [
                channel
                for channel in self.media_spend_sorted
                if channel not in self.media_spend_sorted[self.coef_selector_sorted]
            ]
            if not self.params.quiet:
                print(
                    f"Excluded variables (coefficients are 0): {', '.join(self.zero_coef_channel)}"
                )
        else:
            self.zero_coef_channel = []

        # Determine channels to drop and keep for allocation
        channels_to_drop = np.isin(
            self.media_spend_sorted,
            np.concatenate([self.zero_coef_channel, self.zero_constraint_channel]),
        )
        self.channel_for_allocation = self.media_spend_sorted[~channels_to_drop]

        # Update bounds if any channels are dropped
        if any(channels_to_drop):
            self.temp_init = self.temp_init_all[self.channel_for_allocation]
            self.temp_ub = self.temp_ub_all[self.channel_for_allocation]
            self.temp_lb = self.temp_lb_all[self.channel_for_allocation]
            self.temp_ub_ext = self.temp_ub_ext_all[self.channel_for_allocation]
            self.temp_lb_ext = self.temp_lb_ext_all[self.channel_for_allocation]
            self.x0 = self.x0_all[self.channel_for_allocation]
            self.lb = self.lb_all[self.channel_for_allocation]
            self.ub = self.ub_all[self.channel_for_allocation]
            self.x0_ext = self.x0_ext_all[self.channel_for_allocation]
            self.lb_ext = self.lb_ext_all[self.channel_for_allocation]
            self.ub_ext = self.ub_ext_all[self.channel_for_allocation]

        # Final bound calculations
        self.x0 = self.lb = self.temp_init * self.temp_lb
        self.ub = self.temp_init * self.temp_ub
        self.x0_ext = self.lb_ext = self.temp_init * self.temp_lb_ext
        self.ub_ext = self.temp_init * self.temp_ub_ext
        # Log final bounds after dropping channels
        self.logger.debug(
            json.dumps(
                {
                    "step": "final_bounds",
                    "data": {
                        "channels": list(self.channel_for_allocation),
                        "temp_init": self.temp_init.tolist(),
                        "temp_ub": self.temp_ub.tolist(),
                        "temp_lb": self.temp_lb.tolist(),
                        "x0": self.x0.tolist(),
                        "lb": self.lb.tolist(),
                        "ub": self.ub.tolist(),
                        "x0_ext": self.x0_ext.tolist(),
                        "lb_ext": self.lb_ext.tolist(),
                        "ub_ext": self.ub_ext.tolist(),
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    def _setup_optimization_parameters(self):
        """Setup optimization parameters and options for nloptr."""

        # Gather values for optimization - keep the suffixes as in R
        self.coefs_eval = {
            channel: self.coefs_sorted[channel]
            for channel in self.channel_for_allocation
        }

        # Keep the full parameter names with suffixes
        self.alphas_eval = self.alphas  # Already has _alphas suffix
        self.inflexions_eval = self.inflexions  # Already has _gammas suffix

        self.hist_carryover_eval = {
            channel: self.hist_carryover[channel]
            for channel in self.channel_for_allocation
        }
        # Create evaluation dictionary
        self.eval_dict = {
            "coefs_eval": self.coefs_eval,
            "alphas_eval": self.alphas_eval,
            "inflexions_eval": self.inflexions_eval,
            "total_budget": self.params.total_budget,
            "total_budget_unit": self.total_budget_unit,
            "hist_carryover_eval": self.hist_carryover_eval,
            "target_value": self.params.target_value,
            "target_value_ext": self.target_value_ext,
            "dep_var_type": self.mmm_data.mmmdata_spec.dep_var_type,
        }

        # Set optimization options based on algorithm
        if self.params.optim_algo == "MMA_AUGLAG":
            self.local_opts = {"algorithm": "NLOPT_LD_MMA", "xtol_rel": 1.0e-10}
        elif self.params.optim_algo == "SLSQP_AUGLAG":
            self.local_opts = {"algorithm": "NLOPT_LD_SLSQP", "xtol_rel": 1.0e-10}
        else:
            raise ValueError(
                f"Unsupported optimization algorithm: {self.params.optim_algo}"
            )
        # Log the dictionary with converted values
        self.logger.debug(
            json.dumps(
                {
                    "step": "eval_dict",
                    "data": {
                        "coefs_eval": (
                            list(self.coefs_eval.values())
                            if isinstance(self.coefs_eval, dict)
                            else self.coefs_eval.tolist()
                        ),
                        "alphas_eval": (
                            list(self.alphas_eval.values())
                            if isinstance(self.alphas_eval, dict)
                            else self.alphas_eval.tolist()
                        ),
                        "inflexions_eval": (
                            list(self.inflexions_eval.values())
                            if isinstance(self.inflexions_eval, dict)
                            else self.inflexions_eval.tolist()
                        ),
                        "total_budget": self.params.total_budget,
                        "total_budget_unit": float(self.total_budget_unit),
                        "hist_carryover_eval": {
                            k: v.tolist() if hasattr(v, "tolist") else float(v)
                            for k, v in self.hist_carryover_eval.items()
                        },
                        "target_value": self.params.target_value,
                        "target_value_ext": self.target_value_ext,
                        "dep_var_type": self.mmm_data.mmmdata_spec.dep_var_type,
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    def _optimize(self):
        """Run optimization matching R's implementation"""
        # Calculate mean carryover
        x_hist_carryover = {k: np.mean(v) for k, v in self.hist_carryover_eval.items()}

        # Run optimization based on scenario
        if self.params.scenario == "max_response":
            # Bounded optimization
            nls_mod = run_optimization(
                x0=self.x0,
                eval_f=lambda x, grad: eval_f(x, self.eval_dict),
                eval_g_eq=(
                    (lambda x, grad: eval_g_eq(x, self.eval_dict))
                    if self.params.constr_mode == "eq"
                    else None
                ),
                eval_g_ineq=(
                    (lambda x, grad: eval_g_ineq(x, self.eval_dict))
                    if self.params.constr_mode == "ineq"
                    else None
                ),
                lb=self.lb,
                ub=self.ub,
                maxeval=self.params.maxeval,
                local_opts=self.local_opts,
                target_value=None,
                eval_dict=self.eval_dict,
            )
            # Log bounded optimization results
            self.logger.debug(
                json.dumps(
                    {
                        "step": "bounded_optimization_results",
                        "data": {
                            "solution": nls_mod["solution"].tolist(),
                            "objective": float(nls_mod["objective"]),
                            "status": nls_mod.get("status", "unknown"),
                            "message": nls_mod.get("message", "no message"),
                            "total_spend": float(np.sum(nls_mod["solution"])),
                        },
                    },
                    indent=2,
                )
            )
            # Unbounded optimization
            nls_mod_unbound = run_optimization(
                x0=self.x0_ext,
                eval_f=lambda x, grad: eval_f(x, self.eval_dict),
                eval_g_eq=(
                    (lambda x, grad: eval_g_eq(x, self.eval_dict))
                    if self.params.constr_mode == "eq"
                    else None
                ),
                eval_g_ineq=(
                    (lambda x, grad: eval_g_ineq(x, self.eval_dict))
                    if self.params.constr_mode == "ineq"
                    else None
                ),
                lb=self.lb_ext,
                ub=self.ub_ext,
                maxeval=self.params.maxeval,
                local_opts=self.local_opts,
                target_value=None,
                eval_dict=self.eval_dict,
            )
            # Log unbounded optimization results
            self.logger.debug(
                json.dumps(
                    {
                        "step": "unbounded_optimization_results",
                        "data": {
                            "solution": nls_mod_unbound["solution"].tolist(),
                            "objective": float(nls_mod_unbound["objective"]),
                            "status": nls_mod_unbound.get("status", "unknown"),
                            "message": nls_mod_unbound.get("message", "no message"),
                            "total_spend": float(np.sum(nls_mod_unbound["solution"])),
                        },
                    },
                    indent=2,
                )
            )
        elif self.params.scenario == "target_efficiency":
            # Bounded optimization
            nls_mod = run_optimization(
                x0=self.x0,
                eval_f=lambda x, grad: eval_f(x, self.eval_dict),
                eval_g_eq=(
                    (lambda x, grad: eval_g_eq(x, self.eval_dict))
                    if self.params.constr_mode == "eq"
                    else None
                ),
                eval_g_ineq=(
                    (lambda x, grad: eval_g_ineq(x, self.eval_dict))
                    if self.params.constr_mode == "ineq"
                    else None
                ),
                lb=self.lb,
                ub=self.x0 * self.channel_constr_up_sorted[0],
                maxeval=self.params.maxeval,
                local_opts=self.local_opts,
                target_value=self.params.target_value,
                eval_dict=self.eval_dict,
            )

            # Unbounded optimization
            nls_mod_unbound = run_optimization(
                x0=self.x0,
                eval_f=lambda x, grad: eval_f(x, self.eval_dict),
                eval_g_eq=(
                    (lambda x, grad: eval_g_eq(x, self.eval_dict))
                    if self.params.constr_mode == "eq"
                    else None
                ),
                eval_g_ineq=(
                    (lambda x, grad: eval_g_ineq(x, self.eval_dict))
                    if self.params.constr_mode == "ineq"
                    else None
                ),
                lb=self.lb,
                ub=self.x0 * self.channel_constr_up_sorted[0],
                maxeval=self.params.maxeval,
                local_opts=self.local_opts,
                target_value=self.target_value_ext,
                eval_dict=self.eval_dict,
            )

        # Get marginal responses
        optm_spend_unit = nls_mod["solution"]
        optm_response_unit = eval_f(optm_spend_unit, self.eval_dict)[
            "objective_channel"
        ]
        optm_spend_unit_unbound = nls_mod_unbound["solution"]
        optm_response_unit_unbound = eval_f(optm_spend_unit_unbound, self.eval_dict)[
            "objective_channel"
        ]
        # Debug logging for facebook_S (assuming it's first channel)
        channel_idx = 0
        channel = self.channel_for_allocation[channel_idx]
        self.logger.debug(
            json.dumps(
                {
                    "step": "marginal_response_debug",
                    "channel": channel,
                    "bounded": {
                        "spend": float(optm_spend_unit[channel_idx]),
                        "response": float(optm_response_unit[channel_idx]),
                        "response_at_x_plus_1": float(
                            self._fx_objective(
                                x=optm_spend_unit[channel_idx] + 1,
                                coeff=self.coefs_eval[channel],
                                alpha=self.alphas_eval[f"{channel}_alphas"],
                                inflexion=self.inflexions_eval[f"{channel}_gammas"],
                                x_hist_carryover=x_hist_carryover[channel],
                                get_sum=False,
                            )
                        ),
                    },
                    "unbounded": {
                        "spend": float(optm_spend_unit_unbound[channel_idx]),
                        "response": float(optm_response_unit_unbound[channel_idx]),
                        "response_at_x_plus_1": float(
                            self._fx_objective(
                                x=optm_spend_unit_unbound[channel_idx] + 1,
                                coeff=self.coefs_eval[channel],
                                alpha=self.alphas_eval[f"{channel}_alphas"],
                                inflexion=self.inflexions_eval[f"{channel}_gammas"],
                                x_hist_carryover=x_hist_carryover[channel],
                                get_sum=False,
                            )
                        ),
                    },
                },
                indent=2,
            )
        )

        # Calculate marginal responses for bounded
        optm_response_marg_unit = (
            np.array(
                [
                    self._fx_objective(
                        x=x + 1,
                        coeff=self.coefs_eval[channel],
                        alpha=self.alphas_eval[f"{channel}_alphas"],
                        inflexion=self.inflexions_eval[f"{channel}_gammas"],
                        x_hist_carryover=x_hist_carryover[channel],
                        get_sum=False,
                    )
                    for x, channel in zip(optm_spend_unit, self.channel_for_allocation)
                ]
            )
            - optm_response_unit
        )

        # Calculate marginal responses for unbounded
        optm_response_marg_unit_unbound = (
            np.array(
                [
                    self._fx_objective(
                        x=x + 1,
                        coeff=self.coefs_eval[channel],
                        alpha=self.alphas_eval[f"{channel}_alphas"],
                        inflexion=self.inflexions_eval[f"{channel}_gammas"],
                        x_hist_carryover=x_hist_carryover[channel],
                        get_sum=False,
                    )
                    for x, channel in zip(
                        optm_spend_unit_unbound, self.channel_for_allocation
                    )
                ]
            )
            - optm_response_unit_unbound
        )

        # Collect and organize output
        # First set names for optimized arrays
        optm_spend_unit = pd.Series(optm_spend_unit, index=self.channel_for_allocation)
        optm_response_unit = pd.Series(
            optm_response_unit, index=self.channel_for_allocation
        )
        optm_response_marg_unit = pd.Series(
            optm_response_marg_unit, index=self.channel_for_allocation
        )
        optm_spend_unit_unbound = pd.Series(
            optm_spend_unit_unbound, index=self.channel_for_allocation
        )
        optm_response_unit_unbound = pd.Series(
            optm_response_unit_unbound, index=self.channel_for_allocation
        )
        optm_response_marg_unit_unbound = pd.Series(
            optm_response_marg_unit_unbound, index=self.channel_for_allocation
        )

        # Verify channel mapping
        channels_mapped = np.isin(self.media_spend_sorted, optm_spend_unit.index)

        # Initialize output arrays with initial spend values
        optm_spend_unit_out = pd.Series(self.init_spend_unit.copy())
        optm_response_unit_out = pd.Series(self.init_spend_unit.copy())
        optm_response_marg_unit_out = pd.Series(self.init_spend_unit.copy())
        optm_spend_unit_unbound_out = pd.Series(self.init_spend_unit.copy())
        optm_response_unit_unbound_out = pd.Series(self.init_spend_unit.copy())
        optm_response_marg_unit_unbound_out = pd.Series(self.init_spend_unit.copy())

        # Set dropped channels to 0
        channels_to_drop = np.isin(
            self.media_spend_sorted,
            np.concatenate([self.zero_coef_channel, self.zero_constraint_channel]),
        )

        for series in [
            optm_spend_unit_out,
            optm_response_unit_out,
            optm_response_marg_unit_out,
            optm_spend_unit_unbound_out,
            optm_response_unit_unbound_out,
            optm_response_marg_unit_unbound_out,
        ]:
            series[channels_to_drop] = 0

        # Fill in optimized values for non-dropped channels
        optm_spend_unit_out[~channels_to_drop] = optm_spend_unit
        optm_response_unit_out[~channels_to_drop] = optm_response_unit
        optm_response_marg_unit_out[~channels_to_drop] = optm_response_marg_unit
        optm_spend_unit_unbound_out[~channels_to_drop] = optm_spend_unit_unbound
        optm_response_unit_unbound_out[~channels_to_drop] = optm_response_unit_unbound
        optm_response_marg_unit_unbound_out[~channels_to_drop] = (
            optm_response_marg_unit_unbound
        )

        optim_results = {
            "nls_mod": nls_mod,
            "nls_mod_unbound": nls_mod_unbound,
            "optm_spend_unit": optm_spend_unit_out,
            "optm_spend_unit_unbound": optm_spend_unit_unbound_out,
            "optm_response_unit": optm_response_unit_out,
            "optm_response_unit_unbound": optm_response_unit_unbound_out,
            "optm_response_marg_unit": optm_response_marg_unit_out,
            "optm_response_marg_unit_unbound": optm_response_marg_unit_unbound_out,
            "channels_mapped": channels_mapped,
        }

        # Build output DataFrame
        dt_optim_out = self._build_optim_output(optim_results)

        return dt_optim_out

    def _build_optim_output(self, optim_results):
        """Build optimization output DataFrame matching R's structure"""

        # Get unique simulation period
        unique_sim_period = self.simulation_period.iloc[0]

        # Create output DataFrame
        dt_optim_out = pd.DataFrame(
            {
                "solID": self.select_model,
                "dep_var_type": self.mmm_data.mmmdata_spec.dep_var_type,
                "channels": self.media_spend_sorted,
                "date_min": self.date_min,
                "date_max": self.date_max,
                "periods": f"{unique_sim_period} {self.mmm_data.mmmdata_spec.interval_type}s",
                # Constraints
                "constr_low": self.temp_lb_all,
                "constr_low_abs": self.lb_all,
                "constr_up": self.temp_ub_all,
                "constr_up_abs": self.ub_all,
                "unconstr_mult": self.params.channel_constr_multiplier,
                "constr_low_unb": self.temp_lb_ext_all,
                "constr_low_unb_abs": self.lb_ext_all,
                "constr_up_unb": self.temp_ub_ext_all,
                "constr_up_unb_abs": self.ub_ext_all,
                # Historical spends
                "histSpendAll": self.hist_spend_all,
                "histSpendAllTotal": self.hist_spend_all_total,
                "histSpendAllUnit": self.hist_spend_all_unit,
                "histSpendAllUnitTotal": self.hist_spend_all_unit_total,
                "histSpendAllShare": self.hist_spend_all_share,
                "histSpendWindow": self.hist_spend_window,
                "histSpendWindowTotal": self.hist_spend_window_total,
                "histSpendWindowUnit": self.hist_spend_window_unit,
                "histSpendWindowUnitTotal": self.hist_spend_window_unit_total,
                "histSpendWindowShare": self.hist_spend_window_share,
                # Initial spends
                "initSpendUnit": self.init_spend_unit,
                "initSpendUnitTotal": self.init_spend_unit_total,
                "initSpendShare": self.init_spend_share,
                "initSpendTotal": self.init_spend_unit_total * unique_sim_period,
                # Initial responses
                "initResponseUnit": self.init_response_unit,
                "initResponseUnitTotal": self.init_response_unit.sum(),
                "initResponseMargUnit": self.init_response_marg_unit,
                "initResponseTotal": self.init_response_unit.sum() * unique_sim_period,
                "initResponseUnitShare": self.init_response_unit
                / self.init_response_unit.sum(),
                "initRoiUnit": self.init_response_unit / self.init_spend_unit,
                "initCpaUnit": self.init_spend_unit / self.init_response_unit,
                # Budget
                "total_budget_unit": self.total_budget_unit,
                "total_budget_unit_delta": self.total_budget_unit
                / self.init_spend_unit_total
                - 1,
                # Optimized results - bounded
                "optmSpendUnit": optim_results["optm_spend_unit"],
                "optmSpendUnitDelta": (
                    optim_results["optm_spend_unit"] / self.init_spend_unit - 1
                ),
                "optmSpendUnitTotal": optim_results["optm_spend_unit"].sum(),
                "optmSpendUnitTotalDelta": optim_results["optm_spend_unit"].sum()
                / self.init_spend_unit_total
                - 1,
                "optmSpendShareUnit": optim_results["optm_spend_unit"]
                / optim_results["optm_spend_unit"].sum(),
                "optmSpendTotal": optim_results["optm_spend_unit"].sum()
                * unique_sim_period,
                # Optimized results - unbounded
                "optmSpendUnitUnbound": optim_results["optm_spend_unit_unbound"],
                "optmSpendUnitDeltaUnbound": (
                    optim_results["optm_spend_unit_unbound"] / self.init_spend_unit - 1
                ),
                "optmSpendUnitTotalUnbound": optim_results[
                    "optm_spend_unit_unbound"
                ].sum(),
                "optmSpendUnitTotalDeltaUnbound": optim_results[
                    "optm_spend_unit_unbound"
                ].sum()
                / self.init_spend_unit_total
                - 1,
                "optmSpendShareUnitUnbound": optim_results["optm_spend_unit_unbound"]
                / optim_results["optm_spend_unit_unbound"].sum(),
                "optmSpendTotalUnbound": optim_results["optm_spend_unit_unbound"].sum()
                * unique_sim_period,
                # Response metrics - bounded
                "optmResponseUnit": optim_results["optm_response_unit"],
                "optmResponseMargUnit": optim_results["optm_response_marg_unit"],
                "optmResponseUnitTotal": optim_results["optm_response_unit"].sum(),
                "optmResponseTotal": optim_results["optm_response_unit"].sum()
                * unique_sim_period,
                "optmResponseUnitShare": optim_results["optm_response_unit"]
                / optim_results["optm_response_unit"].sum(),
                "optmRoiUnit": optim_results["optm_response_unit"]
                / optim_results["optm_spend_unit"],
                "optmCpaUnit": optim_results["optm_spend_unit"]
                / optim_results["optm_response_unit"],
                "optmResponseUnitLift": (
                    optim_results["optm_response_unit"] / self.init_response_unit
                )
                - 1,
                # Response metrics - unbounded
                "optmResponseUnitUnbound": optim_results["optm_response_unit_unbound"],
                "optmResponseMargUnitUnbound": optim_results[
                    "optm_response_marg_unit_unbound"
                ],
                "optmResponseUnitTotalUnbound": optim_results[
                    "optm_response_unit_unbound"
                ].sum(),
                "optmResponseTotalUnbound": optim_results[
                    "optm_response_unit_unbound"
                ].sum()
                * unique_sim_period,
                "optmResponseUnitShareUnbound": optim_results[
                    "optm_response_unit_unbound"
                ]
                / optim_results["optm_response_unit_unbound"].sum(),
                "optmRoiUnitUnbound": optim_results["optm_response_unit_unbound"]
                / optim_results["optm_spend_unit_unbound"],
                "optmCpaUnitUnbound": optim_results["optm_spend_unit_unbound"]
                / optim_results["optm_response_unit_unbound"],
                "optmResponseUnitLiftUnbound": (
                    optim_results["optm_response_unit_unbound"]
                    / self.init_response_unit
                )
                - 1,
            }
        )

        # Add calculated fields
        dt_optim_out["optmResponseUnitTotalLift"] = (
            dt_optim_out["optmResponseUnitTotal"]
            / dt_optim_out["initResponseUnitTotal"]
        ) - 1
        dt_optim_out["optmResponseUnitTotalLiftUnbound"] = (
            dt_optim_out["optmResponseUnitTotalUnbound"]
            / dt_optim_out["initResponseUnitTotal"]
        ) - 1

        return dt_optim_out

    def _calculate_curves(self, dt_optim_out: pd.DataFrame) -> Dict:
        """Calculate curves and main points for each channel."""

        # Set level names based on scenario
        if self.params.scenario == "max_response":
            levs1 = [
                "Initial",
                "Bounded",
                f"Bounded x{self.params.channel_constr_multiplier}",
            ]
        elif self.params.scenario == "target_efficiency":
            if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                levs1 = [
                    "Initial",
                    f"Hit ROAS {round(self.params.target_value, 2)}",
                    f"Hit ROAS {self.target_value_ext}",
                ]
            else:
                levs1 = [
                    "Initial",
                    f"Hit CPA {round(self.params.target_value, 2)}",
                    f"Hit CPA {round(self.target_value_ext, 2)}",
                ]

        # Create DataFrame for S-curve
        dt_optim_out_scurve = pd.concat(
            [
                # Initial points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": dt_optim_out["initSpendUnit"],
                        "response": dt_optim_out["initResponseUnit"],
                        "type": levs1[0],
                    }
                ),
                # Bounded optimization points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": dt_optim_out["optmSpendUnit"],
                        "response": dt_optim_out["optmResponseUnit"],
                        "type": levs1[1],
                    }
                ),
                # Unbounded optimization points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": dt_optim_out["optmSpendUnitUnbound"],
                        "response": dt_optim_out["optmResponseUnitUnbound"],
                        "type": levs1[2],
                    }
                ),
                # Carryover points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": 0,
                        "response": 0,
                        "type": "Carryover",
                    }
                ),
            ]
        ).assign(
            spend=lambda x: pd.to_numeric(x["spend"]),
            response=lambda x: pd.to_numeric(x["response"]),
        )  # Remove the groupby here

        return {"dt_optim_out_scurve": dt_optim_out_scurve, "levs1": levs1}

    def _prepare_scurve_data(
        self, dt_optim_out_scurve: pd.DataFrame, levs1: List[str]
    ) -> Dict:
        """Prepare S-curve data for plotting."""

        plot_dt_scurve = []

        # Process each channel
        for channel in self.channel_for_allocation:
            # Get carryover vector for this channel
            carryover_vec = self.hist_carryover_eval[channel]
            carryover_mean = np.mean(carryover_vec)

            # Update spend values with carryover
            mask_levs = dt_optim_out_scurve["type"].isin(levs1) & (
                dt_optim_out_scurve["channels"] == channel
            )
            mask_carryover = (dt_optim_out_scurve["type"] == "Carryover") & (
                dt_optim_out_scurve["channels"] == channel
            )

            dt_optim_out_scurve.loc[mask_levs, "spend"] += carryover_mean
            dt_optim_out_scurve.loc[mask_carryover, "spend"] = carryover_mean

            # Generate simulation points
            max_x = (
                dt_optim_out_scurve[dt_optim_out_scurve["channels"] == channel][
                    "spend"
                ].max()
                * 1.5
            )
            simulate_spend = np.linspace(0, max_x, 100)

            # Calculate responses
            simulate_response = self._fx_objective(
                x=simulate_spend,
                coeff=self.coefs_eval[channel],
                alpha=self.alphas_eval[f"{channel}_alphas"],
                inflexion=self.inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=0,
                get_sum=False,
            )

            simulate_response_carryover = self._fx_objective(
                x=carryover_mean,
                coeff=self.coefs_eval[channel],
                alpha=self.alphas_eval[f"{channel}_alphas"],
                inflexion=self.inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=0,
                get_sum=False,
            )

            # Store simulation data
            plot_dt_scurve.append(
                pd.DataFrame(
                    {
                        "channel": channel,
                        "spend": simulate_spend,
                        "mean_carryover": carryover_mean,
                        "carryover_response": simulate_response_carryover,
                        "total_response": simulate_response,
                    }
                )
            )

            # Update carryover response in main dataframe
            dt_optim_out_scurve.loc[mask_carryover, "response"] = (
                simulate_response_carryover
            )

        # Combine all simulation data
        plot_dt_scurve = pd.concat(plot_dt_scurve, ignore_index=True)

        # Prepare main points data
        main_points = dt_optim_out_scurve.rename(
            columns={
                "response": "response_point",
                "spend": "spend_point",
                "channels": "channel",
            }
        )

        # Log main_points with more information
        self.logger.debug(
            json.dumps(
                {
                    "step": "main_points",
                    "shape": main_points.shape,
                    "columns": list(main_points.columns),
                    "data": main_points.to_dict(
                        orient="records"
                    ),  # 'records' is usually more readable than 'index'
                    "types": {
                        col: str(main_points[col].dtype) for col in main_points.columns
                    },
                },
                indent=2,
                default=str,  # Handles any non-serializable objects like timestamps
            )
        )
        # Calculate mean spend
        temp_caov = main_points[main_points["type"] == "Carryover"]

        # Create a mapping of channel to carryover spend
        caov_spends = dict(zip(temp_caov["channel"], temp_caov["spend_point"]))

        # Initialize mean_spend with spend_point
        main_points["mean_spend"] = main_points["spend_point"]

        # For non-carryover rows, subtract the corresponding channel's carryover spend
        for channel in caov_spends:
            mask = (main_points["channel"] == channel) & (
                main_points["type"] != "Carryover"
            )
            main_points.loc[mask, "mean_spend"] -= caov_spends[channel]

        # Handle duplicate level names
        if levs1[1] == levs1[2]:
            levs1[2] = f"{levs1[2]}."

        # Set factor levels for type
        main_points["type"] = pd.Categorical(
            main_points["type"], categories=["Carryover"] + levs1, ordered=True
        )

        # Calculate ROI and response metrics
        main_points["roi_mean"] = (
            main_points["response_point"] / main_points["mean_spend"]
        )

        # Calculate marginal responses
        mresp_caov = main_points[main_points["type"] == "Carryover"][
            "response_point"
        ].values
        mresp_init = (
            main_points[main_points["type"] == levs1[0]]["response_point"].values
            - mresp_caov
        )
        mresp_b = (
            main_points[main_points["type"] == levs1[1]]["response_point"].values
            - mresp_caov
        )
        mresp_unb = (
            main_points[main_points["type"] == levs1[2]]["response_point"].values
            - mresp_caov
        )

        main_points["marginal_response"] = np.concatenate(
            [
                mresp_init,
                mresp_b,
                mresp_unb,
                np.zeros(len(mresp_init)),  # For Carryover points
            ]
        )

        # Calculate marginal metrics
        main_points["roi_marginal"] = (
            main_points["marginal_response"] / main_points["mean_spend"]
        )
        main_points["cpa_marginal"] = (
            main_points["mean_spend"] / main_points["marginal_response"]
        )
        self.eval_dict["mainPoints"] = main_points
        self.eval_dict["plotDT_scurve"] = plot_dt_scurve
        return {"plot_dt_scurve": plot_dt_scurve, "main_points": main_points}
