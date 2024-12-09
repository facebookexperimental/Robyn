import numpy as np
from typing import List, Dict
from robyn.allocator.entities.constraints import Constraints
from robyn.allocator.utils import (
    check_metric_dates,
    check_allocator_constraints,
    get_hill_params,
)
import logging

from .entities.allocation_params import AllocatorParams
from .entities.constraints import Constraints
from .constants import (
    SCENARIO_MAX_RESPONSE,
    SCENARIO_TARGET_EFFICIENCY,
)
from typing import List
import numpy as np
import logging
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.pareto_result import ParetoResult
from .entities.allocation_params import AllocatorParams
from .entities.constraints import Constraints
from .constants import (
    SCENARIO_MAX_RESPONSE,
    SCENARIO_TARGET_EFFICIENCY,
)


class AllocatorDataPreparation:
    """Prepare data for the allocator optimization."""

    def __init__(
        self,
        mmm_data: MMMData,
        pareto_result: ParetoResult,
        hyperparameters: Hyperparameters,
        params: AllocatorParams,
        select_model: str,
    ) -> None:
        self.mmm_data = mmm_data
        self.pareto_result = pareto_result
        self.hyperparameters = hyperparameters
        self.params = params
        self.select_model = select_model
        self.media_spend_sorted = None
        self.init_spend_unit = None
        self.init_response = None
        self.hist_filtered = None
        self.dt_best_coef = None
        self.exclude = None
        self.dep_var_type = None
        self.date_min = None
        self.date_max = None

        self.logger = logging.getLogger(__name__)

    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if len(self.mmm_data.mmmdata_spec.paid_media_spends) <= 1:
            raise ValueError("Must have at least two paid media spends")
        self.logger.debug("self.logger.debuging params scenario", self.params.scenario)
        if self.params.scenario not in [
            SCENARIO_MAX_RESPONSE,
            SCENARIO_TARGET_EFFICIENCY,
        ]:
            raise ValueError(f"Invalid scenario: {self.params.scenario}")

        check_allocator_constraints(
            self.params.channel_constr_low, self.params.channel_constr_up
        )

    def _validate_initialization(self) -> None:
        """Validate that all necessary parameters are properly initialized."""
        required_attrs = [
            "init_spend_unit",
            "init_spend_total",
            "init_response",
            "hill_params",
            "total_budget",
            "initial_metrics",
        ]

        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(f"Missing required attribute: {attr}")

            value = getattr(self, attr)
            if value is None:
                raise ValueError(f"Required attribute is None: {attr}")

            if isinstance(value, (np.ndarray, list)):
                if len(value) != len(self.media_spend_sorted):
                    raise ValueError(
                        f"Length mismatch for {attr}: "
                        f"got {len(value)}, expected {len(self.media_spend_sorted)}"
                    )

        # Validate there are no NaN or inf values in critical arrays
        for attr in ["init_spend_unit", "init_response"]:
            value = getattr(self, attr)
            if np.any(~np.isfinite(value)):
                raise ValueError(f"Found non-finite values in {attr}")

    def _determine_usecase(self) -> str:
        """Determine the use case based on initial spend and date range."""
        if self.params.date_range == "all":
            base_case = "all_historical_vec"
        elif self.params.date_range == "last":
            base_case = "last_historical_vec"
        else:
            base_case = "custom_window_vec"

        return f"{base_case} + {'defined' if self.params.total_budget else 'historical'}_budget"

    def _identify_zero_coefficient_channels(self) -> List[str]:
        """Identify channels with zero coefficients."""
        return [
            channel
            for channel, coef in zip(self.media_spend_sorted, self.hill_params.coefs)
            if coef == 0
        ]

    def _identify_zero_constraint_channels(self) -> List[str]:
        """Identify channels with zero constraints."""
        zero_constraints = (np.array(self.params.channel_constr_low) == 0) & (
            np.array(self.params.channel_constr_up) == 0
        )
        return [
            channel
            for channel, is_zero in zip(self.media_spend_sorted, zero_constraints)
            if is_zero
        ]

    def _identify_zero_spend_channels(self) -> List[str]:
        """Identify channels with zero historical spend."""
        return [
            channel
            for channel, spend in zip(
                self.media_spend_sorted, self.hist_spend["histSpendWindowUnit"]
            )
            if spend == 0
        ]

    def _setup_constraints(self) -> Constraints:
        """Setup optimization constraints matching R implementation"""
        # Calculate bounds exactly as R does
        lower_bounds = self.init_spend_unit * self.params.channel_constr_low
        upper_bounds = self.init_spend_unit * self.params.channel_constr_up
        budget_constraint = self.init_spend_total

        self.logger.debug("\nOptimization constraints:")
        self.logger.debug(f"Total budget: {budget_constraint:,.2f}")
        self.logger.debug(f"Bounds multiplier: {self.params.channel_constr_multiplier}")

        return Constraints(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            budget_constraint=budget_constraint,
        )

    def _setup_target_efficiency_constraints(self) -> Constraints:
        """Setup constraints specifically for target efficiency scenario."""
        lower_bounds = self.init_spend_unit * self.params.channel_constr_low[0]
        upper_bounds = self.init_spend_unit * self.params.channel_constr_up[0]

        # Calculate target value
        if self.params.target_value is None:
            if self.dep_var_type == "revenue":
                initial_roas = np.sum(self.init_response) / np.sum(self.init_spend_unit)
                target_value = initial_roas * 0.8  # Target 80% of initial ROAS
                self.logger.debug(
                    f"Target ROAS: {target_value:.4f} (80% of initial {initial_roas:.4f})"
                )
            else:
                initial_cpa = np.sum(self.init_spend_unit) / np.sum(self.init_response)
                target_value = initial_cpa * 1.2  # Target 120% of initial CPA
                self.logger.debug(
                    f"Target CPA: {target_value:.4f} (120% of initial {initial_cpa:.4f})"
                )
        else:
            target_value = self.params.target_value
            self.logger.debug(f"Using provided target value: {target_value:.4f}")

        return Constraints(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            budget_constraint=None,  # No fixed budget for target efficiency
            target_constraint=target_value,
        )

    def _setup_date_ranges(self) -> None:
        """Setup date ranges and windows for optimization."""
        window_loc = slice(
            self.mmm_data.mmmdata_spec.rolling_window_start_which,
            self.mmm_data.mmmdata_spec.rolling_window_end_which,
        )
        self.dt_optim_cost = self.mmm_data.data.iloc[window_loc]

        date_range = check_metric_dates(
            self.params.date_range,
            self.dt_optim_cost[self.mmm_data.mmmdata_spec.date_var],
            self.mmm_data.mmmdata_spec.rolling_window_length,
            is_allocator=True,
        )

        self.date_min = date_range["date_range_updated"][0]
        self.date_max = date_range["date_range_updated"][-1]

        mask = (
            self.dt_optim_cost[self.mmm_data.mmmdata_spec.date_var] >= self.date_min
        ) & (self.dt_optim_cost[self.mmm_data.mmmdata_spec.date_var] <= self.date_max)
        self.hist_filtered = self.dt_optim_cost[mask]

    def _initialize_data(self) -> None:
        """Initialize and prepare data for optimization."""
        # Extract paid media data
        self.paid_media_spends = np.array(self.mmm_data.mmmdata_spec.paid_media_spends)
        self.media_spend_sorted = self.paid_media_spends  # Keep original order

        # Get model parameters
        self.dep_var_type = self.mmm_data.mmmdata_spec.dep_var_type

        # Handle column renames if needed
        for df_name in ["result_hyp_param", "x_decomp_agg"]:
            df = getattr(self.pareto_result, df_name)
            if "sol_id" in df.columns:
                setattr(
                    self.pareto_result, df_name, df.rename(columns={"sol_id": "solID"})
                )

        # Filter for selected model
        self.dt_hyppar = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["solID"] == self.select_model
        ]
        self.dt_best_coef = self.pareto_result.x_decomp_agg[
            (self.pareto_result.x_decomp_agg["solID"] == self.select_model)
            & (self.pareto_result.x_decomp_agg["rn"].isin(self.paid_media_spends))
        ]
        self.logger.debug("Model Coefficients:")
        self.logger.debug(self.dt_best_coef)

        # Initialize hill parameters
        self.hill_params = get_hill_params(
            self.mmm_data,
            self.hyperparameters,
            self.dt_hyppar,
            self.dt_best_coef,
            self.media_spend_sorted,
            self.select_model,
        )

        # Handle zero coefficients like R
        self.exclude = np.array([coef == 0 for coef in self.hill_params.coefs])

        if np.any(self.exclude):
            excluded_channels = [
                channel
                for channel, is_excluded in zip(self.media_spend_sorted, self.exclude)
                if is_excluded
            ]
            self.logger.warning(
                f"The following media channels have zero coefficients and will be excluded: "
                f"{', '.join(excluded_channels)}"
            )

        # Pre-calculate adstocked data and inflexion points
        self.adstocked_ranges = {}
        self.inflexions = {}
        adstocked_data = self.pareto_result.media_vec_collect[
            self.pareto_result.media_vec_collect["type"] == "adstockedMedia"
        ]

        for i, channel in enumerate(self.media_spend_sorted):
            model_data = adstocked_data[channel].values
            x_range = [min(model_data), max(model_data)]
            gamma = self.hill_params.gammas[i]
            inflexion = x_range[0] * (1 - gamma) + x_range[1] * gamma
            self.adstocked_ranges[channel] = x_range
            self.inflexions[channel] = inflexion
        self.logger.debug("\n=== Initialization ===")
        self.logger.debug("Media spend sorted:", self.media_spend_sorted)
        self.logger.debug("Initial spend unit:", self.init_spend_unit)
        self.logger.debug("Initial response:", self.init_response)
        self.logger.debug("Total budget:", self.params.total_budget)
        self._setup_date_ranges()
        self._initialize_optimization_params()

    def _calculate_historical_spend(self) -> Dict[str, np.ndarray]:
        """Calculate historical spend metrics."""
        media_cols = self.media_spend_sorted
        return {
            "histSpendAll": np.array(
                [self.dt_optim_cost[col].sum() for col in media_cols]
            ),
            "histSpendAllUnit": np.array(
                [self.dt_optim_cost[col].mean() for col in media_cols]
            ),
            "histSpendWindow": np.array(
                [self.hist_filtered[col].sum() for col in media_cols]
            ),
            "histSpendWindowUnit": np.array(
                [self.hist_filtered[col].mean() for col in media_cols]
            ),
        }

    def calculate_marginal_response(self, spend: float, channel_index: int) -> float:
        """Calculate marginal response by evaluating at x and x+1."""
        # Skip calculation for excluded channels
        if self.exclude[channel_index]:
            return 0.0

        response_x = self.calculate_response(spend, channel_index)
        response_x_plus_1 = self.calculate_response(spend + 1, channel_index)
        return response_x_plus_1 - response_x

    def _initialize_optimization_params(self) -> None:
        """Initialize optimization parameters"""
        # Calculate historical spend metrics
        self.hist_spend = self._calculate_historical_spend()
        self.init_spend_unit = self.hist_spend["histSpendWindowUnit"]
        self.init_spend_total = np.sum(self.init_spend_unit)

        # Calculate initial responses and marginal responses
        self.init_response = np.array([])
        self.init_response_marg = np.array([])

        for i, spend in enumerate(self.init_spend_unit):
            response = self.calculate_response(spend, i)
            marginal_response = self.calculate_marginal_response(spend, i)
            self.init_response = np.append(self.init_response, response)
            self.init_response_marg = np.append(
                self.init_response_marg, marginal_response
            )

        # Set total budget
        self.total_budget = self.params.total_budget or self.init_spend_total

        # Store initial metrics
        self.initial_metrics = {
            "total_spend": self.init_spend_total,
            "total_response": np.sum(self.init_response),
            "overall_roi": np.sum(self.init_response) / self.init_spend_total,
            "channel_roi": {
                channel: (resp / spend if spend > 0 else 0)
                for channel, resp, spend in zip(
                    self.media_spend_sorted, self.init_response, self.init_spend_unit
                )
            },
            "marginal_response": self.init_response_marg,
        }

        self._validate_initialization()

    def calculate_response(self, spend: float, channel_index: int) -> float:
        """Calculate response using pre-calculated ranges and inflexions."""
        # Return 0 response for excluded channels
        if self.exclude[channel_index]:
            return 0.0

        channel = self.media_spend_sorted[channel_index]

        # Get parameters
        alpha = self.hill_params.alphas[channel_index]
        coef = self.hill_params.coefs[channel_index]
        carryover = self.hill_params.carryover[channel_index]
        inflexion = self.inflexions[channel]

        # Calculate response
        x_adstocked = spend + carryover
        x_saturated = (x_adstocked**alpha) / (x_adstocked**alpha + inflexion**alpha)
        response = coef * x_saturated
        self.logger.debug("\n=== Response Calculation ===")
        self.logger.debug("Channel:", channel)
        self.logger.debug("Input spend:", spend)
        self.logger.debug("Alpha:", alpha)
        self.logger.debug("Coef:", coef)
        self.logger.debug("Carryover:", carryover)
        self.logger.debug("Inflexion:", inflexion)
        self.logger.debug("Response:", response)
        return response
