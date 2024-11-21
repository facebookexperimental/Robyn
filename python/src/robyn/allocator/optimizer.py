from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from nevergrad.optimization import optimizerlib
from scipy.optimize import minimize
import logging
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.data.entities.enums import DependentVarType, AdstockType
from robyn.modeling.entities.featurized_mmm_data import FeaturizedMMMData

from .entities.allocation_params import AllocatorParams
from .entities.allocation_result import AllocationResult, OptimOutData, MainPoints
from .entities.optimization_result import OptimizationResult
from .entities.constraints import Constraints
from .constants import (
    SCENARIO_MAX_RESPONSE,
    SCENARIO_TARGET_EFFICIENCY,
    ALGO_SLSQP_AUGLAG,
    ALGO_MMA_AUGLAG,
    CONSTRAINT_MODE_EQ,
    CONSTRAINT_MODE_INEQ,
    DEP_VAR_TYPE_REVENUE,
    DEP_VAR_TYPE_CONVERSION,
)

from .utils import check_allocator_constraints, check_metric_dates, get_hill_params

logger = logging.getLogger(__name__)


class BudgetAllocator:
    """
    Budget Allocator for marketing mix modeling optimization.
    """

    def __init__(
        self,
        mmm_data: MMMData,
        featurized_mmm_data: FeaturizedMMMData,
        hyperparameters: Hyperparameters,
        pareto_result: ParetoResult,
        select_model: str,
        params: AllocatorParams,
    ):
        """Initialize the Budget Allocator."""
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.featurized_mmm_data = featurized_mmm_data
        self.pareto_result = pareto_result
        self.select_model = select_model
        self.params = params

        self._validate_inputs()
        self._initialize_data()

        # After loading data
        logger.debug("\nInitial model parameters:")
        for i, channel in enumerate(self.paid_media_spends):
            logger.debug(f"\n{channel}:")
            logger.debug(f"Initial spend: {self.init_spend_unit[i]:,.2f}")
            logger.debug(f"Coefficient: {self.hill_params.coefs[i]:,.2f}")
            logger.debug(f"Alpha: {self.hill_params.alphas[i]:.4f}")
            logger.debug(f"Gamma: {self.hill_params.gammas[i]:.4f}")
            logger.debug(f"Carryover: {self.hill_params.carryover[i]:.4f}")

            # Calculate initial response for validation
            response = self.calculate_response(self.init_spend_unit[i], i)
            logger.debug(f"Initial response: {response:,.2f}")

    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if len(self.mmm_data.mmmdata_spec.paid_media_spends) <= 1:
            raise ValueError("Must have at least two paid media spends")

        if self.params.scenario not in [
            SCENARIO_MAX_RESPONSE,
            SCENARIO_TARGET_EFFICIENCY,
        ]:
            raise ValueError(f"Invalid scenario: {self.params.scenario}")

        check_allocator_constraints(
            self.params.channel_constr_low, self.params.channel_constr_up
        )

    def _initialize_data(self) -> None:
        """Initialize and prepare data for optimization."""
        # Extract paid media data
        self.paid_media_spends = np.array(self.mmm_data.mmmdata_spec.paid_media_spends)

        # Remove the sorting since we want to keep original order
        # self.media_order = np.argsort(self.paid_media_spends)
        # self.media_spend_sorted = self.paid_media_spends[self.media_order]
        self.media_spend_sorted = self.paid_media_spends  # Keep original order

        # Get model parameters
        self.dep_var_type = self.mmm_data.mmmdata_spec.dep_var_type
        # Check and rename 'sol_id' to 'solID' in result_hyp_param
        if "sol_id" in self.pareto_result.result_hyp_param.columns:
            self.pareto_result.result_hyp_param = (
                self.pareto_result.result_hyp_param.rename(columns={"sol_id": "solID"})
            )
        # Check and rename 'sol_id' to 'solID' in x_decomp_agg
        if "sol_id" in self.pareto_result.x_decomp_agg.columns:
            self.pareto_result.x_decomp_agg = self.pareto_result.x_decomp_agg.rename(
                columns={"sol_id": "solID"}
            )
        # Now proceed with your existing code
        self.dt_hyppar = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["solID"] == self.select_model
        ]
        self.dt_best_coef = self.pareto_result.x_decomp_agg[
            (self.pareto_result.x_decomp_agg["solID"] == self.select_model)
            & (self.pareto_result.x_decomp_agg["rn"].isin(self.paid_media_spends))
        ]
        self.dt_hyppar = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["solID"] == self.select_model
        ]
        self.dt_best_coef = self.pareto_result.x_decomp_agg[
            (self.pareto_result.x_decomp_agg["solID"] == self.select_model)
            & (self.pareto_result.x_decomp_agg["rn"].isin(self.paid_media_spends))
        ]

        # Initialize hill parameters before using them
        self.hill_params = get_hill_params(
            self.mmm_data,
            self.hyperparameters,
            self.dt_hyppar,
            self.dt_best_coef,
            self.media_spend_sorted,
            self.select_model,
        )
        # Pre-calculate adstocked data and inflexion points for all channels
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
        self._setup_date_ranges()
        self._initialize_optimization_params()

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

    def _calculate_historical_spend(self) -> Dict[str, np.ndarray]:
        """Calculate historical spend metrics."""
        media_cols = self.media_spend_sorted

        # Ensure we maintain column order
        hist_spend_all = np.array([self.dt_optim_cost[col].sum() for col in media_cols])
        hist_spend_all_unit = np.array(
            [self.dt_optim_cost[col].mean() for col in media_cols]
        )
        hist_spend_window = np.array(
            [self.hist_filtered[col].sum() for col in media_cols]
        )
        hist_spend_window_unit = np.array(
            [self.hist_filtered[col].mean() for col in media_cols]
        )

        return {
            "histSpendAll": hist_spend_all,
            "histSpendAllUnit": hist_spend_all_unit,
            "histSpendWindow": hist_spend_window,
            "histSpendWindowUnit": hist_spend_window_unit,
        }

    def _initialize_optimization_params(self) -> None:
        """Initialize optimization parameters"""
        logger.debug("\nInitializing optimization parameters...")

        # Calculate historical spend metrics
        self.hist_spend = self._calculate_historical_spend()

        # Get mean spends
        self.init_spend_unit = self.hist_spend["histSpendWindowUnit"]
        self.init_spend_total = np.sum(self.init_spend_unit)

        # logger.debug channel order verification
        logger.debug("\nChannel order verification:")
        for i, channel in enumerate(self.media_spend_sorted):
            logger.debug(f"{i}. {channel}")

        logger.debug("\nInitial spend values:")
        for channel, spend in zip(self.media_spend_sorted, self.init_spend_unit):
            logger.debug(f"{channel}: {spend:,.2f}")

        # Calculate initial responses
        self.init_response = np.zeros(len(self.media_spend_sorted))
        logger.debug("\nCalculating initial responses:")
        for i, (channel, spend) in enumerate(
            zip(self.media_spend_sorted, self.init_spend_unit)
        ):
            # Calculate response
            response = self.calculate_response(spend, i)
            self.init_response[i] = response

            # Calculate ROI
            roi = response / spend if spend > 0 else 0

            # logger.debug detailed information
            logger.debug(f"{channel}:")
            logger.debug(f"  Spend: {spend:,.2f}")
            logger.debug(f"  Response: {response:,.2f}")
            logger.debug(f"  ROI: {roi:.4f}")

        # logger.debug model parameters for validation
        logger.debug("\nInitial model parameters:")
        for i, channel in enumerate(self.media_spend_sorted):
            logger.debug(f"\n{channel}:")
            logger.debug(f"Initial spend: {self.init_spend_unit[i]:,.2f}")
            logger.debug(f"Coefficient: {self.hill_params.coefs[i]:,.4f}")
            logger.debug(f"Alpha: {self.hill_params.alphas[i]:.4f}")
            logger.debug(f"Gamma: {self.hill_params.gammas[i]:.4f}")
            logger.debug(f"Carryover: {self.hill_params.carryover[i]:.4f}")
            logger.debug(f"Initial response: {self.init_response[i]:,.2f}")

        # Validate total budget
        logger.debug(f"\nTotal budget validation:")
        logger.debug(f"Total initial spend: {self.init_spend_total:,.2f}")
        logger.debug(f"Total initial response: {np.sum(self.init_response):,.2f}")
        logger.debug(
            f"Overall ROI: {np.sum(self.init_response)/self.init_spend_total:.4f}"
        )

        # Set up total budget constraint based on initial spend
        if self.params.total_budget is None:
            self.total_budget = self.init_spend_total
        else:
            self.total_budget = self.params.total_budget

        logger.debug(f"\nBudget constraint: {self.total_budget:,.2f}")

        # Calculate and store initial metrics for later comparison
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
        }

        # Validate all parameters are properly initialized
        self._validate_initialization()

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

        logger.debug(
            "\nInitialization validation complete - all parameters properly set"
        )

    def _calculate_initial_response(self) -> np.ndarray:
        """Calculate initial response for each channel."""
        responses = np.zeros(len(self.media_spend_sorted))
        for i, spend in enumerate(self.init_spend_unit):
            responses[i] = self.calculate_response(spend, i)
        return responses

    def _setup_constraints(self) -> Constraints:
        """Setup optimization constraints matching R implementation"""
        logger.debug("\nSetting up optimization constraints...")

        # Calculate bounds exactly as R does
        lower_bounds = self.init_spend_unit * self.params.channel_constr_low
        upper_bounds = self.init_spend_unit * self.params.channel_constr_up
        budget_constraint = self.init_spend_total

        logger.debug(f"\nTotal budget constraint: {budget_constraint:,.2f}")
        logger.debug("\nConstraints per channel:")
        for channel, lb, ub in zip(self.paid_media_spends, lower_bounds, upper_bounds):
            logger.debug(f"{channel}:")
            logger.debug(f"  Lower bound: {lb:,.2f}")
            logger.debug(f"  Upper bound: {ub:,.2f}")

        return Constraints(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            budget_constraint=budget_constraint,
        )

    def optimize(self) -> AllocationResult:
        """Run the budget allocation optimization."""
        # Initialize constraints before optimization
        self.constraints = self._setup_constraints()  # Add this line

        bounded_result = self._run_optimization(bounded=True)
        unbounded_result = self._run_optimization(bounded=False)

        allocation_result = self._process_optimization_results(
            bounded_result, unbounded_result
        )

        return allocation_result

    def _run_optimization(
        self, bounded: bool = True, debug: bool = False
    ) -> OptimizationResult:
        """
        Enhanced optimization process with multiple starting points and improved validation.
        """
        logger.debug("\nStarting optimization run")
        logger.debug(f"Bounded: {bounded}")

        """Enhanced optimization process with multiple starting points and improved validation."""
        if debug:
            logger.debug("\nDEBUG: Starting optimization run")
            logger.debug(f"Bounded: {bounded}")
            self._validate_hill_params()

        # Calculate bounds
        if bounded:
            lower_bounds = self.constraints.lower_bounds
            upper_bounds = self.constraints.upper_bounds
        else:
            multiplier = self.params.channel_constr_multiplier
            lower_bounds = np.maximum(
                0,
                self.init_spend_unit
                * (1 - (1 - self.params.channel_constr_low) * multiplier),
            )
            upper_bounds = self.init_spend_unit * (
                1 + (self.params.channel_constr_up - 1) * multiplier
            )

        bounds = list(zip(lower_bounds, upper_bounds))

        logger.debug("\nOptimization bounds:")
        for channel, bound in zip(self.media_spend_sorted, bounds):
            logger.debug(f"{channel}: [{bound[0]:,.2f}, {bound[1]:,.2f}]")

        # Generate multiple starting points
        starting_points = [
            self.init_spend_unit,  # Current allocation
            lower_bounds,  # Lower bounds
            upper_bounds,  # Upper bounds
            (lower_bounds + upper_bounds) / 2,  # Midpoint
            np.random.uniform(lower_bounds, upper_bounds),  # Random point
        ]

        # Budget constraint
        constraints = [
            {
                "type": "eq" if self.params.constr_mode == "eq" else "ineq",
                "fun": lambda x: np.sum(x) - self.constraints.budget_constraint,
                "jac": lambda x: np.ones_like(x),
            }
        ]

        best_result = None
        best_objective = float("inf")
        no_improvement_count = 0
        tolerance = 1e-6

        for i, x0 in enumerate(starting_points):
            logger.debug(f"\nOptimization attempt {i+1}")
            logger.debug("Starting point:")
            for channel, spend in zip(self.media_spend_sorted, x0):
                logger.debug(f"{channel}: {spend:,.2f}")

            try:
                # Run optimization
                result = minimize(
                    fun=self._objective_function,
                    x0=x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "ftol": 1e-10,
                        "maxiter": self.params.maxeval,
                        "disp": True,
                    },
                )

                # Validate result
                if not result.success:
                    logger.debug(f"Optimization attempt {i+1} failed: {result.message}")
                    continue

                # Calculate responses for current solution
                current_responses = np.array(
                    [
                        self.calculate_response(spend, i)
                        for i, spend in enumerate(result.x)
                    ]
                )

                # Check if this is the best solution so far
                if result.success and result.fun < best_objective:
                    logger.debug(
                        f"New best solution found! Objective: {result.fun:,.2f}"
                    )
                    best_objective = result.fun
                    best_result = result
                    best_solution_debug = {
                        "responses": current_responses,
                        "total_spend": np.sum(result.x),
                        "total_response": -result.fun,
                    }

                    logger.debug("\nBest allocation so far:")
                    for channel, init, opt in zip(
                        self.media_spend_sorted, self.init_spend_unit, result.x
                    ):
                        change = (opt / init - 1) * 100
                        logger.debug(f"{channel}:")
                        logger.debug(f"  Initial: {init:,.2f}")
                        logger.debug(f"  Optimized: {opt:,.2f}")
                        logger.debug(f"  Change: {change:+.1f}%")

                    # logger.debug response details
                    logger.debug("\nResponse details:")
                    for channel, init_resp, opt_resp in zip(
                        self.media_spend_sorted, self.init_response, current_responses
                    ):
                        resp_change = (
                            ((opt_resp / init_resp) - 1) * 100
                            if init_resp > 0
                            else float("nan")
                        )
                        logger.debug(f"{channel}:")
                        logger.debug(f"  Initial response: {init_resp:,.2f}")
                        logger.debug(f"  Optimized response: {opt_resp:,.2f}")
                        logger.debug(f"  Change: {resp_change:+.1f}%")

            except Exception as e:
                logger.debug(f"Optimization attempt {i+1} failed with error: {str(e)}")
                continue

        if best_result is None:
            raise ValueError("All optimization attempts failed")

        # Final validation
        final_spend = best_result.x
        budget_violation = abs(np.sum(final_spend) - self.constraints.budget_constraint)
        bounds_violation = np.sum(
            np.maximum(0, lower_bounds - final_spend)
            + np.maximum(0, final_spend - upper_bounds)
        )

        logger.debug("\nFinal solution validation:")
        logger.debug(f"Budget violation: {budget_violation:,.2f}")
        logger.debug(f"Bounds violation: {bounds_violation:,.2f}")

        if best_solution_debug:
            logger.debug("\nOptimization metrics:")
            logger.debug(f"Total initial response: {np.sum(self.init_response):,.2f}")
            logger.debug(
                f"Total new response: {best_solution_debug['total_response']:,.2f}"
            )
            logger.debug(
                f"Response change: {((best_solution_debug['total_response']/np.sum(self.init_response))-1)*100:+.1f}%"
            )
            logger.debug(
                f"Initial ROI: {np.sum(self.init_response)/np.sum(self.init_spend_unit):.4f}"
            )
            logger.debug(
                f"New ROI: {best_solution_debug['total_response']/best_solution_debug['total_spend']:.4f}"
            )

        # Calculate gradient if available
        gradient = best_result.jac if hasattr(best_result, "jac") else None

        return OptimizationResult(
            solution=best_result.x,
            objective=best_result.fun,
            gradient=gradient,
            constraints={
                "budget_violation": budget_violation,
                "bounds_violation": bounds_violation,
            },
        )

    def _objective_function(self, x: np.ndarray) -> float:
        """Modified objective function with proper handling of zero cases."""
        debug_output = True  # Set to True temporarily for debugging

        if debug_output:
            logger.debug("\nDEBUG: Starting objective function calculation")
            logger.debug("Input spend vector:", x)

        responses = np.zeros(len(self.media_spend_sorted))

        if debug_output:
            logger.debug("\nDEBUG: Optimization step")
            logger.debug("Current allocation:")
            for i, (channel, spend) in enumerate(zip(self.media_spend_sorted, x)):
                logger.debug(f"  {channel}: {spend:,.2f}")
        # Calculate responses
        for i, (spend, channel) in enumerate(zip(x, self.media_spend_sorted)):
            response = self.calculate_response(spend, i)
            responses[i] = response

            if debug_output:
                init_spend = self.init_spend_unit[i]
                init_response = self.calculate_response(init_spend, i)

                # Safe division for ROI calculation
                if spend > 0:
                    roi = response / spend
                else:
                    roi = 0

                logger.debug(f"\nDEBUG: Channel {channel} metrics:")
                logger.debug(f"  Initial spend: {init_spend:,.2f}")
                logger.debug(f"  Initial response: {init_response:,.2f}")
                logger.debug(f"  Proposed spend: {spend:,.2f}")
                if init_spend > 0:
                    logger.debug(f"  Spend change: {((spend/init_spend)-1)*100:+.1f}%")
                logger.debug(f"  New response: {response:,.2f}")
                logger.debug(f"  ROI: {roi:,.4f}")

        total_response = np.sum(responses)
        total_spend = np.sum(x)

        # Calculate penalties
        budget_violation = abs(total_spend - self.constraints.budget_constraint)
        bounds_violation = np.sum(
            np.maximum(0, self.constraints.lower_bounds - x)
            + np.maximum(0, x - self.constraints.upper_bounds)
        )

        penalty = 1e6 * (budget_violation + bounds_violation)

        if debug_output:
            logger.debug("\nDEBUG: Optimization metrics:")
            logger.debug(f"  Total response: {total_response:,.2f}")
            logger.debug(f"  Total spend: {total_spend:,.2f}")
            logger.debug(f"  Budget violation: {budget_violation:,.2f}")
            logger.debug(f"  Bounds violation: {bounds_violation:,.2f}")
            logger.debug(f"  Total penalty: {penalty:,.2f}")

        return -total_response + penalty

    def _calculate_total_response(self, spends: np.ndarray) -> float:
        """Calculate total response across all channels."""
        return sum(self.calculate_response(spend, i) for i, spend in enumerate(spends))

    def calculate_response(
        self, spend: float, channel_index: int, debug: bool = False
    ) -> float:
        """Calculate response using pre-calculated ranges and inflexions."""
        channel = self.media_spend_sorted[channel_index]

        if debug:
            logger.debug(f"\nCalculating response for {channel}")

        # Get parameters
        alpha = self.hill_params.alphas[channel_index]
        coef = self.hill_params.coefs[channel_index]
        carryover = self.hill_params.carryover[channel_index]

        # Get pre-calculated values
        x_range = self.adstocked_ranges[channel]
        inflexion = self.inflexions[channel]

        # Step 1: Adstock transformation
        x_adstocked = spend + carryover

        # Step 2: Hill transformation
        x_saturated = (x_adstocked**alpha) / (x_adstocked**alpha + inflexion**alpha)

        # Step 3: Apply coefficient
        response = coef * x_saturated

        if debug:
            logger.debug(f"Parameters:")
            logger.debug(f"  spend: {spend:.4f}")
            logger.debug(f"  alpha: {alpha:.4f}")
            logger.debug(f"  coef: {coef:.4f}")
            logger.debug(f"  carryover: {carryover:.4f}")
            logger.debug(f"Steps:")
            logger.debug(f"  1a. x_range: [{x_range[0]:.4f}, {x_range[1]:.4f}]")
            logger.debug(f"  1b. x_adstocked: {x_adstocked:.4f}")
            logger.debug(f"  2. inflexion: {inflexion:.4f}")
            logger.debug(f"  3. x_saturated: {x_saturated:.4f}")
            logger.debug(f"  4. final_response: {response:.4f}")

        return response

    def _calculate_lower_bounds(self) -> np.ndarray:
        """Calculate lower bounds for optimization."""
        channel_constr_low = np.array(self.params.channel_constr_low)
        if len(channel_constr_low) == 1:
            channel_constr_low = np.repeat(
                channel_constr_low, len(self.media_spend_sorted)
            )
        return self.init_spend_unit * channel_constr_low

    def _calculate_upper_bounds(self) -> np.ndarray:
        """Calculate upper bounds for optimization."""
        channel_constr_up = np.array(self.params.channel_constr_up)
        if len(channel_constr_up) == 1:
            channel_constr_up = np.repeat(
                channel_constr_up, len(self.media_spend_sorted)
            )
        return self.init_spend_unit * channel_constr_up

    def _calculate_budget_constraint(self) -> float:
        """Calculate budget constraint for optimization."""
        return self.params.total_budget or self.init_spend_total

    def _calculate_target_constraint(self) -> Optional[float]:
        """Calculate target constraint for efficiency optimization."""
        if self.params.scenario != SCENARIO_TARGET_EFFICIENCY:
            return None

        if self.params.target_value is not None:
            return self.params.target_value

        if self.dep_var_type == DependentVarType.CONVERSION:
            return self.init_spend_total / np.sum(self.init_response) * 1.2
        return np.sum(self.init_response) / self.init_spend_total * 0.8

    def _process_optimization_results(
        self, bounded_result: OptimizationResult, unbounded_result: OptimizationResult
    ) -> AllocationResult:
        """Process optimization results with proper handling of zero/nan cases."""
        # Calculate responses
        bounded_response = np.array(
            [
                self.calculate_response(spend, i)
                for i, spend in enumerate(bounded_result.solution)
            ]
        )

        unbounded_response = np.array(
            [
                self.calculate_response(spend, i)
                for i, spend in enumerate(unbounded_result.solution)
            ]
        )

        # Create OptimOutData
        optim_out = OptimOutData(
            channels=self.media_spend_sorted,
            init_spend_unit=self.init_spend_unit,
            init_response_unit=self.init_response,
            optm_spend_unit=bounded_result.solution,
            optm_response_unit=bounded_response,
            optm_spend_unit_unbound=unbounded_result.solution,
            optm_response_unit_unbound=unbounded_response,
            date_min=str(self.date_min),
            date_max=str(self.date_max),
            metric="ROAS" if self.dep_var_type == "revenue" else "CPA",
            periods=f"{len(self.hist_filtered)} {self.mmm_data.mmmdata_spec.interval_type}s",
        )

        # Create MainPoints with proper handling of zero responses
        response_points = np.vstack(
            [self.init_response, bounded_response, unbounded_response]
        )

        spend_points = np.vstack(
            [self.init_spend_unit, bounded_result.solution, unbounded_result.solution]
        )

        main_points = MainPoints(
            response_points=response_points,
            spend_points=spend_points,
            channels=self.media_spend_sorted,
        )

        return AllocationResult(
            dt_optimOut=optim_out,
            mainPoints=main_points,
            scenario=self.params.scenario,
            usecase=self._determine_usecase(),
            total_budget=self.constraints.budget_constraint,
            skipped_coef0=self._identify_zero_coefficient_channels(),
            skipped_constr=self._identify_zero_constraint_channels(),
            no_spend=self._identify_zero_spend_channels(),
        )

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

    def _determine_usecase(self) -> str:
        """Determine the use case based on initial spend and date range."""
        if self.params.date_range == "all":
            base_case = "all_historical_vec"
        elif self.params.date_range == "last":
            base_case = "last_historical_vec"
        else:
            base_case = "custom_window_vec"

        return f"{base_case} + {'defined' if self.params.total_budget else 'historical'}_budget"

    @property
    def total_response_lift(self) -> float:
        """Calculate total response lift from optimization."""
        if not hasattr(self, "_optimization_result"):
            raise ValueError("Optimization hasn't been run yet")

        initial_total_response = np.sum(self.init_response)
        optimized_total_response = -self._optimization_result.objective

        return (optimized_total_response / initial_total_response) - 1

    @property
    def spend_efficiency(self) -> Dict[str, float]:
        """Calculate spend efficiency metrics."""
        if not hasattr(self, "_optimization_result"):
            raise ValueError("Optimization hasn't been run yet")

        return {
            "initial_efficiency": np.sum(self.init_response) / self.init_spend_total,
            "optimized_efficiency": -self._optimization_result.objective
            / np.sum(self._optimization_result.solution),
            "efficiency_improvement": (
                (
                    -self._optimization_result.objective
                    / np.sum(self._optimization_result.solution)
                )
                / (np.sum(self.init_response) / self.init_spend_total)
                - 1
            ),
        }

    def _get_efficiency_based_allocation(self) -> np.ndarray:
        """Generate a starting point based on channel efficiencies."""
        initial_responses = np.array(
            [
                self.calculate_response(spend, i)
                for i, spend in enumerate(self.init_spend_unit)
            ]
        )

        efficiencies = np.where(
            self.init_spend_unit > 0, initial_responses / self.init_spend_unit, 0
        )

        # Normalize and scale to total budget
        weights = (
            efficiencies / np.sum(efficiencies)
            if np.sum(efficiencies) > 0
            else np.ones_like(efficiencies) / len(efficiencies)
        )
        allocation = self.init_spend_total * weights

        # Handle any invalid values
        allocation = np.where(
            np.isfinite(allocation),
            allocation,
            self.init_spend_total / len(self.init_spend_unit),
        )

        return allocation

    def print_allocation_summary(
        self, initial: np.ndarray, optimized: np.ndarray
    ) -> None:
        """logger.debug a summary of allocation changes."""
        logger.debug("\nAllocation Summary:")
        for i, (channel, init, opt) in enumerate(
            zip(self.media_spend_sorted, initial, optimized)
        ):
            change = (opt / init - 1) * 100
            logger.debug(f"{channel}:")
            logger.debug(f"  Initial: {init:,.2f}")
            logger.debug(f"  Optimized: {opt:,.2f}")
            logger.debug(f"  Change: {change:+.1f}%")

    def _validate_hill_params(self) -> None:
        """Validate Hill transformation parameters."""
        logger.debug("\nValidating Hill parameters:")
        for i, channel in enumerate(self.media_spend_sorted):
            params = {
                "alpha": self.hill_params.alphas[i],
                "gamma": self.hill_params.gammas[i],
                "coef": self.hill_params.coefs[i],
                "carryover": self.hill_params.carryover[i],
            }
            logger.debug(f"\nChannel: {channel}")
            for param, value in params.items():
                logger.debug(f"  {param}: {value}")
                if not np.isfinite(value):
                    raise ValueError(f"Invalid {param} for {channel}: {value}")

    def _validate_results(self, result: OptimizationResult) -> None:
        """Validate optimization results with detailed output"""
        logger.debug("\nValidating optimization results:")

        # Check budget constraint
        total_spend = np.sum(result.solution)
        budget_violation = abs(total_spend - self.constraints.budget_constraint)
        logger.debug(f"Budget constraint check:")
        logger.debug(f"  Total spend: {total_spend:,.2f}")
        logger.debug(f"  Budget constraint: {self.constraints.budget_constraint:,.2f}")
        logger.debug(f"  Violation: {budget_violation:,.2f}")

        # Check bounds
        for i, channel in enumerate(self.paid_media_spends):
            spend = result.solution[i]
            lb = self.constraints.lower_bounds[i]
            ub = self.constraints.upper_bounds[i]
            logger.debug(f"\n{channel} bounds check:")
            logger.debug(f"  Spend: {spend:,.2f}")
            logger.debug(f"  Lower bound: {lb:,.2f}")
            logger.debug(f"  Upper bound: {ub:,.2f}")
