from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from nevergrad.optimization import optimizerlib
from scipy.optimize import minimize

from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.data.entities.enums import DependentVarType, AdstockType

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
from .utils import (
    get_hill_params,
    check_allocator_constraints,
    check_metric_dates,
    calculate_response,
)


class BudgetAllocator:
    """
    Budget Allocator for marketing mix modeling optimization.
    """

    def __init__(
        self,
        mmm_data: MMMData,
        hyperparameters: Hyperparameters,
        pareto_result: ParetoResult,
        select_model: str,
        params: AllocatorParams,
    ):
        """Initialize the Budget Allocator."""
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.pareto_result = pareto_result
        self.select_model = select_model
        self.params = params

        self._validate_inputs()
        self._initialize_data()

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
        self.media_order = np.argsort(self.paid_media_spends)
        self.media_spend_sorted = self.paid_media_spends[self.media_order]

        # Get model parameters
        self.dep_var_type = self.mmm_data.mmmdata_spec.dep_var_type
        self.dt_hyppar = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["solID"] == self.select_model
        ]
        self.dt_best_coef = self.pareto_result.x_decomp_agg[
            (self.pareto_result.x_decomp_agg["solID"] == self.select_model)
            & (self.pareto_result.x_decomp_agg["rn"].isin(self.paid_media_spends))
        ]

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
        hist_spend_all = self.dt_optim_cost[media_cols].sum().values
        hist_spend_all_unit = self.dt_optim_cost[media_cols].mean().values
        hist_spend_window = self.hist_filtered[media_cols].sum().values
        hist_spend_window_unit = self.hist_filtered[media_cols].mean().values

        return {
            "histSpendAll": hist_spend_all,
            "histSpendAllUnit": hist_spend_all_unit,
            "histSpendWindow": hist_spend_window,
            "histSpendWindowUnit": hist_spend_window_unit,
        }

    def _initialize_optimization_params(self) -> None:
        """Initialize parameters for optimization."""
        self.hist_spend = self._calculate_historical_spend()
        self.init_spend_unit = self.hist_spend["histSpendWindowUnit"]
        self.init_spend_total = np.sum(self.init_spend_unit)

        self.hill_params = get_hill_params(
            self.mmm_data,
            self.hyperparameters,
            self.dt_hyppar,
            self.dt_best_coef,
            self.media_spend_sorted,
            self.select_model,
            self.pareto_result.media_vec_collect,
        )

        self.init_response = self._calculate_initial_response()
        self.constraints = self._setup_constraints()

    def _calculate_initial_response(self) -> np.ndarray:
        """Calculate initial response for each channel."""
        responses = np.zeros(len(self.media_spend_sorted))
        for i, spend in enumerate(self.init_spend_unit):
            responses[i] = self._calculate_response(spend, i)
        return responses

    def _setup_constraints(self) -> Constraints:
        """Setup optimization constraints."""
        lower_bounds = self._calculate_lower_bounds()
        upper_bounds = self._calculate_upper_bounds()
        budget_constraint = self._calculate_budget_constraint()
        target_constraint = self._calculate_target_constraint()

        return Constraints(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            budget_constraint=budget_constraint,
            target_constraint=target_constraint,
        )

    def optimize(self) -> AllocationResult:
        """Run the budget allocation optimization."""
        bounded_result = self._run_optimization(bounded=True)
        unbounded_result = self._run_optimization(bounded=False)

        allocation_result = self._process_optimization_results(
            bounded_result, unbounded_result
        )

        return allocation_result

    def _run_optimization(self, bounded: bool = True) -> OptimizationResult:
        """
        Enhanced optimization process with multiple starting points and improved validation.
        """
        print("\nStarting optimization run")
        print(f"Bounded: {bounded}")

        # Validate Hill parameters before optimization
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

        print("\nOptimization bounds:")
        for channel, bound in zip(self.media_spend_sorted, bounds):
            print(f"{channel}: [{bound[0]:,.2f}, {bound[1]:,.2f}]")

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
        best_solution_debug = None

        for i, x0 in enumerate(starting_points):
            print(f"\nOptimization attempt {i+1}")
            print("Starting point:")
            for channel, spend in zip(self.media_spend_sorted, x0):
                print(f"{channel}: {spend:,.2f}")

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
                    print(f"Optimization attempt {i+1} failed: {result.message}")
                    continue

                # Calculate responses for current solution
                current_responses = np.array(
                    [
                        self._calculate_response(spend, i)
                        for i, spend in enumerate(result.x)
                    ]
                )

                # Check if this is the best solution so far
                if result.success and result.fun < best_objective:
                    print(f"New best solution found! Objective: {result.fun:,.2f}")
                    best_objective = result.fun
                    best_result = result
                    best_solution_debug = {
                        "responses": current_responses,
                        "total_spend": np.sum(result.x),
                        "total_response": -result.fun,
                    }

                    print("\nBest allocation so far:")
                    for channel, init, opt in zip(
                        self.media_spend_sorted, self.init_spend_unit, result.x
                    ):
                        change = (opt / init - 1) * 100
                        print(f"{channel}:")
                        print(f"  Initial: {init:,.2f}")
                        print(f"  Optimized: {opt:,.2f}")
                        print(f"  Change: {change:+.1f}%")

                    # Print response details
                    print("\nResponse details:")
                    for channel, init_resp, opt_resp in zip(
                        self.media_spend_sorted, self.init_response, current_responses
                    ):
                        resp_change = (
                            ((opt_resp / init_resp) - 1) * 100
                            if init_resp > 0
                            else float("nan")
                        )
                        print(f"{channel}:")
                        print(f"  Initial response: {init_resp:,.2f}")
                        print(f"  Optimized response: {opt_resp:,.2f}")
                        print(f"  Change: {resp_change:+.1f}%")

            except Exception as e:
                print(f"Optimization attempt {i+1} failed with error: {str(e)}")
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

        print("\nFinal solution validation:")
        print(f"Budget violation: {budget_violation:,.2f}")
        print(f"Bounds violation: {bounds_violation:,.2f}")

        if best_solution_debug:
            print("\nOptimization metrics:")
            print(f"Total initial response: {np.sum(self.init_response):,.2f}")
            print(f"Total new response: {best_solution_debug['total_response']:,.2f}")
            print(
                f"Response change: {((best_solution_debug['total_response']/np.sum(self.init_response))-1)*100:+.1f}%"
            )
            print(
                f"Initial ROI: {np.sum(self.init_response)/np.sum(self.init_spend_unit):.4f}"
            )
            print(
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
        """Modified objective function with stronger response optimization."""
        responses = np.zeros(len(self.media_spend_sorted))
        debug_output = np.random.random() < 0.01

        # Calculate responses
        for i, (spend, channel) in enumerate(zip(x, self.media_spend_sorted)):
            response = self._calculate_response(spend, i)
            responses[i] = response

            if debug_output:
                init_spend = self.init_spend_unit[i]
                init_response = self._calculate_response(init_spend, i)
                print(f"\n{channel}:")
                print(f"  Initial spend: {init_spend:,.2f}")
                print(f"  Initial response: {init_response:,.2f}")
                print(
                    f"  Proposed spend: {spend:,.2f} ({((spend/init_spend)-1)*100:+.1f}%)"
                )
                print(
                    f"  New response: {response:,.2f} ({((response/init_response)-1)*100:+.1f}%)"
                )
                print(f"  ROI: {response/spend if spend > 0 else 0:,.4f}")

        total_response = np.sum(responses)
        total_spend = np.sum(x)

        # Calculate penalties
        budget_violation = abs(total_spend - self.constraints.budget_constraint)
        penalty = 1e6 * budget_violation

        # Add bounds violation penalty
        bounds_violation = np.sum(
            np.maximum(0, self.constraints.lower_bounds - x)
            + np.maximum(0, x - self.constraints.upper_bounds)
        )
        penalty += 1e6 * bounds_violation

        # Add ROI improvement incentive
        init_roi = np.sum(self.init_response) / np.sum(self.init_spend_unit)
        new_roi = total_response / total_spend if total_spend > 0 else 0
        roi_penalty = max(0, init_roi - new_roi) * 1e4
        penalty += roi_penalty

        if debug_output:
            print(f"\nOptimization metrics:")
            print(f"Total initial response: {np.sum(self.init_response):,.2f}")
            print(f"Total new response: {total_response:,.2f}")
            print(
                f"Response change: {((total_response/np.sum(self.init_response))-1)*100:+.1f}%"
            )
            print(f"Initial ROI: {init_roi:.4f}")
            print(f"New ROI: {new_roi:.4f}")
            print(f"Budget violation: {budget_violation:,.2f}")
            print(f"Bounds violation: {bounds_violation:,.2f}")
            print(f"Total penalty: {penalty:,.2f}")

        return -total_response + penalty

    def _calculate_total_response(self, spends: np.ndarray) -> float:
        """Calculate total response across all channels."""
        return sum(self._calculate_response(spend, i) for i, spend in enumerate(spends))

    def _calculate_response(self, spend_value: float, channel_index: int) -> float:
        """Calculate response with improved debugging and validation."""
        if spend_value <= 0:
            return 0.0

        # Get parameters
        alpha = self.hill_params.alphas[channel_index]
        gamma = self.hill_params.gammas[channel_index]
        coef = self.hill_params.coefs[channel_index]
        carryover = self.hill_params.carryover[channel_index]

        # Validate inputs
        if any(not np.isfinite(x) for x in [alpha, gamma, coef, carryover]):
            print(
                f"Warning: Invalid parameters for {self.media_spend_sorted[channel_index]}"
            )
            return 0.0

        # Adstock transformation
        x_adstocked = spend_value + carryover

        # Hill transformation with numerical stability
        try:
            hill_expr = np.power(gamma / x_adstocked, alpha, where=(x_adstocked > 0))
            response = coef / (1 + hill_expr)
        except Exception as e:
            print(f"Error in response calculation: {str(e)}")
            print(f"Parameters: alpha={alpha}, gamma={gamma}, coef={coef}")
            print(f"Spend: {spend_value}, Adstocked: {x_adstocked}")
            return 0.0

        # Debug output for validation
        if np.random.random() < 0.01:  # Reduce frequency of prints
            print(f"\nResponse calculation for spend {spend_value:,.2f}:")
            print(f"Alpha: {alpha}")
            print(f"Gamma: {gamma}")
            print(f"Coefficient: {coef}")
            print(f"Carryover: {carryover}")
            print(f"Adstocked value: {x_adstocked:,.2f}")
            print(f"Hill expression: {hill_expr}")
            print(f"Final response: {response:,.2f}")

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
        """Process optimization results into final allocation results."""
        # Calculate responses for optimized allocations
        bounded_response = np.array(
            [
                self._calculate_response(spend, i)
                for i, spend in enumerate(bounded_result.solution)
            ]
        )

        unbounded_response = np.array(
            [
                self._calculate_response(spend, i)
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
            date_min=str(self.date_min),  # Added
            date_max=str(self.date_max),  # Added
            metric="ROAS" if self.dep_var_type == "revenue" else "CPA",  # Added
            periods=f"{len(self.hist_filtered)} {self.mmm_data.mmmdata_spec.interval_type}s",  # Added
        )

        # Create MainPoints
        main_points = MainPoints(
            response_points=np.vstack(
                [self.init_response, bounded_response, unbounded_response]
            ),
            spend_points=np.vstack(
                [
                    self.init_spend_unit,
                    bounded_result.solution,
                    unbounded_result.solution,
                ]
            ),
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
                self._calculate_response(spend, i)
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

    def _print_allocation_summary(
        self, initial: np.ndarray, optimized: np.ndarray
    ) -> None:
        """Print a summary of allocation changes."""
        print("\nAllocation Summary:")
        for i, (channel, init, opt) in enumerate(
            zip(self.media_spend_sorted, initial, optimized)
        ):
            change = (opt / init - 1) * 100
            print(f"{channel}:")
            print(f"  Initial: {init:,.2f}")
            print(f"  Optimized: {opt:,.2f}")
            print(f"  Change: {change:+.1f}%")

    def _validate_hill_params(self) -> None:
        """Validate Hill transformation parameters."""
        print("\nValidating Hill parameters:")
        for i, channel in enumerate(self.media_spend_sorted):
            params = {
                "alpha": self.hill_params.alphas[i],
                "gamma": self.hill_params.gammas[i],
                "coef": self.hill_params.coefs[i],
                "carryover": self.hill_params.carryover[i],
            }
            print(f"\nChannel: {channel}")
            for param, value in params.items():
                print(f"  {param}: {value}")
                if not np.isfinite(value):
                    raise ValueError(f"Invalid {param} for {channel}: {value}")

    def _validate_results(self, result: OptimizationResult) -> None:
        """Validate optimization results."""
        print("\nValidating optimization results:")

        # Check budget constraint
        total_spend = np.sum(result.solution)
        budget_violation = abs(total_spend - self.constraints.budget_constraint)
        print(f"Budget violation: {budget_violation:,.2f}")

        # Check bounds
        lb_violation = np.sum(
            np.maximum(0, self.constraints.lower_bounds - result.solution)
        )
        ub_violation = np.sum(
            np.maximum(0, result.solution - self.constraints.upper_bounds)
        )
        print(f"Bounds violation: {lb_violation:,.2f}")
        print(f"Upper bound violation: {ub_violation:,.2f}")

        # Check responses
        final_responses = np.array(
            [
                self._calculate_response(spend, i)
                for i, spend in enumerate(result.solution)
            ]
        )

        print("\nFinal channel responses:")
        for channel, response, init_response in zip(
            self.media_spend_sorted, final_responses, self.init_response
        ):
            change = (
                ((response / init_response) - 1) * 100
                if init_response > 0
                else float("nan")
            )
            print(f"{channel}: {response:,.2f} ({change:+.1f}% change)")
