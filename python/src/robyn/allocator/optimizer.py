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
    """Budget Allocator for marketing mix modeling optimization."""

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

        # Log initial model parameters
        print("\nInitial model metrics:")
        print(f"Total initial spend: {self.init_spend_total:,.2f}")
        print(f"Total initial response: {np.sum(self.init_response):,.2f}")
        print(f"Overall ROI: {np.sum(self.init_response)/self.init_spend_total:.4f}")

        print("\nPareto to Allocator transfer:")
        print(f"Selected model: {select_model}")
        print("Media coefficients from Pareto:")
        for channel in self.media_spend_sorted:
            coef = self.dt_best_coef[self.dt_best_coef["rn"] == channel]["coef"].values[
                0
            ]
            print(f"{channel}: {coef}")

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
        print("Model Coefficients:")
        print(self.dt_best_coef)

        # Initialize hill parameters
        self.hill_params = get_hill_params(
            self.mmm_data,
            self.hyperparameters,
            self.dt_hyppar,
            self.dt_best_coef,
            self.media_spend_sorted,
            self.select_model,
        )
        # Add debug prints after getting hill params:
        print("Hill Parameters:")
        print(f"Alphas: {self.hill_params.alphas}")
        print(f"Gammas: {self.hill_params.gammas}")
        print(f"Coefficients: {self.hill_params.coefs}")
        print(f"Carryover: {self.hill_params.carryover}")

        # Handle zero coefficients like R
        self.exclude = np.array([coef == 0 for coef in self.hill_params.coefs])

        if np.any(self.exclude):
            excluded_channels = [
                channel
                for channel, is_excluded in zip(self.media_spend_sorted, self.exclude)
                if is_excluded
            ]
            logger.warning(
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

    def _initialize_optimization_params(self) -> None:
        """Initialize optimization parameters"""
        # Calculate historical spend metrics
        self.hist_spend = self._calculate_historical_spend()
        self.init_spend_unit = self.hist_spend["histSpendWindowUnit"]
        self.init_spend_total = np.sum(self.init_spend_unit)

        # Calculate initial responses
        self.init_response = np.array(
            [
                self.calculate_response(spend, i)
                for i, spend in enumerate(self.init_spend_unit)
            ]
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
        }

        self._validate_initialization()

    def _setup_constraints(self) -> Constraints:
        """Setup optimization constraints matching R implementation"""
        # Calculate bounds exactly as R does
        lower_bounds = self.init_spend_unit * self.params.channel_constr_low
        upper_bounds = self.init_spend_unit * self.params.channel_constr_up
        budget_constraint = self.init_spend_total

        print("\nOptimization constraints:")
        print(f"Total budget: {budget_constraint:,.2f}")
        print(f"Bounds multiplier: {self.params.channel_constr_multiplier}")

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
                print(
                    f"Target ROAS: {target_value:.4f} (80% of initial {initial_roas:.4f})"
                )
            else:
                initial_cpa = np.sum(self.init_spend_unit) / np.sum(self.init_response)
                target_value = initial_cpa * 1.2  # Target 120% of initial CPA
                print(
                    f"Target CPA: {target_value:.4f} (120% of initial {initial_cpa:.4f})"
                )
        else:
            target_value = self.params.target_value
            print(f"Using provided target value: {target_value:.4f}")

        return Constraints(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            budget_constraint=None,  # No fixed budget for target efficiency
            target_constraint=target_value,
        )

    def optimize(self) -> AllocationResult:
        """Run the budget allocation optimization."""
        print(f"\nStarting optimization for scenario: {self.params.scenario}")

        # Initialize constraints based on scenario
        if self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
            self.constraints = self._setup_target_efficiency_constraints()
        else:
            self.constraints = self._setup_constraints()

        bounded_result = self._run_optimization(bounded=True)
        unbounded_result = self._run_optimization(bounded=False)

        return self._process_optimization_results(bounded_result, unbounded_result)

    def _run_optimization(self, bounded: bool = True) -> OptimizationResult:
        """Run optimization while respecting excluded channels."""
        print(f"\nOptimization run (Bounded: {bounded})")

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

        # For excluded channels, set bounds to initial spend
        if np.any(self.exclude):
            lower_bounds[self.exclude] = self.init_spend_unit[self.exclude]
            upper_bounds[self.exclude] = self.init_spend_unit[self.exclude]

        bounds = list(zip(lower_bounds, upper_bounds))

        # Generate starting points
        starting_points = [
            self.init_spend_unit,
            lower_bounds,
            upper_bounds,
            (lower_bounds + upper_bounds) / 2,
            np.random.uniform(lower_bounds, upper_bounds),
        ]

        # Setup constraints based on scenario
        constraints = []
        if self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
            if self.dep_var_type == "revenue":
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: (
                            np.sum(
                                [
                                    self.calculate_response(spend, i)
                                    for i, spend in enumerate(x)
                                ]
                            )
                            / np.sum(x)
                            - self.constraints.target_constraint
                        ),
                    }
                )
            else:  # CPA
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: (
                            self.constraints.target_constraint
                            - np.sum(x)
                            / np.sum(
                                [
                                    self.calculate_response(spend, i)
                                    for i, spend in enumerate(x)
                                ]
                            )
                        ),
                    }
                )
        else:
            constraints.append(
                {
                    "type": "eq" if self.params.constr_mode == "eq" else "ineq",
                    "fun": lambda x: np.sum(x) - self.constraints.budget_constraint,
                    "jac": lambda x: np.ones_like(x),
                }
            )

        best_result = None
        best_objective = float("inf")

        for i, x0 in enumerate(starting_points):
            try:
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

                if result.success and result.fun < best_objective:
                    # Ensure excluded channels maintain initial spend
                    final_solution = result.x.copy()
                    final_solution[self.exclude] = self.init_spend_unit[self.exclude]

                    print(f"\nNew best solution (attempt {i+1}):")
                    print(f"Objective value: {result.fun:,.2f}")
                    total_response = np.sum(
                        [
                            self.calculate_response(spend, i)
                            for i, spend in enumerate(final_solution)
                        ]
                    )
                    print(f"Total spend: {np.sum(final_solution):,.2f}")
                    print(f"Total response: {total_response:,.2f}")

                    best_objective = result.fun
                    best_result = OptimizationResult(
                        solution=final_solution,
                        objective=result.fun,
                        gradient=result.jac if hasattr(result, "jac") else None,
                        constraints={},
                    )

            except Exception as e:
                logger.error(f"Optimization attempt {i+1} failed: {str(e)}")
                continue

        if best_result is None:
            raise ValueError("All optimization attempts failed")

        return best_result

    def _objective_function(self, x: np.ndarray) -> float:
        """Objective function with target efficiency handling."""
        responses = np.array(
            [self.calculate_response(spend, i) for i, spend in enumerate(x)]
        )

        total_response = np.sum(responses)
        total_spend = np.sum(x)

        if self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
            if self.dep_var_type == "revenue":
                actual_roas = total_response / total_spend if total_spend > 0 else 0
                roas_violation = max(
                    0, self.constraints.target_constraint - actual_roas
                )
                return -total_response + 1e6 * roas_violation
            else:
                actual_cpa = (
                    total_spend / total_response if total_response > 0 else float("inf")
                )
                cpa_violation = max(0, actual_cpa - self.constraints.target_constraint)
                return total_spend + 1e6 * cpa_violation
        else:
            budget_violation = abs(total_spend - self.constraints.budget_constraint)
            bounds_violation = np.sum(
                np.maximum(0, self.constraints.lower_bounds - x)
                + np.maximum(0, x - self.constraints.upper_bounds)
            )
            return -total_response + 1e6 * (budget_violation + bounds_violation)

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

        print(f"\n{channel} Response Calculation:")
        print(f"Input spend: {spend:,.2f}")
        print(f"Adstocked value: {x_adstocked:,.2f}")
        print(f"Saturated value: {x_saturated:.4f}")
        print(f"Final response: {response:.4f}")
        # In calculate_response method
        print(f"Raw spend: {spend}")
        print(f"After adstock: {x_adstocked}")
        print(f"After hill transform: {x_saturated}")

        print("\nResponse calculation components:")
        print(f"Alpha: {self.hill_params.alphas[channel_index]}")
        print(f"Gamma: {self.hill_params.gammas[channel_index]}")
        print(f"Coefficient: {self.hill_params.coefs[channel_index]}")
        return response

    def _process_optimization_results(
        self, bounded_result: OptimizationResult, unbounded_result: OptimizationResult
    ) -> AllocationResult:
        """Process optimization results."""
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

        # Create MainPoints
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

        # Log final results summary
        print("\nOptimization Results Summary:")
        print(f"Initial total response: {np.sum(self.init_response):,.2f}")
        print(f"Optimized total response: {np.sum(bounded_response):,.2f}")
        print(
            f"Response lift: {((np.sum(bounded_response)/np.sum(self.init_response))-1)*100:,.2f}%"
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

    def _validate_hill_params(self) -> None:
        """Validate Hill transformation parameters."""
        invalid_params = []
        for i, channel in enumerate(self.media_spend_sorted):
            params = {
                "alpha": self.hill_params.alphas[i],
                "gamma": self.hill_params.gammas[i],
                "coef": self.hill_params.coefs[i],
                "carryover": self.hill_params.carryover[i],
            }
            for param, value in params.items():
                if not np.isfinite(value):
                    invalid_params.append(f"{channel} {param}: {value}")

        if invalid_params:
            raise ValueError("Invalid Hill parameters:\n" + "\n".join(invalid_params))

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

        optimized_efficiency = -self._optimization_result.objective / np.sum(
            self._optimization_result.solution
        )
        initial_efficiency = np.sum(self.init_response) / self.init_spend_total

        return {
            "initial_efficiency": initial_efficiency,
            "optimized_efficiency": optimized_efficiency,
            "efficiency_improvement": (optimized_efficiency / initial_efficiency) - 1,
        }
