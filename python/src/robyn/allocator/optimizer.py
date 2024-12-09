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

from .data_preparation import AllocatorDataPreparation

from .utils import check_allocator_constraints, check_metric_dates, get_hill_params


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
        self.allocator_data_preparer = AllocatorDataPreparation(
            mmm_data=self.mmm_data,
            pareto_result=self.pareto_result,
            hyperparameters=self.hyperparameters,
            params=self.params,
            select_model=self.select_model,
        )

        self.allocator_data_preparer._validate_inputs()
        self.allocator_data_preparer._initialize_data()

        self.logger = logging.getLogger(__name__)
        for channel in self.allocator_data_preparer.media_spend_sorted:
            coef = self.allocator_data_preparer.dt_best_coef[
                self.allocator_data_preparer.dt_best_coef["rn"] == channel
            ]["coef"].values[0]
            self.logger.debug(f"{channel}: {coef}")

    def optimize(self) -> AllocationResult:
        """Run the budget allocation optimization."""
        self.logger.debug(
            f"\nStarting optimization for scenario: {self.params.scenario}"
        )

        # Initialize constraints based on scenario
        if self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
            self.constraints = (
                self.allocator_data_preparer._setup_target_efficiency_constraints()
            )
        else:
            self.constraints = self.allocator_data_preparer._setup_constraints()

        bounded_result = self._run_optimization(bounded=True)
        unbounded_result = self._run_optimization(bounded=False)

        return self._process_optimization_results(bounded_result, unbounded_result)

    def _run_optimization(self, bounded: bool = True) -> OptimizationResult:
        """Run optimization while respecting excluded channels."""
        self.logger.debug(f"\nOptimization run (Bounded: {bounded})")

        # Calculate bounds
        if bounded:
            lower_bounds = self.constraints.lower_bounds
            upper_bounds = self.constraints.upper_bounds
        else:
            multiplier = self.params.channel_constr_multiplier
            lower_bounds = np.maximum(
                0,
                self.allocator_data_preparer.init_spend_unit
                * (1 - (1 - self.params.channel_constr_low) * multiplier),
            )
            upper_bounds = self.allocator_data_preparer.init_spend_unit * (
                1 + (self.params.channel_constr_up - 1) * multiplier
            )

        # For excluded channels, set bounds to initial spend
        if np.any(self.allocator_data_preparer.exclude):
            lower_bounds[self.allocator_data_preparer.exclude] = (
                self.allocator_data_preparer.init_spend_unit[
                    self.allocator_data_preparer.exclude
                ]
            )
            upper_bounds[self.allocator_data_preparer.exclude] = (
                self.allocator_data_preparer.init_spend_unit[
                    self.allocator_data_preparer.exclude
                ]
            )

        bounds = list(zip(lower_bounds, upper_bounds))

        # Generate starting points
        starting_points = [
            self.allocator_data_preparer.init_spend_unit,
            lower_bounds,
            upper_bounds,
            (lower_bounds + upper_bounds) / 2,
            np.random.uniform(lower_bounds, upper_bounds),
        ]

        # Setup constraints based on scenario
        constraints = []
        if self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
            if self.allocator_data_preparer.dep_var_type == "revenue":
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: (
                            np.sum(
                                [
                                    self.allocator_data_preparer.calculate_response(
                                        spend, i
                                    )
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
                                    self.allocator_data_preparer.calculate_response(
                                        spend, i
                                    )
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
                        "disp": False,
                    },
                )

                if result.success and result.fun < best_objective:
                    # Ensure excluded channels maintain initial spend
                    final_solution = result.x.copy()
                    final_solution[self.allocator_data_preparer.exclude] = (
                        self.allocator_data_preparer.init_spend_unit[
                            self.allocator_data_preparer.exclude
                        ]
                    )

                    total_response = np.sum(
                        [
                            self.allocator_data_preparer.calculate_response(spend, i)
                            for i, spend in enumerate(final_solution)
                        ]
                    )

                    best_objective = result.fun
                    best_result = OptimizationResult(
                        solution=final_solution,
                        objective=result.fun,
                        gradient=result.jac if hasattr(result, "jac") else None,
                        constraints={},
                    )
                    self.logger.debug("\n=== Optimization Run ===")
                    self.logger.debug("Bounds:", bounds)
                    self.logger.debug("Starting points:", starting_points)
                    self.logger.debug("Constraints:", constraints)
            except Exception as e:
                self.logger.error(f"Optimization attempt {i+1} failed: {str(e)}")
                continue

        if best_result is None:
            raise ValueError("All optimization attempts failed")

        return best_result

    def _objective_function(self, x: np.ndarray) -> float:
        """Objective function with target efficiency handling."""
        responses = np.array(
            [
                self.allocator_data_preparer.calculate_response(spend, i)
                for i, spend in enumerate(x)
            ]
        )

        total_response = np.sum(responses)
        total_spend = np.sum(x)
        self.logger.debug("\n=== Objective Function Call ===")
        self.logger.debug("Input x:", x)
        self.logger.debug("Responses:", responses)
        self.logger.debug("Total response:", total_response)
        self.logger.debug("Total spend:", total_spend)
        if self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
            if self.allocator_data_preparer.dep_var_type == "revenue":
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

    def _process_optimization_results(
        self, bounded_result: OptimizationResult, unbounded_result: OptimizationResult
    ) -> AllocationResult:
        """Process optimization results."""
        # Calculate responses
        bounded_response = np.array([])
        bounded_marginal = np.array([])
        bounded_spend = np.array([])

        unbounded_response = np.array([])
        unbounded_marginal = np.array([])
        unbounded_spend = np.array([])

        for i, channel in enumerate(self.allocator_data_preparer.media_spend_sorted):
            # If channel was excluded (has 0 coefficient or 0 constraints), set spend to 0
            if self.allocator_data_preparer.exclude[i]:
                bounded_spend = np.append(bounded_spend, 0.0)
                unbounded_spend = np.append(unbounded_spend, 0.0)
                bounded_response = np.append(bounded_response, 0.0)
                unbounded_response = np.append(unbounded_response, 0.0)
                bounded_marginal = np.append(bounded_marginal, 0.0)
                unbounded_marginal = np.append(unbounded_marginal, 0.0)
            else:
                # Use optimization results for non-excluded channels
                bounded_spend = np.append(bounded_spend, bounded_result.solution[i])
                unbounded_spend = np.append(
                    unbounded_spend, unbounded_result.solution[i]
                )

                response = self.allocator_data_preparer.calculate_response(
                    bounded_result.solution[i], i
                )
                marginal = self.allocator_data_preparer.calculate_marginal_response(
                    bounded_result.solution[i], i
                )
                bounded_response = np.append(bounded_response, response)
                bounded_marginal = np.append(bounded_marginal, marginal)

                response = self.allocator_data_preparer.calculate_response(
                    unbounded_result.solution[i], i
                )
                marginal = self.allocator_data_preparer.calculate_marginal_response(
                    unbounded_result.solution[i], i
                )
                unbounded_response = np.append(unbounded_response, response)
                unbounded_marginal = np.append(unbounded_marginal, marginal)

        # Create OptimOutData
        optim_out = OptimOutData(
            channels=self.allocator_data_preparer.media_spend_sorted,
            init_spend_unit=self.allocator_data_preparer.init_spend_unit,
            init_response_unit=self.allocator_data_preparer.init_response,
            init_response_marg_unit=self.allocator_data_preparer.init_response_marg,
            optm_spend_unit=bounded_spend,  # Use modified spend array
            optm_response_unit=bounded_response,
            optm_response_marg_unit=bounded_marginal,
            optm_spend_unit_unbound=unbounded_spend,  # Use modified spend array
            optm_response_unit_unbound=unbounded_response,
            optm_response_marg_unit_unbound=unbounded_marginal,
            date_min=str(self.allocator_data_preparer.date_min),
            date_max=str(self.allocator_data_preparer.date_max),
            metric=(
                "ROAS"
                if self.allocator_data_preparer.dep_var_type == "revenue"
                else "CPA"
            ),
            periods=f"{len(self.allocator_data_preparer.hist_filtered)} {self.mmm_data.mmmdata_spec.interval_type}s",
        )
        # Create MainPoints
        response_points = np.vstack(
            [
                self.allocator_data_preparer.init_response,
                bounded_response,
                unbounded_response,
            ]
        )

        spend_points = np.vstack(
            [
                self.allocator_data_preparer.init_spend_unit,
                bounded_result.solution,
                unbounded_result.solution,
            ]
        )

        main_points = MainPoints(
            response_points=response_points,
            spend_points=spend_points,
            channels=self.allocator_data_preparer.media_spend_sorted,
        )

        return AllocationResult(
            dt_optimOut=optim_out,
            mainPoints=main_points,
            scenario=self.params.scenario,
            usecase=self.allocator_data_preparer._determine_usecase(),
            total_budget=self.constraints.budget_constraint,
            skipped_coef0=self.allocator_data_preparer._identify_zero_coefficient_channels(),
            skipped_constr=self.allocator_data_preparer._identify_zero_constraint_channels(),
            no_spend=self.allocator_data_preparer._identify_zero_spend_channels(),
        )

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
