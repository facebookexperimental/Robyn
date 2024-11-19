# pyre-strict

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import nevergrad as ng
from scipy import optimize

from .objective_function import ObjectiveFunction
from .constraints import Constraints
from ..entities.optimization_spec import OptimizationSpec

logger = logging.getLogger(__name__)


class Optimizer:
    """Handles optimization for budget allocation."""

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        optimization_spec: OptimizationSpec,
        channel_names: List[str],
        initial_spend: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        """Initialize optimizer."""
        self.objective_function = objective_function
        self.optimization_spec = optimization_spec
        self.channel_names = channel_names
        self.initial_spend = initial_spend
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.total_budget = (
            self.optimization_spec.total_budget
            if self.optimization_spec.total_budget is not None
            else np.sum(self.initial_spend)
        )
        self.constraints = Constraints(
            total_budget=self.total_budget,
            target_response=self.optimization_spec.target_value,
        )

        logger.debug(
            "Initialized Optimizer with %d channels and %s scenario", len(channel_names), optimization_spec.scenario
        )

    def _objective_wrapper(
        self,
        x: np.ndarray,
        sign: float = -1.0,
    ) -> Dict[str, np.ndarray]:
        """Wraps objective function for optimizer.

        Args:
            x: Spend values
            sign: Direction of optimization (-1 for maximize, 1 for minimize)

        Returns:
            Dict with objective and gradient values
        """
        total_response, channel_responses, gradients = self.objective.evaluate_total_response(x, self.channel_names)

        return {
            "objective": sign * total_response,
            "gradient": sign * gradients,
            "objective_channel": channel_responses,
        }

    def optimize(self) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """Runs optimization."""
        logger.info("Starting optimization")

        # Use log scaling for better numerical stability
        scale = np.mean(np.abs(self.initial_spend))
        x0_scaled = self.initial_spend / scale
        lb_scaled = self.lower_bounds / scale
        ub_scaled = self.upper_bounds / scale
        budget_scaled = self.total_budget / scale

        # Add constraint scaling factor
        constraint_scale = self.total_budget

        # Modified constraints with better scaling
        constraints = [
            {
                "type": "eq" if self.optimization_spec.constr_mode == "eq" else "ineq",
                "fun": lambda x: (np.sum(x * scale) - self.total_budget) / constraint_scale,
                "jac": lambda x: np.ones_like(x) * scale / constraint_scale,
            }
        ]

        # Modified optimization parameters
        options = {
            "maxiter": 1000,
            "ftol": 1e-8,
            "disp": True,
            "iprint": 2,
            "eps": 1e-4,  # Larger step size for finite differences
        }

        # Run optimization with better initial point perturbation
        x0_perturbed = x0_scaled * (1 + np.random.uniform(-0.1, 0.1, len(x0_scaled)))

        result = optimize.minimize(
            fun=lambda x: -self.objective_function.evaluate_total_response(x * scale, self.channel_names)[0],
            x0=x0_perturbed,
            method="SLSQP",
            jac=lambda x: -self.objective_function.evaluate_total_response(x * scale, self.channel_names)[2] * scale,
            bounds=optimize.Bounds(lb_scaled, ub_scaled),
            constraints=constraints,
            options=options,
        )

        # If first attempt doesn't improve, try with different initial points
        if result.fun >= -self.objective_function.evaluate_total_response(self.initial_spend, self.channel_names)[0]:
            best_result = result
            for _ in range(3):
                x0_new = x0_scaled * (1 + np.random.uniform(-0.2, 0.2, len(x0_scaled)))
                temp_result = optimize.minimize(
                    fun=lambda x: -self.objective_function.evaluate_total_response(x * scale, self.channel_names)[0],
                    x0=x0_new,
                    method="SLSQP",
                    jac=lambda x: -self.objective_function.evaluate_total_response(x * scale, self.channel_names)[2]
                    * scale,
                    bounds=optimize.Bounds(lb_scaled, ub_scaled),
                    constraints=constraints,
                    options=options,
                )
                if temp_result.fun < best_result.fun:
                    best_result = temp_result
            result = best_result

        # Unscale the solution
        optimal_spend = result.x * scale

        return optimal_spend, vars(result), self._calculate_responses(optimal_spend)

    def _get_constraints(self, budget_scaled: float) -> List[Dict]:
        """Gets optimization constraints."""
        constraints = [
            {
                "type": "eq" if self.optimization_spec.constr_mode == "eq" else "ineq",
                "fun": lambda x: (np.sum(x) - budget_scaled),
                "jac": lambda x: np.ones_like(x),
            }
        ]

        print("\nConstraint validation:")
        print(f"Budget constraint: sum(x) = {budget_scaled}")
        return constraints

    def _calculate_responses(self, spend_values: np.ndarray) -> np.ndarray:
        """Calculates responses for given spend values.

        Args:
            spend_values: Array of spend values

        Returns:
            Array of response values per channel
        """
        _, channel_responses, _ = self.objective_function.evaluate_total_response(spend_values, self.channel_names)
        return channel_responses
