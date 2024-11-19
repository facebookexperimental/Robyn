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

        print("\nOptimizer Initialization:")
        print(f"Channel names: {self.channel_names}")
        print(f"Initial spend: {self.initial_spend}")
        print(f"Lower bounds: {self.lower_bounds}")
        print(f"Upper bounds: {self.upper_bounds}")
        print(f"Total budget: {self.total_budget}")

    def _objective_wrapper(self, x: np.ndarray, scale: float = 1.0) -> Dict[str, np.ndarray]:
        """Wraps objective function for optimizer."""
        # Scale the input values back to original scale
        x_scaled = x * scale
        print(f"\nObjective evaluation:")
        print(f"Input x: {x}")
        print(f"Scaled x: {x_scaled}")

        # Get response and gradients
        total_response, channel_responses, gradients = self.objective_function.evaluate_total_response(
            x_scaled, self.channel_names
        )

        # We minimize negative response (maximizing response)
        return {
            "objective": -total_response,
            "gradient": -gradients * scale,  # Scale gradients
            "objective_channel": channel_responses,
        }

    def _check_constraint_bounds(self, x: np.ndarray) -> bool:
        """Verifies if solution respects all constraints."""
        total_spend = np.sum(x)
        budget_ok = abs(total_spend - self.total_budget) < 1e-6
        bounds_ok = np.all(x >= self.lower_bounds) and np.all(x <= self.upper_bounds)
        return budget_ok and bounds_ok

    def optimize(self) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """Runs optimization."""
        print("\nStarting optimization...")

        # Calculate scale factor for numerical stability
        scale = np.mean(np.abs(self.initial_spend))
        if scale == 0:
            scale = 1.0

        # Scale values
        x0_scaled = self.initial_spend / scale
        lb_scaled = self.lower_bounds / scale
        ub_scaled = self.upper_bounds / scale
        budget_scaled = self.total_budget / scale

        print(f"\nScaling parameters:")
        print(f"Scale factor: {scale}")
        print(f"Initial scaled: {x0_scaled}")
        print(f"Bounds scaled: [{lb_scaled}, {ub_scaled}]")
        print(f"Budget scaled: {budget_scaled}")

        # Store best result
        best_result = None
        best_value = float("inf")

        # Multiple optimization attempts with different starting points
        for i in range(4):
            print(f"\nOptimization attempt {i+1}:")

            # Perturb initial point while respecting constraints
            x0_perturbed = np.clip(
                x0_scaled * (1 + np.random.uniform(-0.2, 0.2, len(x0_scaled))), lb_scaled, ub_scaled
            )
            # Normalize to match budget constraint
            x0_perturbed = x0_perturbed * (budget_scaled / np.sum(x0_perturbed))
            print(f"Starting point: {x0_perturbed}")

            result = optimize.minimize(
                fun=lambda x: self._objective_wrapper(x, scale)["objective"],
                x0=x0_perturbed,
                method="SLSQP",
                jac=lambda x: self._objective_wrapper(x, scale)["gradient"],
                bounds=optimize.Bounds(lb_scaled, ub_scaled),
                constraints=[
                    {
                        "type": "eq" if self.optimization_spec.constr_mode == "eq" else "ineq",
                        "fun": lambda x: (np.sum(x * scale) - self.total_budget) / self.total_budget,
                        "jac": lambda x: np.ones_like(x) * scale / self.total_budget,
                    }
                ],
                options={
                    "maxiter": 1000,
                    "ftol": 1e-8,
                    "disp": True,
                    "iprint": 2,
                },
            )

            print(f"Attempt {i+1} results:")
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Function value: {result.fun}")
            print(f"Solution: {result.x * scale}")

            if result.success and result.fun < best_value:
                if self._check_constraint_bounds(result.x * scale):
                    best_value = result.fun
                    best_result = result

        if best_result is None:
            raise RuntimeError("Optimization failed to produce valid results")

        # Unscale the solution
        optimal_spend = best_result.x * scale

        # Calculate final responses
        final_responses = self._calculate_responses(optimal_spend)

        print("\nOptimization complete:")
        print(f"Initial spend: {self.initial_spend}")
        print(f"Optimal spend: {optimal_spend}")
        print(f"Initial objective: {self._objective_wrapper(x0_scaled, scale)['objective']}")
        print(f"Final objective: {best_result.fun}")
        print(
            f"Improvement: {(self._objective_wrapper(x0_scaled, scale)['objective'] - best_result.fun)/abs(self._objective_wrapper(x0_scaled, scale)['objective']):.2%}"
        )

        return optimal_spend, vars(best_result), final_responses

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
        """Calculates responses for given spend values."""
        print("\nCalculating final responses:")
        print(f"Input spend values: {spend_values}")

        _, channel_responses, _ = self.objective_function.evaluate_total_response(spend_values, self.channel_names)

        print(f"Channel responses: {channel_responses}")
        return channel_responses
