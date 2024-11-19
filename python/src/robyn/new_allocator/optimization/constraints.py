# pyre-strict

from typing import Dict, List, Optional, Tuple

import numpy as np


class Constraints:
    """Handles optimization constraints for budget allocation."""

    def __init__(
        self,
        total_budget: Optional[float] = None,
        target_response: Optional[float] = None,
        dep_var_type: str = "revenue",
    ) -> None:
        """Initialize constraint handler."""
        self.total_budget = total_budget
        self.target_response = target_response
        self.dep_var_type = dep_var_type

    def evaluate_equality(
        self,
        x: np.ndarray,
        responses: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Evaluates equality constraints."""
        # For max_response scenario, use total spend constraint if provided
        # Otherwise use the sum of current spends as the budget constraint
        if self.total_budget is None:
            self.total_budget = np.sum(x)  # Use current total spend as constraint

        constr = np.sum(x) - self.total_budget
        grad = np.ones_like(x)

        return float(constr), grad

    def evaluate_inequality(
        self,
        x: np.ndarray,
        responses: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Evaluates inequality constraints.

        Args:
            x: Current spend values
            responses: Response values if using target_response constraint

        Returns:
            Tuple of (constraint_value, gradient)
        """
        # Currently same as equality since we use >= constraints
        return self.evaluate_equality(x, responses)

    def check_bounds(
        self,
        x: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> bool:
        """Checks if values are within bounds.

        Args:
            x: Values to check
            lower_bounds: Minimum allowed values
            upper_bounds: Maximum allowed values

        Returns:
            bool indicating if all values are within bounds
        """
        return bool(np.all(x >= lower_bounds) and np.all(x <= upper_bounds))

    def _constraint_wrapper(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Wraps constraints for optimizer.

        Args:
            x: Current spend values

        Returns:
            Dict with constraint value and gradient
        """
        total_spend = np.sum(x)
        print(f"\nEvaluating constraint")
        print(f"Current spend: {x}")
        print(f"Total spend: {total_spend}")
        print(f"Target budget: {self.total_budget}")

        # Scale constraint to improve numerical stability
        scale_factor = self.total_budget if self.total_budget is not None else 1.0

        constr = (total_spend - self.total_budget) / scale_factor
        grad = np.ones_like(x) / scale_factor

        print(f"Constraint value: {constr}")
        print(f"Gradient: {grad}")

        return {"constraints": constr, "jacobian": grad}
