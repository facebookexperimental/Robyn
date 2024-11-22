from dataclasses import dataclass
from typing import Optional
import numpy as np
from typing import List, Tuple


@dataclass
class Constraints:
    """
    Constraints for the optimization problem.

    Attributes:
        lower_bounds: Lower bounds for each channel's spend
        upper_bounds: Upper bounds for each channel's spend
        budget_constraint: Total budget constraint (can be None for target efficiency)
        target_constraint: Target efficiency constraint (optional)
    """

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    budget_constraint: Optional[float]
    target_constraint: Optional[float] = None

    def __post_init__(self):
        """Validate constraints after initialization."""
        if not isinstance(self.lower_bounds, np.ndarray):
            self.lower_bounds = np.array(self.lower_bounds)
        if not isinstance(self.upper_bounds, np.ndarray):
            self.upper_bounds = np.array(self.upper_bounds)

        # Validate bounds
        if np.any(self.lower_bounds < 0):
            raise ValueError("Lower bounds must be non-negative")
        if np.any(self.upper_bounds <= 0):
            raise ValueError("Upper bounds must be positive")
        if np.any(self.upper_bounds < self.lower_bounds):
            raise ValueError("Upper bounds must be greater than lower bounds")

        # Only validate budget_constraint if it's not None
        if self.budget_constraint is not None and self.budget_constraint <= 0:
            raise ValueError("Budget constraint must be positive")

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds as list of tuples for optimization."""
        return list(zip(self.lower_bounds, self.upper_bounds))

    def is_feasible(self, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a solution is feasible within given tolerance."""
        # Check bounds
        if np.any(x < self.lower_bounds - tolerance) or np.any(
            x > self.upper_bounds + tolerance
        ):
            return False

        # Check budget constraint only if it exists
        if self.budget_constraint is not None:
            if abs(np.sum(x) - self.budget_constraint) > tolerance:
                return False

        # Check target constraint if applicable
        if self.target_constraint is not None:
            # Note: Actual target constraint check would depend on the specific form
            # of the constraint (CPA or ROAS)
            pass

        return True

    def scale_constraints(self, factor: float) -> "Constraints":
        """Create new Constraints object with scaled bounds."""
        return Constraints(
            lower_bounds=self.lower_bounds * factor,
            upper_bounds=self.upper_bounds * factor,
            budget_constraint=(
                self.budget_constraint * factor
                if self.budget_constraint is not None
                else None
            ),
            target_constraint=self.target_constraint,
        )
