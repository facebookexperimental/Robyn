from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class OptimizationResult:
    """
    Results from the optimization process.

    Attributes:
        solution: Optimal solution vector
        objective: Objective function value at the solution
        gradient: Gradient at the solution (if available)
        constraints: Constraint values at the solution (if available)
    """

    solution: np.ndarray
    objective: float
    gradient: Optional[np.ndarray] = None
    constraints: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate optimization results after initialization."""
        if not isinstance(self.solution, np.ndarray):
            self.solution = np.array(self.solution)
        if self.gradient is not None and not isinstance(self.gradient, np.ndarray):
            self.gradient = np.array(self.gradient)

    def is_feasible(self, tolerance: float = 1e-6) -> bool:
        """Check if the solution is feasible within given tolerance."""
        if self.constraints is None:
            return True
        return all(abs(v) <= tolerance for v in self.constraints.values())

    def get_metrics(self) -> Dict[str, float]:
        """Get optimization metrics."""
        return {
            "objective_value": self.objective,
            "solution_norm": np.linalg.norm(self.solution),
            "gradient_norm": (
                np.linalg.norm(self.gradient) if self.gradient is not None else None
            ),
        }
