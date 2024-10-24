from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class AllocationConstraints:
    """Constraints for channel allocations."""

    channel_constr_low: Dict[str, float]
    channel_constr_up: Dict[str, float]
    channel_constr_multiplier: float = 3.0

    def __post_init__(self):
        """Validate constraints after initialization."""
        self._validate_constraints()

    def _validate_constraints(self):
        """Ensure constraints are valid."""
        if any(v < 0.01 for v in self.channel_constr_low.values()):
            raise ValueError("Lower bounds must be >= 0.01")

        if any(v > 5 for v in self.channel_constr_up.values()):
            raise ValueError("Upper bounds should be < 5")

    def get_bounds(self, initial_spend: Dict[str, float]) -> List[tuple]:
        """Get bounds for optimization."""
        bounds = []
        for channel in initial_spend.keys():
            lower = initial_spend[channel] * self.channel_constr_low[channel]
            upper = initial_spend[channel] * self.channel_constr_up[channel]
            bounds.append((lower, upper))
        return bounds
