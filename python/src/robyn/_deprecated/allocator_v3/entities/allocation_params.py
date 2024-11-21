from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class AllocatorParams:
    """
    Parameters for budget allocation optimization.

    Attributes:
        scenario: Type of optimization scenario ('max_response' or 'target_efficiency')
        total_budget: Total budget constraint (optional)
        target_value: Target value for efficiency optimization (optional)
        date_range: Date range for optimization ('all', 'last', or specific range)
        channel_constr_low: Lower bounds for channel constraints
        channel_constr_up: Upper bounds for channel constraints
        channel_constr_multiplier: Multiplier for constraint ranges
        optim_algo: Optimization algorithm to use
        maxeval: Maximum number of evaluations
        constr_mode: Constraint mode ('eq' or 'ineq')
        plots: Whether to generate plots
    """

    scenario: str
    total_budget: Optional[float] = None
    target_value: Optional[float] = None
    date_range: str = "all"
    channel_constr_low: List[float] = None
    channel_constr_up: List[float] = None
    channel_constr_multiplier: float = 3.0
    optim_algo: str = "SLSQP_AUGLAG"
    maxeval: int = 100000
    constr_mode: str = "eq"
    plots: bool = True

    def __post_init__(self):
        """Validate and process parameters after initialization."""
        if self.channel_constr_low is None:
            self.channel_constr_low = [0.7]
        if self.channel_constr_up is None:
            self.channel_constr_up = [1.2]

        # Convert to numpy arrays for easier manipulation
        self.channel_constr_low = np.array(self.channel_constr_low)
        self.channel_constr_up = np.array(self.channel_constr_up)

        # Validate constraints
        if np.any(self.channel_constr_low < 0):
            raise ValueError("Lower bounds must be non-negative")
        if np.any(self.channel_constr_up <= 0):
            raise ValueError("Upper bounds must be positive")
        if np.any(self.channel_constr_up < self.channel_constr_low):
            raise ValueError("Upper bounds must be greater than lower bounds")
