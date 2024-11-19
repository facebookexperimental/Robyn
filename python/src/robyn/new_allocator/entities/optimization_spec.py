# pyre-strict

from dataclasses import dataclass
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSpec:
    """Specification for budget allocation optimization run.

    Attributes:
        scenario: Optimization scenario - either "max_response" or "target_efficiency"
        total_budget: Total marketing budget for all paid channels
        date_range: Date(s) to apply transformations and pick mean spends per channel
        channel_constraints_low: Lower bounds for each paid media variable
        channel_constraints_up: Upper bounds for each paid media variable
        channel_constraint_multiplier: Multiplier for constraint ranges
        max_eval: Maximum iterations for global optimization
        constr_mode: Constraint mode - either "eq" (equality) or "ineq" (inequality)
        target_value: Target ROAS or CPA value for target_efficiency scenario
    """

    scenario: str
    total_budget: Optional[float] = None
    date_range: str = "all"
    channel_constraints_low: Union[float, List[float]] = 0.7
    channel_constraints_up: Union[float, List[float]] = 1.2
    channel_constraint_multiplier: float = 3.0
    max_eval: int = 100000
    constr_mode: str = "eq"
    target_value: Optional[float] = None

    def __post_init__(self) -> None:
        """Validates optimization specifications after initialization."""
        logger.debug("Validating optimization specifications")
        self._validate_specs()

    def _validate_specs(self) -> None:
        """Performs validation of optimization specifications."""
        # Validate scenario
        if self.scenario not in ["max_response", "target_efficiency"]:
            raise ValueError(f"Invalid scenario '{self.scenario}'. Must be 'max_response' or 'target_efficiency'")

        # Validate constraint mode
        if self.constr_mode not in ["eq", "ineq"]:
            raise ValueError(f"Invalid constraint mode '{self.constr_mode}'. Must be 'eq' or 'ineq'")

        # For max_response scenario, total_budget will be set automatically if not provided
        if self.scenario == "max_response":
            if self.total_budget is not None and self.total_budget <= 0:
                raise ValueError("total_budget must be positive when provided")

        # Validate numeric values
        if self.total_budget is not None and self.total_budget <= 0:
            raise ValueError("total_budget must be positive")

        if self.channel_constraint_multiplier <= 0:
            raise ValueError("channel_constraint_multiplier must be positive")

        if self.max_eval <= 0:
            raise ValueError("max_eval must be positive")

        # Validate constraints
        self._validate_constraints()

    def _validate_constraints(self) -> None:
        """Validates constraint values and formats."""
        # Convert single values to lists for consistent handling
        if isinstance(self.channel_constraints_low, (int, float)):
            # If low is single value but up is list, expand low to match length
            if isinstance(self.channel_constraints_up, list):
                self.channel_constraints_low = [float(self.channel_constraints_low)] * len(self.channel_constraints_up)
            else:
                self.channel_constraints_low = [float(self.channel_constraints_low)]

        if isinstance(self.channel_constraints_up, (int, float)):
            # If up is single value but low is list, expand up to match length
            if isinstance(self.channel_constraints_low, list):
                self.channel_constraints_up = [float(self.channel_constraints_up)] * len(self.channel_constraints_low)
            else:
                self.channel_constraints_up = [float(self.channel_constraints_up)]

        # Validate constraint values
        for low in self.channel_constraints_low:
            if low < 0.01:
                raise ValueError("Lower bound constraints must be >= 0.01")

        for up in self.channel_constraints_up:
            if up >= 5:
                raise ValueError("Upper bound constraints must be < 5")

        # Validate constraint relationships
        if len(self.channel_constraints_low) != len(self.channel_constraints_up):
            raise ValueError("channel_constraints_low and channel_constraints_up " "must have the same length")

        for low, up in zip(self.channel_constraints_low, self.channel_constraints_up):
            if low >= up:
                raise ValueError(f"Lower bound constraint ({low}) must be less than " f"upper bound constraint ({up})")

    def _is_valid_date_range(self, date_range: str) -> bool:
        """Checks if a date range string is valid.

        Args:
            date_range: String to validate as date range

        Returns:
            bool indicating if date range is valid
        """
        from datetime import datetime

        try:
            # Try parsing as single date
            datetime.strptime(date_range, "%Y-%m-%d")
            return True
        except ValueError:
            try:
                # Try parsing as date range
                start, end = date_range.split(":")
                datetime.strptime(start, "%Y-%m-%d")
                datetime.strptime(end, "%Y-%m-%d")
                return True
            except ValueError:
                return False

    def get_constraint_bounds(self, num_channels: int) -> tuple[List[float], List[float]]:
        """Returns constraint bounds expanded to match number of channels.

        Args:
            num_channels: Number of media channels

        Returns:
            Tuple of (lower_bounds, upper_bounds) lists
        """
        logger.debug(f"Getting constraint bounds for {num_channels} channels")

        # If single values provided, expand to list
        lower = (
            self.channel_constraints_low * num_channels
            if len(self.channel_constraints_low) == 1
            else self.channel_constraints_low
        )

        upper = (
            self.channel_constraints_up * num_channels
            if len(self.channel_constraints_up) == 1
            else self.channel_constraints_up
        )

        return lower, upper
