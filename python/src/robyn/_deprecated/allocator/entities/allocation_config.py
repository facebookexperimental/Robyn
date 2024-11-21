# pyre-strict
"""Allocation configuration entities."""
from dataclasses import dataclass
from typing import Optional, Union, List
from datetime import datetime

from robyn.allocator.entities.enums import OptimizationScenario, ConstrMode
from robyn.allocator.entities.allocation_constraints import AllocationConstraints


@dataclass
class AllocationConfig:
    """Configuration for the allocation optimization process."""

    scenario: OptimizationScenario
    total_budget: Optional[float] = None
    target_value: Optional[float] = None
    date_range: Union[str, List[str], datetime] = "all"
    constraints: AllocationConstraints = None
    maxeval: int = 100000
    optim_algo: str = "SLSQP_AUGLAG"  # Sequential Least-Squares Quadratic Programming
    constr_mode: ConstrMode = ConstrMode.EQUALITY
    plots: bool = True
    export: bool = True
    quiet: bool = False

    def __str__(self) -> str:
        return (
            f"AllocationConfig(scenario={self.scenario}, "
            f"budget={self.total_budget}, target={self.target_value}, "
            f"date_range={self.date_range}, maxeval={self.maxeval}, "
            f"algo={self.optim_algo}, constr_mode={self.constr_mode})"
        )


@dataclass
class DateRange:
    """Date range information for allocation calculations."""

    start_date: datetime
    end_date: datetime
    start_index: int
    end_index: int
    n_periods: int
    interval_type: str

    def __str__(self) -> str:
        return (
            f"DateRange({self.start_date.strftime('%Y-%m-%d')} to "
            f"{self.end_date.strftime('%Y-%m-%d')}, "
            f"{self.n_periods} {self.interval_type}s)"
        )
