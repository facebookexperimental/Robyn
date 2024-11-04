from dataclasses import dataclass
from typing import Optional, Union, List
from datetime import datetime

from .enums import OptimizationScenario, ConstrMode
from .allocation_constraints import AllocationConstraints


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


@dataclass
class DateRange:
    """Date range information for allocation calculations."""

    start_date: datetime
    end_date: datetime
    start_index: int
    end_index: int
    n_periods: int
    interval_type: str
