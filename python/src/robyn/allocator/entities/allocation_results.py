from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd
import numpy as np


@dataclass
class ChannelAllocation:
    """Allocation results for a single channel."""

    channel: str
    current_spend: float
    optimal_spend: float
    spend_change_pct: float
    response: float
    roi: float
    contribution_pct: float


@dataclass
class AllocationResult:
    """Complete results from budget allocation optimization."""

    optimal_allocations: pd.DataFrame
    predicted_responses: pd.DataFrame
    response_curves: pd.DataFrame
    metrics: Dict[str, float]
    plots: Dict[str, Any]
    summary: str = field(init=False)

    def __post_init__(self):
        """Generate summary after initialization."""
        self.summary = self._generate_summary()

    def _generate_summary(self) -> str:
        """Generate a text summary of allocation results."""
        return f"""
        Optimization Results Summary:
        Total Budget: ${self.metrics['total_budget']:,.2f}
        Expected Response Lift: {self.metrics['response_lift']*100:.1f}%
        Channels Optimized: {len(self.optimal_allocations)}
        """
