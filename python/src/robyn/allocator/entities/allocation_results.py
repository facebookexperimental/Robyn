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
    summary: str = field(init=False)

    def __post_init__(self):
        """Generate summary after initialization."""
        self.summary = self._generate_summary()

    def _generate_summary(self) -> str:
        """Generate a text summary of allocation results."""
        # Handle both max_response and target_efficiency scenarios
        budget_value = self.metrics.get("total_budget") or self.metrics.get("total_spend", 0)

        return f"""
        Optimization Results Summary:
        Total Spend: ${budget_value:,.2f}
        Expected Response Lift: {self.metrics['response_lift']*100:.1f}%
        Channels Optimized: {len(self.optimal_allocations)}
        {self._get_efficiency_summary()}
        """

    def _get_efficiency_summary(self) -> str:
        """Generate efficiency metrics summary if available."""
        if "achieved_efficiency" in self.metrics and "target_efficiency" in self.metrics:
            return f"""
        Target Efficiency: {self.metrics['target_efficiency']:.2f}
        Achieved Efficiency: {self.metrics['achieved_efficiency']:.2f}
        """
        return ""
