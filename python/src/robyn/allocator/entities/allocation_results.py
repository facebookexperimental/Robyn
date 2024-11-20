# pyre-strict
"""Allocation results for budget allocation optimization."""

from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd


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

    def __str__(self) -> str:
        return (
            f"ChannelAllocation({self.channel}:\n"
            f"  Current Spend: ${self.current_spend:,.2f}\n"
            f"  Optimal Spend: ${self.optimal_spend:,.2f}\n"
            f"  Change: {self.spend_change_pct:+.1f}%\n"
            f"  ROI: {self.roi:.2f}\n"
            f"  Contribution: {self.contribution_pct:.1f}%)"
        )


@dataclass
class AllocationResult:
    """Complete results from budget allocation optimization."""

    optimal_allocations: pd.DataFrame
    predicted_responses: pd.DataFrame
    response_curves: pd.DataFrame
    metrics: Dict[str, Any]
    summary: str = field(init=False)

    def __post_init__(self):
        """Process and enhance results after initialization."""
        # Calculate additional metrics
        self._calculate_additional_metrics()
        # Generate summary
        self.summary = self._generate_summary()

    def _calculate_additional_metrics(self):
        """Calculate additional metrics for allocation results."""
        # Calculate spend shares
        total_current_spend = self.optimal_allocations["current_spend"].sum()
        total_optimal_spend = self.optimal_allocations["optimal_spend"].sum()

        self.optimal_allocations["current_spend_share"] = (
            self.optimal_allocations["current_spend"] / total_current_spend
        )
        self.optimal_allocations["optimal_spend_share"] = (
            self.optimal_allocations["optimal_spend"] / total_optimal_spend
        )

        # Calculate response shares
        total_current_response = self.optimal_allocations["current_response"].sum()
        total_optimal_response = self.optimal_allocations["optimal_response"].sum()

        self.optimal_allocations["current_response_share"] = (
            self.optimal_allocations["current_response"] / total_current_response
        )
        self.optimal_allocations["optimal_response_share"] = (
            self.optimal_allocations["optimal_response"] / total_optimal_response
        )

        # Add spend and response lift metrics
        self.metrics.update(
            {
                "total_current_spend": total_current_spend,
                "total_optimal_spend": total_optimal_spend,
                "spend_lift_abs": (total_optimal_spend - total_current_spend)
                / 1000,  # In thousands
                "spend_lift_pct": ((total_optimal_spend / total_current_spend) - 1)
                * 100,
                "response_lift": (total_optimal_response / total_current_response) - 1,
            }
        )

    def _generate_summary(self) -> str:
        """Generate text summary of allocation results."""
        return f"""
Model ID: {self.metrics.get('model_id', '')}
Scenario: {self.metrics.get('scenario', '')}
Use case: {self.metrics.get('use_case', '')}
Window: {self.metrics.get('date_range_start')}:{self.metrics.get('date_range_end')} ({self.metrics.get('n_periods', '')} {self.metrics.get('interval_type', '')})

Dep. Variable Type: {self.metrics.get('dep_var_type', '')}
Media Skipped: {self.metrics.get('skipped_channels', 'None')}
Relative Spend Increase: {self.metrics.get('spend_lift_pct', 0):.1f}% ({self.metrics.get('spend_lift_abs', 0):+.0f}K)
Total Response Increase (Optimized): {self.metrics.get('response_lift', 0)*100:.1f}%

Allocation Summary:
{self._get_channel_summaries()}
"""

    def _get_channel_summaries(self) -> str:
        """Generate channel-level summaries."""
        summaries = []
        for _, row in self.optimal_allocations.iterrows():
            channel_summary = f"""
- {row['channel']}:
  Optimizable bound: [{(row.get('constr_low', 1)-1)*100:.0f}%, {(row.get('constr_up', 1)-1)*100:.0f}%],
  Initial spend share: {row['current_spend_share']*100:.2f}% -> Optimized bounded: {row['optimal_spend_share']*100:.2f}%
  Initial response share: {row['current_response_share']*100:.2f}% -> Optimized bounded: {row['optimal_response_share']*100:.2f}%
  Initial abs. mean spend: {row['current_spend']/1000:.3f}K -> Optimized: {row['optimal_spend']/1000:.3f}K [Delta = {(row['optimal_spend']/row['current_spend']-1)*100:.0f}%]"""
            summaries.append(channel_summary)

        return "\n".join(summaries)

    def __str__(self) -> str:
        return (
            f"AllocationResult(\n"
            f"Total Current Spend: ${self.metrics['total_current_spend']:,.2f}\n"
            f"Total Optimal Spend: ${self.metrics['total_optimal_spend']:,.2f}\n"
            f"Spend Lift: {self.metrics['spend_lift_pct']:+.1f}%\n"
            f"Response Lift: {self.metrics['response_lift']*100:+.1f}%)"
        )
