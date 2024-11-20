from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class OptimOutData:
    """
    Optimization output data for each channel.

    Attributes:
        channels: List of channel names
        init_spend_unit: Initial spend per unit for each channel
        init_response_unit: Initial response per unit for each channel
        optm_spend_unit: Optimized spend per unit for each channel
        optm_response_unit: Optimized response per unit for each channel
        optm_spend_unit_unbound: Unbounded optimized spend per unit
        optm_response_unit_unbound: Unbounded optimized response per unit
    """

    channels: np.ndarray
    init_spend_unit: np.ndarray
    init_response_unit: np.ndarray
    optm_spend_unit: np.ndarray
    optm_response_unit: np.ndarray
    optm_spend_unit_unbound: np.ndarray
    optm_response_unit_unbound: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert optimization results to a DataFrame."""
        return pd.DataFrame(
            {
                "channel": self.channels,
                "init_spend": self.init_spend_unit,
                "init_response": self.init_response_unit,
                "optm_spend": self.optm_spend_unit,
                "optm_response": self.optm_response_unit,
                "optm_spend_unbound": self.optm_spend_unit_unbound,
                "optm_response_unbound": self.optm_response_unit_unbound,
            }
        )


@dataclass
class MainPoints:
    """
    Main points data for visualization.

    Attributes:
        response_points: Response values for initial, optimized, and unbounded scenarios
        spend_points: Spend values for initial, optimized, and unbounded scenarios
        channels: Channel names
    """

    response_points: np.ndarray
    spend_points: np.ndarray
    channels: np.ndarray

    def get_scenario_data(self, scenario_index: int) -> Dict[str, np.ndarray]:
        """Get data for a specific scenario."""
        return {
            "response": self.response_points[scenario_index],
            "spend": self.spend_points[scenario_index],
            "channels": self.channels,
        }


@dataclass
class AllocationResult:
    """
    Complete allocation optimization results.

    Attributes:
        dt_optimOut: Detailed optimization output data
        mainPoints: Main points data for visualization
        scenario: Optimization scenario used
        usecase: Use case description
        total_budget: Total budget used
        skipped_coef0: Channels skipped due to zero coefficients
        skipped_constr: Channels skipped due to constraints
        no_spend: Channels with no historical spend
    """

    dt_optimOut: OptimOutData
    mainPoints: MainPoints
    scenario: str
    usecase: str
    total_budget: float
    skipped_coef0: List[str]
    skipped_constr: List[str]
    no_spend: List[str]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the allocation results."""
        return {
            "scenario": self.scenario,
            "usecase": self.usecase,
            "total_budget": self.total_budget,
            "total_channels": len(self.dt_optimOut.channels),
            "active_channels": len(self.dt_optimOut.channels)
            - len(self.skipped_coef0)
            - len(self.skipped_constr),
            "total_response_lift": (
                np.sum(self.dt_optimOut.optm_response_unit)
                / np.sum(self.dt_optimOut.init_response_unit)
                - 1
            ),
            "skipped_channels": {
                "zero_coef": self.skipped_coef0,
                "zero_constraint": self.skipped_constr,
                "no_spend": self.no_spend,
            },
        }
