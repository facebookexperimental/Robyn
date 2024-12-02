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
        init_response_marg_unit: Initial marginal response per unit for each channel
        optm_spend_unit: Optimized spend per unit for each channel
        optm_response_unit: Optimized response per unit for each channel
        optm_response_marg_unit: Optimized marginal response per unit for each channel
        optm_spend_unit_unbound: Unbounded optimized spend per unit
        optm_response_unit_unbound: Unbounded optimized response per unit
        optm_response_marg_unit_unbound: Unbounded optimized marginal response per unit
        date_min: Start date of optimization window
        date_max: End date of optimization window
        metric: Metric type (ROAS or CPA)
        periods: Time period description
    """

    channels: np.ndarray
    init_spend_unit: np.ndarray
    init_response_unit: np.ndarray
    init_response_marg_unit: np.ndarray
    optm_spend_unit: np.ndarray
    optm_response_unit: np.ndarray
    optm_response_marg_unit: np.ndarray
    optm_spend_unit_unbound: np.ndarray
    optm_response_unit_unbound: np.ndarray
    optm_response_marg_unit_unbound: np.ndarray
    date_min: str
    date_max: str
    metric: str
    periods: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert optimization results to a DataFrame."""
        return pd.DataFrame(
            {
                "channel": self.channels,
                "init_spend": self.init_spend_unit,
                "init_response": self.init_response_unit,
                "init_response_marg": self.init_response_marg_unit,
                "optm_spend": self.optm_spend_unit,
                "optm_response": self.optm_response_unit,
                "optm_response_marg": self.optm_response_marg_unit,
                "optm_spend_unbound": self.optm_spend_unit_unbound,
                "optm_response_unbound": self.optm_response_unit_unbound,
                "optm_response_marg_unbound": self.optm_response_marg_unit_unbound,
            }
        )

    # Share calculations
    @property
    def init_spend_share(self) -> np.ndarray:
        """Calculate initial spend shares."""
        return self.init_spend_unit / np.sum(self.init_spend_unit)

    @property
    def init_response_share(self) -> np.ndarray:
        """Calculate initial response shares."""
        return self.init_response_unit / np.sum(self.init_response_unit)

    @property
    def optm_spend_share_unit(self) -> np.ndarray:
        """Calculate optimized spend shares."""
        return self.optm_spend_unit / np.sum(self.optm_spend_unit)

    @property
    def optm_response_share_unit(self) -> np.ndarray:
        """Calculate optimized response shares."""
        return self.optm_response_unit / np.sum(self.optm_response_unit)

    @property
    def optm_spend_share_unit_unbound(self) -> np.ndarray:
        """Calculate unbounded optimized spend shares."""
        return self.optm_spend_unit_unbound / np.sum(self.optm_spend_unit_unbound)

    @property
    def optm_response_share_unit_unbound(self) -> np.ndarray:
        """Calculate unbounded optimized response shares."""
        return self.optm_response_unit_unbound / np.sum(self.optm_response_unit_unbound)

    @property
    def init_roi(self) -> np.ndarray:
        """Calculate initial ROI."""
        return self.init_response_unit / np.where(
            self.init_spend_unit > 0, self.init_spend_unit, np.inf
        )

    @property
    def optm_roi(self) -> np.ndarray:
        """Calculate optimized ROI."""
        return self.optm_response_unit / np.where(
            self.optm_spend_unit > 0, self.optm_spend_unit, np.inf
        )

    @property
    def optm_roi_unbound(self) -> np.ndarray:
        """Calculate unbounded optimized ROI."""
        return self.optm_response_unit_unbound / np.where(
            self.optm_spend_unit_unbound > 0, self.optm_spend_unit_unbound, np.inf
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
