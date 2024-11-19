# pyre-strict

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

import logging

logger = logging.getLogger(__name__)


@dataclass
class AllocationPlots:
    """Container for allocation visualization plots."""

    spend_response_plot: Optional[Dict] = None
    decomposition_plot: Optional[Dict] = None
    response_curves_plot: Optional[Dict] = None
    onepager_plot: Optional[Dict] = None


@dataclass
class OptimizationResult:
    """Contains results from the budget allocation optimization process."""

    dt_optim_out: pd.DataFrame
    main_points: Dict
    nls_mod: Dict
    plots: AllocationPlots
    scenario: str
    usecase: str
    total_budget: float
    skipped_coef0: List[str]
    skipped_constr: List[str]
    no_spend: List[str]

    def __post_init__(self) -> None:
        """Validates the optimization results after initialization."""
        logger.debug("Validating optimization results")
        self._validate_results()

        self.objective_function = None
        # Ensure columns match R implementation
        self.dt_optim_out = self._standardize_column_names(self.dt_optim_out)

    def _validate_results(self) -> None:
        """Performs basic validation of optimization results."""
        if self.dt_optim_out.empty:
            logger.warning("Optimization output DataFrame is empty")

        if not isinstance(self.total_budget, (int, float)) or self.total_budget < 0:
            raise ValueError("Total budget must be a non-negative number")

        if self.scenario not in ["max_response", "target_efficiency"]:
            raise ValueError(f"Invalid scenario '{self.scenario}'. " "Must be 'max_response' or 'target_efficiency'")

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names to match R implementation."""
        # Map Python column names to R column names
        column_mapping = {
            "channel": "channels",
            "optimal_spend": "optmSpendUnit",
            "initial_spend": "initSpendUnit",
            "response": "optmResponseUnit",
            "spend_share": "optmSpendShareUnit",
            "response_share": "optmResponseShareUnit",
            "roi": "optmRoiUnit",
            "total_spend": "optmSpendUnitTotal",
            "total_response": "optmResponseUnitTotal",
            "response_lift": "optmResponseUnitTotalLift",
        }

        # Rename columns if they exist
        df = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Calculate additional metrics if needed
        if "optmSpendUnitTotal" not in df.columns:
            df["optmSpendUnitTotal"] = df["optmSpendUnit"].sum()
        if "optmResponseUnitTotal" not in df.columns:
            df["optmResponseUnitTotal"] = df["optmResponseUnit"].sum()
        if "optmResponseUnitTotalLift" not in df.columns:
            df["optmResponseUnitTotalLift"] = df["optmResponseUnit"].sum() / df["initSpendUnit"].sum() - 1

        return df

    def get_optimization_metrics(self) -> Dict[str, float]:
        """Returns key optimization metrics."""
        logger.debug("Calculating optimization metrics")
        df = self.dt_optim_out
        return {
            "total_spend": df["optmSpendUnit"].sum(),
            "total_response": df["optmResponseUnit"].sum(),
            "avg_roi": df["optmResponseUnit"].sum() / df["optmSpendUnit"].sum(),
            "response_lift": df["optmResponseUnitTotalLift"].iloc[0],  # Same for all rows
        }

    def __str__(self) -> str:
        """Returns a string representation of the optimization results."""
        metrics = self.get_optimization_metrics()
        return (
            f"OptimizationResult(scenario={self.scenario}, "
            f"total_spend={metrics['total_spend']:.2f}, "
            f"total_response={metrics['total_response']:.2f}, "
            f"response_lift={metrics['response_lift']:.2%})"
        )
