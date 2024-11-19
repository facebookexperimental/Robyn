# pyre-strict
# validation_utils.py

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ValidationUtils:
    """Utilities for input validation in budget allocation."""

    @staticmethod
    def validate_spend_data(
        spends: pd.DataFrame,
        paid_media_vars: List[str],
    ) -> None:
        """Validates media spend data.

        Args:
            spends: DataFrame containing spend data
            paid_media_vars: List of media variable names

        Raises:
            ValueError: If validation fails
        """
        if not all(var in spends.columns for var in paid_media_vars):
            missing = set(paid_media_vars) - set(spends.columns)
            raise ValueError(f"Missing spend variables: {missing}")

        if (spends[paid_media_vars] < 0).any().any():
            raise ValueError("Negative spend values found")

    @staticmethod
    def validate_constraints(
        channel_constraints_low: np.ndarray,
        channel_constraints_up: np.ndarray,
        channel_names: List[str],
    ) -> None:
        """Validates channel constraints.

        Args:
            channel_constraints_low: Lower bounds
            channel_constraints_up: Upper bounds
            channel_names: Channel names

        Raises:
            ValueError: If validation fails
        """
        if len(channel_constraints_low) != len(channel_names):
            raise ValueError("Lower constraints must match number of channels")

        if len(channel_constraints_up) != len(channel_names):
            raise ValueError("Upper constraints must match number of channels")

        if any(channel_constraints_low < 0.01):
            raise ValueError("Lower bounds must be >= 0.01")

        if any(channel_constraints_up >= 5):
            raise ValueError("Upper bounds must be < 5")

        if any(channel_constraints_low >= channel_constraints_up):
            raise ValueError("Lower bounds must be less than upper bounds")

    @staticmethod
    def verify_budget(
        total_budget: float,
        historical_spend: float,
        window_size: int,
    ) -> None:
        """Verifies budget feasibility.

        Args:
            total_budget: Proposed total budget
            historical_spend: Historical spend amount
            window_size: Number of periods

        Raises:
            ValueError: If budget is invalid
        """
        if total_budget <= 0:
            raise ValueError("Total budget must be positive")

        budget_per_period = total_budget / window_size
        hist_per_period = historical_spend / window_size

        if budget_per_period < 0.1 * hist_per_period:
            logger.warning(
                "Budget per period (%.2f) is <10%% of historical spend (%.2f)", budget_per_period, hist_per_period
            )
