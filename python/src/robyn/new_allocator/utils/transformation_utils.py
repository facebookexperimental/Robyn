# pyre-strict
# transformation_utils.py

from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd


class TransformationUtils:
    """Utilities for data transformations in budget allocation."""

    @staticmethod
    def calculate_adstock(
        x: np.ndarray,
        theta: float = 0.0,
        adstock_type: str = "geometric",
        shape: Optional[float] = None,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Applies adstock transformation to spend data."""
        adstock_type = str(adstock_type).lower()

        if adstock_type == "geometric":
            if theta is None:
                raise ValueError("theta parameter required for geometric adstock")
            return x * theta

        elif adstock_type in ["weibull_cdf", "weibull_pdf", "weibull"]:
            if shape is None or scale is None:
                raise ValueError("shape and scale required for Weibull adstock")
            if adstock_type == "weibull_cdf":
                return 1 - np.exp(-((x / scale) ** shape))
            else:  # weibull_pdf
                return (shape / scale) * (x / scale) ** (shape - 1) * np.exp(-((x / scale) ** shape))
        else:
            raise ValueError(f"Unsupported adstock type: {adstock_type}")

    @staticmethod
    def apply_saturation(
        x: np.ndarray,
        alpha: float,
        gamma: float,
    ) -> np.ndarray:
        """Applies hill function saturation transformation.

        Args:
            x: Input values
            alpha: Hill function alpha parameter
            gamma: Hill function gamma parameter

        Returns:
            Transformed values
        """
        return (1 + (gamma**alpha) / (x**alpha)) ** -1

    @staticmethod
    def get_date_range_indices(
        dates: pd.Series,
        date_spec: str,
    ) -> tuple[int, int]:
        """Gets index range for date specification.

        Args:
            dates: Series of dates
            date_spec: Date range specification

        Returns:
            Tuple of (start_index, end_index)
        """
        if date_spec == "all":
            return 0, len(dates) - 1
        elif date_spec.startswith("last_"):
            n_periods = int(date_spec.split("_")[1])
            return len(dates) - n_periods, len(dates) - 1
        elif ":" in date_spec:
            start_date, end_date = date_spec.split(":")
            start_idx = dates[dates >= pd.Timestamp(start_date)].index[0]
            end_idx = dates[dates <= pd.Timestamp(end_date)].index[-1]
            return start_idx, end_idx
        else:
            try:
                idx = dates[dates == pd.Timestamp(date_spec)].index[0]
                return idx, idx
            except:
                raise ValueError(f"Invalid date specification: {date_spec}")
