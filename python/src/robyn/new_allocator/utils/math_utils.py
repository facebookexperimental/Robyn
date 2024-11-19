# pyre-strict
# math_utils.py

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class MathUtils:
    """Mathematical utilities for budget allocation calculations."""

    @staticmethod
    def calculate_roi(
        response: Union[float, np.ndarray],
        spend: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Calculates ROI (Return on Investment).

        Args:
            response: Response values
            spend: Spend values

        Returns:
            ROI values
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            roi = np.where(spend > 0, response / spend, 0)
        return roi

    @staticmethod
    def calculate_cpa(
        spend: Union[float, np.ndarray],
        conversions: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Calculates CPA (Cost per Acquisition).

        Args:
            spend: Spend values
            conversions: Conversion values

        Returns:
            CPA values
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            cpa = np.where(conversions > 0, spend / conversions, np.inf)
        return cpa

    @staticmethod
    def normalize_values(
        values: np.ndarray,
        method: str = "sum",
    ) -> np.ndarray:
        """Normalizes values using specified method.

        Args:
            values: Values to normalize
            method: Normalization method ("sum", "max", "minmax")

        Returns:
            Normalized values
        """
        if method == "sum":
            return values / np.sum(values)
        elif method == "max":
            return values / np.max(values)
        elif method == "minmax":
            vmin, vmax = np.min(values), np.max(values)
            return (values - vmin) / (vmax - vmin)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def aggregate_metrics(
        df: pd.DataFrame,
        group_cols: List[str],
        metric_cols: List[str],
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """Aggregates metrics by groups.

        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            metric_cols: Metric columns to aggregate
            agg_func: Aggregation function

        Returns:
            Aggregated DataFrame
        """
        return df.groupby(group_cols)[metric_cols].agg(agg_func).reset_index()

    @staticmethod
    def calculate_response_lift(
        response_new: Union[float, np.ndarray],
        response_base: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Calculates response lift percentage.

        Args:
            response_new: New response values
            response_base: Base response values

        Returns:
            Lift percentage values
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(response_base > 0, (response_new / response_base) - 1, 0)
        return lift
