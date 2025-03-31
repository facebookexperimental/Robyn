from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd


def check_adstock(adstock: Optional[str]) -> str:
    """Check and validate adstock type.

    Args:
        adstock: Type of adstock transformation

    Returns:
        Validated adstock type

    Raises:
        ValueError: If adstock is None or not a valid type
    """
    if adstock is None:
        raise ValueError(
            "Input 'adstock' can't be NULL. Set any of: 'geometric', 'weibull_cdf' or 'weibull_pdf'"
        )

    # Convert 'weibull' to 'weibull_cdf'
    if adstock == "weibull":
        adstock = "weibull_cdf"

    # Check valid types
    if adstock not in ["geometric", "weibull_cdf", "weibull_pdf"]:
        raise ValueError(
            "Input 'adstock' must be 'geometric', 'weibull_cdf' or 'weibull_pdf'"
        )

    return adstock


def check_metric_type(
    metric_name: str,
    paid_media_spends: List[str],
    paid_media_vars: List[str],
    exposure_vars: List[str],
    organic_vars: Optional[List[str]] = None,
) -> str:
    """Check and determine the type of metric.

    Args:
        metric_name: Name of the metric to check
        paid_media_spends: List of paid media spend variables
        paid_media_vars: List of paid media variables
        exposure_vars: List of exposure variables
        organic_vars: List of organic variables (optional)

    Returns:
        str: Type of metric ('spend', 'exposure', or 'organic')

    Raises:
        ValueError: If metric_name is not found in any of the variable lists
    """
    # Handle case where organic_vars is None
    organic_vars = organic_vars or []

    if metric_name in paid_media_spends:
        return "spend"
    elif metric_name in exposure_vars:
        return "exposure"
    elif metric_name in organic_vars:
        return "organic"
    else:
        error_msg = (
            f"Invalid 'metric_name' input: {metric_name}\n"
            f"Input should be any media variable from paid_media_spends (spend), "
            f"paid_media_vars (exposure), or organic_vars (organic):\n"
            f"- paid_media_spends: {', '.join(paid_media_spends)}\n"
            f"- paid_media_vars: {', '.join(paid_media_vars)}\n"
            f"- organic_vars: {', '.join(organic_vars)}"
        )
        raise ValueError(error_msg)


def check_metric_value(
    metric_value: Optional[Union[float, List[float]]],
    metric_name: str,
    all_values: pd.Series,
    metric_loc: List[int],
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Check and validate metric values.

    Args:
        metric_value: Value(s) for the metric
        metric_name: Name of the metric
        all_values: Series containing all values for the metric
        metric_loc: List of indices for the metric locations

    Returns:
        Dictionary containing updated metric values and all values
    """
    # Handle NaN values
    if metric_value is not None and any(pd.isna(metric_value)):
        metric_value = None

    # Get number of locations
    get_n = len(metric_loc)
    metric_value_updated = all_values.iloc[metric_loc].copy()

    if metric_value is not None:
        # Check if numeric
        if not isinstance(metric_value, (int, float, list, np.ndarray, pd.Series)):
            raise ValueError(
                f"Input 'metric_value' for {metric_name} ({metric_value}) "
                f"must be a numerical value"
            )

        # Convert to numpy array for easier handling
        metric_value = (
            np.array(metric_value)
            if isinstance(metric_value, (list, pd.Series))
            else np.array([metric_value])
        )

        # Check for negative values
        if any(metric_value < 0):
            raise ValueError(f"Input 'metric_value' for {metric_name} must be positive")

        # Handle different length cases
        if get_n > 1 and len(metric_value) == 1:
            # Split value proportionally across periods
            metric_value_updated = metric_value[0] * (
                metric_value_updated / metric_value_updated.sum()
            )
        elif get_n == 1 and len(metric_value) == 1:
            metric_value_updated = metric_value[0]
        else:
            raise ValueError(
                "robyn_response metric_value & date_range must have same length"
            )

    # Update all values
    all_values_updated = all_values.copy()
    all_values_updated.iloc[metric_loc] = metric_value_updated

    return {
        "metric_value_updated": metric_value_updated,
        "all_values_updated": all_values_updated,
    }


def check_metric_dates(
    date_range: Optional[Union[str, List[str]]] = None,
    all_dates: pd.Series = None,
    day_interval: Optional[int] = None,
    quiet: bool = False,
    is_allocator: bool = True,
) -> Dict[str, Union[List[pd.Timestamp], List[int]]]:
    """Check and validate metric dates."""
    # Reset index of all_dates to ensure continuous indexing from 0
    all_dates = all_dates.reset_index(drop=True)

    # Debug prints
    print("\nDebug info for check_metric_dates:")
    print(f"date_range: {date_range}")
    print(f"all_dates type: {type(all_dates)}")
    print(
        f"all_dates shape: {all_dates.shape if hasattr(all_dates, 'shape') else 'no shape'}"
    )
    print(
        f"all_dates head: {all_dates.head() if hasattr(all_dates, 'head') else 'no head method'}"
    )
    print(f"day_interval: {day_interval}")

    # Default handling
    if date_range is None:
        if day_interval is None:
            raise ValueError("Input 'date_range' or 'day_interval' must be defined")
        date_range = "all"
        if not quiet:
            print(f"Automatically picked date_range = '{date_range}'")

    # Handle "last_n" or "all" format
    if isinstance(date_range, str) and any(x in date_range for x in ["last", "all"]):
        if date_range == "all":
            date_range = f"last_{len(all_dates)}"

        # Get number of periods
        get_n = int(date_range.replace("last_", "")) if "_" in date_range else 1

        # Get date range and locations
        date_range = all_dates.tail(get_n).tolist()
        date_range_loc = all_dates[all_dates.isin(date_range)].index.tolist()
        date_range_updated = all_dates.iloc[date_range_loc].tolist()

    # Handle specific dates
    else:
        try:
            # Convert to datetime if string
            if isinstance(date_range, str):
                date_range = [pd.to_datetime(date_range)]
            elif isinstance(date_range, list):
                date_range = [pd.to_datetime(d) for d in date_range]

            # Single date handling
            if len(date_range) == 1:
                if date_range[0] in all_dates.values:
                    date_range_updated = date_range
                    date_range_loc = all_dates[
                        all_dates == date_range[0]
                    ].index.tolist()
                    if not quiet:
                        print(
                            f"Using ds '{date_range_updated[0]}' as the response period"
                        )
                else:
                    # Find closest date
                    date_range_loc = [abs(all_dates - date_range[0]).argmin()]
                    date_range_updated = [all_dates.iloc[date_range_loc[0]]]
                    if not quiet:
                        print(
                            f"Input 'date_range' ({date_range[0]}) has no match. "
                            f"Picking closest date: {date_range_updated[0]}"
                        )

            # Date range handling
            elif len(date_range) == 2:
                # Find closest dates for range
                date_range_loc = [abs(all_dates - date).argmin() for date in date_range]
                date_range_loc = list(range(date_range_loc[0], date_range_loc[1] + 1))
                date_range_updated = all_dates.iloc[date_range_loc].tolist()

                if not quiet and not all(d in date_range_updated for d in date_range):
                    print(
                        f"At least one date in 'date_range' input does not match any date. "
                        f"Picking closest dates for range: "
                        f"{date_range_updated[0]}:{date_range_updated[-1]}"
                    )

            # Multiple specific dates
            else:
                if all(d in all_dates.values for d in date_range):
                    date_range_loc = all_dates[
                        all_dates.isin(date_range)
                    ].index.tolist()
                    date_range_updated = date_range
                else:
                    date_range_loc = [abs(all_dates - d).argmin() for d in date_range]

                # Check if dates are sequential
                if all(np.diff(date_range_loc) == 1):
                    date_range_updated = all_dates.iloc[date_range_loc].tolist()
                    if not quiet:
                        print(
                            f"At least one date in 'date_range' does not match ds. "
                            f"Picking closest dates"
                        )
                else:
                    raise ValueError(
                        "Input 'date_range' needs to have sequential dates"
                    )

        except (ValueError, TypeError):
            raise ValueError(
                "Input 'date_range' must have date format 'YYYY-MM-DD' or use 'last_n'"
            )

    return {"date_range_updated": date_range_updated, "metric_loc": date_range_loc}


def check_daterange(
    date_min: Optional[pd.Timestamp],
    date_max: Optional[pd.Timestamp],
    dates: pd.Series,
) -> None:
    """Check if date range is valid.

    Args:
        date_min: Minimum date
        date_max: Maximum date
        dates: Series of all available dates

    Raises:
        ValueError: If date_min or date_max is invalid
    """
    if date_min is not None:
        # Check if date_min is a single date
        if isinstance(date_min, (list, pd.Series)) and len(date_min) > 1:
            raise ValueError("Set a single date for 'date_min' parameter")

        # Check if date_min is in range
        if date_min < dates.min():
            print(
                f"Warning: Parameter 'date_min' not in your data's date range. "
                f"Changed to '{dates.min()}'"
            )

    if date_max is not None:
        # Check if date_max is a single date
        if isinstance(date_max, (list, pd.Series)) and len(date_max) > 1:
            raise ValueError("Set a single date for 'date_max' parameter")

        # Check if date_max is in range
        if date_max > dates.max():
            print(
                f"Warning: Parameter 'date_max' not in your data's date range. "
                f"Changed to '{dates.max()}'"
            )
