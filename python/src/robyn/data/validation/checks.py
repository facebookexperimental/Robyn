from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np


def check_vector(x: Any, name: str) -> None:
    """Check if input is a valid vector (not a dict/list with named elements).

    Args:
        x: Input to check
        name: Name of the input for error message

    Raises:
        ValueError: If input is not a valid vector
    """
    if isinstance(x, dict) or (isinstance(x, list) and hasattr(x[0], "__dict__")):
        raise ValueError(f"Input '{name}' must be a valid vector")


def check_paidmedia(
    dt_input: pd.DataFrame,
    paid_media_vars: List[str],
    paid_media_signs: List[str],
    paid_media_spends: List[str],
) -> Dict[str, Any]:
    """Check paid media variables and their properties.

    Args:
        dt_input: Input DataFrame
        paid_media_vars: List of paid media variable names
        paid_media_signs: List of signs for paid media variables
        paid_media_spends: List of paid media spend variable names

    Returns:
        Dictionary containing validated and processed media information

    Raises:
        ValueError: If any validation check fails
    """
    OPTS_PDN = ["positive", "negative"]  # Define allowed signs

    if paid_media_spends is None:
        raise ValueError("Must provide 'paid_media_spends'")

    # Check vector types
    check_vector(paid_media_vars, "paid_media_vars")
    check_vector(paid_media_signs, "paid_media_signs")
    check_vector(paid_media_spends, "paid_media_spends")

    exp_var_count = len(paid_media_vars)
    spend_var_count = len(paid_media_spends)

    # Check if variables exist in data
    missing_vars = [var for var in paid_media_vars if var not in dt_input.columns]
    if missing_vars:
        raise ValueError(
            f"Input 'paid_media_vars' not included in data. Check: {', '.join(missing_vars)}"
        )

    missing_spends = [var for var in paid_media_spends if var not in dt_input.columns]
    if missing_spends:
        raise ValueError(
            f"Input 'paid_media_spends' not included in data. Check: {', '.join(missing_spends)}"
        )

    # Handle paid_media_signs
    if paid_media_signs is None:
        paid_media_signs = ["positive"] * exp_var_count

    if not all(sign in OPTS_PDN for sign in paid_media_signs):
        raise ValueError(
            f"Allowed values for 'paid_media_signs' are: {', '.join(OPTS_PDN)}"
        )

    if len(paid_media_signs) == 1:
        paid_media_signs = paid_media_signs * len(paid_media_vars)

    if len(paid_media_signs) != len(paid_media_vars):
        raise ValueError(
            "Input 'paid_media_signs' must have same length as 'paid_media_vars'"
        )

    if spend_var_count != exp_var_count:
        raise ValueError(
            "Input 'paid_media_spends' must have same length as 'paid_media_vars'"
        )

    # Check numeric types
    non_numeric_vars = [
        var
        for var in paid_media_vars
        if not pd.api.types.is_numeric_dtype(dt_input[var])
    ]
    if non_numeric_vars:
        raise ValueError(
            f"All your 'paid_media_vars' must be numeric. Check: {', '.join(non_numeric_vars)}"
        )

    non_numeric_spends = [
        var
        for var in paid_media_spends
        if not pd.api.types.is_numeric_dtype(dt_input[var])
    ]
    if non_numeric_spends:
        raise ValueError(
            f"All your 'paid_media_spends' must be numeric. Check: {', '.join(non_numeric_spends)}"
        )

    # Check for negative values
    all_media_vars = list(set(paid_media_vars + paid_media_spends))
    negative_vars = [var for var in all_media_vars if (dt_input[var] < 0).any()]
    if negative_vars:
        raise ValueError(
            f"{', '.join(negative_vars)} contains negative values. Media must be >=0"
        )

    # Create exposure selector and selected media
    exposure_selector = [s != v for s, v in zip(paid_media_spends, paid_media_vars)]
    paid_media_selected = [
        var if sel else spend
        for var, spend, sel in zip(
            paid_media_vars, paid_media_spends, exposure_selector
        )
    ]

    return {
        "paid_media_signs": paid_media_signs,
        "paid_media_vars": paid_media_vars,
        "exposure_selector": exposure_selector,
        "paid_media_selected": paid_media_selected,
    }
