# pyre-strict

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def check_opts(value: Any, options: List[Any]) -> None:
    """Check if a value is in a list of options"""
    if value not in options:
        raise ValueError(f"Value {value} not in options: {options}")


def removenacols(df: pd.DataFrame, all: bool = False) -> pd.DataFrame:
    """Remove columns with NA values"""
    if all:
        return df.dropna(axis=1, how="all")
    else:
        return df.dropna(axis=1, how="any")


def formatNum(num: float, abbr: bool = False) -> str:
    """Format numbers for display"""
    # Implementation here
    pass


def glued(template: str, **kwargs: Any) -> str:
    """String interpolation function similar to R's glue"""
    return template.format(**kwargs)


# Add any other utility functions here
