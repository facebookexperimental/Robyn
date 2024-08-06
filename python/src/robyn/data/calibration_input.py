from dataclasses import dataclass, field
from typing import List
import pandas as pd
from enums import CalibrationScope

@dataclass(frozen=True)
class CalibrationInput:
    """
    CalibrationInput is an immutable data class that holds the necessary inputs for a calibration process.

    Attributes:
        channel (List[str]): List of channel names.
        lift_start_date (pd.Series): Series of lift start dates.
        lift_end_date (pd.Series): Series of lift end dates.
        lift_abs (List[int]): List of absolute lift values.
        spend (List[int]): List of spend values.
        confidence (List[float]): List of confidence intervals.
        metric (List[str]): List of metrics.
        calibration_scopes (List[CalibrationScope]): List of calibration scopes.
    """
    channel: List[str] = field(default_factory=list)
    lift_start_date: pd.Series = field(default_factory=pd.Series)
    lift_end_date: pd.Series = field(default_factory=pd.Series)
    lift_abs: List[int] = field(default_factory=list)
    spend: List[int] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    metric: List[str] = field(default_factory=list)
    calibration_scope: List[CalibrationScope] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"CalibrationInput(\n"
            f"  channel={self.channel},\n"
            f"  lift_start_date={self.lift_start_date.tolist()},\n"
            f"  lift_end_date={self.lift_end_date.tolist()},\n"
            f"  lift_abs={self.lift_abs},\n"
            f"  spend={self.spend},\n"
            f"  confidence={self.confidence},\n"
            f"  metric={self.metric},\n"
            f"  calibration_scope={self.calibration_scope}\n"
            f")"
        )
