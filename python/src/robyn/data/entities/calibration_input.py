from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
from robyn.data.entities.enums import CalibrationScope, DependentVarType

# Define a new data class to hold the calibration data for each channel
@dataclass(frozen=True)
class ChannelCalibrationData:
    """
    ChannelCalibrationData is an immutable data class that holds the calibration data for a single channel.

    Attributes:
        lift_start_date (pd.Timestamp): Lift start date.
        lift_end_date (pd.Timestamp): Lift end date.
        lift_abs (float): Absolute lift value.
        spend (float): Spend value.
        confidence (float): Confidence interval.
        metric (str): DependentVarType.
        calibration_scope (CalibrationScope): Calibration scope.
    """

    lift_start_date: pd.Timestamp = field(default_factory=pd.Timestamp)
    lift_end_date: pd.Timestamp = field(default_factory=pd.Timestamp)
    lift_abs: float = 0
    spend: float = 0
    confidence: float = 0.0
    metric: DependentVarType = None
    calibration_scope: CalibrationScope = CalibrationScope.IMMEDIATE

    def __str__(self) -> str:
        return (
            f"ChannelCalibrationData(\n"
            f"  lift_start_date={self.lift_start_date},\n"
            f"  lift_end_date={self.lift_end_date},\n"
            f"  lift_abs={self.lift_abs},\n"
            f"  spend={self.spend},\n"
            f"  confidence={self.confidence},\n"
            f"  metric={self.metric},\n"
            f"  calibration_scope={self.calibration_scope}\n"
            f")"
        )


@dataclass(frozen=True)
class CalibrationInput:
    """
    CalibrationInput is an immutable data class that holds the necessary inputs for a calibration process.

    Attributes:
        channel_data (Dict[str, ChannelCalibrationData]): Dictionary with channel names as keys and ChannelCalibrationData instances as values.
    """

    channel_data: Dict[str, ChannelCalibrationData] = field(default_factory=dict)

    def __str__(self) -> str:
        channel_data_str = "\n".join(
            f"  {channel}: {data}" for channel, data in self.channel_data.items()
        )
        return f"CalibrationInput(\n{channel_data_str}\n)"
