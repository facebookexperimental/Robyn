# pyre-strict

from dataclasses import dataclass, field
from typing import Dict, List, Union, Tuple
import pandas as pd
from robyn.data.entities.enums import CalibrationScope, DependentVarType


@dataclass(frozen=True)
class ChannelCalibrationData:
    """
    ChannelCalibrationData is an immutable data class that holds the calibration data for a single channel
    or combination of channels.
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
        channel_data: Dictionary mapping channel identifiers to their calibration data.
            Keys can be either strings or tuples of strings for combined channels.
    """

    channel_data: Dict[Union[str, Tuple[str, ...]], ChannelCalibrationData] = field(default_factory=dict)

    def __post_init__(self):
        # Convert string keys with '+' to tuples if needed
        new_channel_data = {}
        for key, value in self.channel_data.items():
            if isinstance(key, str) and "+" in key:
                new_key = tuple(key.split("+"))
            elif isinstance(key, str):
                new_key = (key,)
            else:
                new_key = key
            new_channel_data[new_key] = value

        object.__setattr__(self, "channel_data", new_channel_data)

    def __str__(self) -> str:
        channel_data_str = "\n".join(f"  {'+'.join(channels)}: {data}" for channels, data in self.channel_data.items())
        return f"CalibrationInput(\n{channel_data_str}\n)"
