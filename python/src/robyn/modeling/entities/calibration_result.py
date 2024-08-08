# pyre-strict

from dataclasses import dataclass
from typing import List
from datetime import date
import pandas as pd

@dataclass(frozen=True)
class CalibrationResult:
    """Dataclass to represent calibration results"""
    lift_media: List[str]  # Renamed to follow PEP8 naming conventions
    lift_start: List[date]
    lift_end: List[date]
    lift_abs: List[float]
    decomp_start: List[date]
    decomp_end: List[date]
    decomp_abs_scaled: List[float]
    decomp_abs_total_scaled: List[float]
    calibrated_pct: List[float]
    mape_lift: List[float]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'CalibrationResult':
        """Create CalibrationResult from a pandas DataFrame"""
        return cls(
            lift_media=df['liftMedia'].tolist(),
            lift_start=df['liftStart'].tolist(),
            lift_end=df['liftEnd'].tolist(),
            lift_abs=df['liftAbs'].tolist(),
            decomp_start=df['decompStart'].tolist(),
            decomp_end=df['decompEnd'].tolist(),
            decomp_abs_scaled=df['decompAbsScaled'].tolist(),
            decomp_abs_total_scaled=df['decompAbsTotalScaled'].tolist(),
            calibrated_pct=df['calibrated_pct'].tolist(),
            mape_lift=df['mape_lift'].tolist()
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert CalibrationResult to a pandas DataFrame"""
        return pd.DataFrame({
            'liftMedia': self.lift_media,
            'liftStart': self.lift_start,
            'liftEnd': self.lift_end,
            'liftAbs': self.lift_abs,
            'decompStart': self.decomp_start,
            'decompEnd': self.decomp_end,
            'decompAbsScaled': self.decomp_abs_scaled,
            'decompAbsTotalScaled': self.decomp_abs_total_scaled,
            'calibrated_pct': self.calibrated_pct,
            'mape_lift': self.mape_lift
        })

    def __post_init__(self) -> None:
        """Validate that all lists have the same length"""
        list_lengths = [len(getattr(self, attr)) for attr in self.__dataclass_fields__]
        if len(set(list_lengths)) > 1:
            raise ValueError("All lists in CalibrationResult must have the same length")