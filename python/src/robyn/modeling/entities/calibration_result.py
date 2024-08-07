#pyre-strict

from dataclasses import dataclass
from typing import List, Optional
from datetime import date
import pandas as pd

@dataclass(frozen=True)
class CalibrationResult:
    liftMedia: List[str]
    liftStart: List[date]
    liftEnd: List[date]
    liftAbs: List[float]
    decompStart: List[date]
    decompEnd: List[date]
    decompAbsScaled: List[float]
    decompAbsTotalScaled: List[float]
    calibrated_pct: List[float]
    mape_lift: List[float]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'CalibrationResult':
        return cls(
            liftMedia=df['liftMedia'].tolist(),
            liftStart=df['liftStart'].tolist(),
            liftEnd=df['liftEnd'].tolist(),
            liftAbs=df['liftAbs'].tolist(),
            decompStart=df['decompStart'].tolist(),
            decompEnd=df['decompEnd'].tolist(),
            decompAbsScaled=df['decompAbsScaled'].tolist(),
            decompAbsTotalScaled=df['decompAbsTotalScaled'].tolist(),
            calibrated_pct=df['calibrated_pct'].tolist(),
            mape_lift=df['mape_lift'].tolist()
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'liftMedia': self.liftMedia,
            'liftStart': self.liftStart,
            'liftEnd': self.liftEnd,
            'liftAbs': self.liftAbs,
            'decompStart': self.decompStart,
            'decompEnd': self.decompEnd,
            'decompAbsScaled': self.decompAbsScaled,
            'decompAbsTotalScaled': self.decompAbsTotalScaled,
            'calibrated_pct': self.calibrated_pct,
            'mape_lift': self.mape_lift
        })

    def __post_init__(self):
        # Validate that all lists have the same length
        list_lengths = [len(getattr(self, attr)) for attr in self.__annotations__]
        if len(set(list_lengths)) > 1:
            raise ValueError("All lists in CalibrationResult must have the same length")
