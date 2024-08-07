from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

@dataclass(frozen=True)
class ModelOutputTrialResult:
    resultHypParam: Dict[str, float]
    xDecompAgg: pd.DataFrame
    decompSpendDist: pd.DataFrame
    rsq_train: float
    rsq_val: float
    rsq_test: float
    nrmse_train: float
    nrmse_val: float
    nrmse_test: float
    nrmse: float
    decomp_rssd: float
    mape: float
    lambda_: float
    lambda_hp: float
    lambda_max: float
    lambda_min_ratio: float
    solID: str
    trial: int
    iterNG: int
    iterPar: int
    liftCalibration: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        for field_name, field_value in self.__dict__.items():
            if field_value is None and field_name != 'liftCalibration':
                raise ValueError(f"{field_name} must not be None")

        if not isinstance(self.resultHypParam, dict):
            raise TypeError("resultHypParam must be a dictionary")
        
        if not isinstance(self.xDecompAgg, pd.DataFrame):
            raise TypeError("xDecompAgg must be a pandas DataFrame")
        
        if not isinstance(self.decompSpendDist, pd.DataFrame):
            raise TypeError("decompSpendDist must be a pandas DataFrame")
        
        if self.liftCalibration is not None and not isinstance(self.liftCalibration, pd.DataFrame):
            raise TypeError("liftCalibration must be a pandas DataFrame or None")
