from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from robyn.modeling.entities.modeloutput import ModelOutput, ResultHypParam, XDecompAgg

@dataclass(frozen=True)
class ModelOutputCollection:
    # Group 1: Model Results
    resultHypParam: ResultHypParam
    xDecompAgg: XDecompAgg
    mediaVecCollect: pd.DataFrame
    xDecompVecCollect: pd.DataFrame
    resultCalibration: Optional[pd.DataFrame] = None
    model_output: ModelOutput

    # Group 2: Model Solutions
    allSolutions: List[str]
    allPareto: Dict[str, Any]
    calibration_constraint: float
    pareto_fronts: int
    selectID: Optional[str] = None

    # Group 3: Model Configuration
    cores: int
    iterations: int
    trials: List[Any]
    intercept_sign: str
    nevergrad_algo: str
    add_penalty_factor: bool
    seed: int
    hyper_fixed: bool
    hyper_updated: Optional[Dict[str, List[float]]] = None

    # Group 4: Output and Visualization
    UI: Optional[Any] = None
    convergence: Optional[Dict[str, Any]] = None
    ts_validation_plot: Optional[Any] = None
    plot_folder: Optional[str] = None

    # Group 5: Performance Metrics
    runTime: Optional[float] = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                object.__setattr__(self, key, value)
            else:
                raise AttributeError(f"'ModelOutputCollection' object has no attribute '{key}'")