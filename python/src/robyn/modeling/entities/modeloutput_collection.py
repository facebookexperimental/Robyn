from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from robyn.modeling.entities.modeloutput import ModelOutput, ResultHypParam, XDecompAgg


@dataclass
class ModelOutputCollection:

    # Group 1: Model Results
    resultHypParam: Optional[ResultHypParam] = None
    xDecompAgg: Optional[XDecompAgg] = None
    mediaVecCollect: Optional[pd.DataFrame] = None
    xDecompVecCollect: Optional[pd.DataFrame] = None
    resultCalibration: Optional[pd.DataFrame] = None
    model_output: Optional[ModelOutput] = None

    # Group 2: Model Solutions
    allSolutions: List[str] = field(default_factory=list)
    allPareto: Dict[str, Any] = field(default_factory=dict)
    calibration_constraint: float = 0.0
    pareto_fronts: int = 0
    selectID: Optional[str] = None

    # Group 3: Model Configuration
    cores: int = 0
    iterations: int = 0
    trials: List[Any] = field(default_factory=list)
    intercept_sign: str = ""
    nevergrad_algo: str = ""
    add_penalty_factor: bool = False
    seed: int = 0
    hyper_fixed: bool = False
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
                raise AttributeError(
                    f"'ModelOutputCollection' object has no attribute '{key}'"
                )
