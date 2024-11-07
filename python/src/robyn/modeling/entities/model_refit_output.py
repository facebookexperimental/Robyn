# model_refit_output.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class ModelRefitOutput:
    rsq_train: float
    rsq_val: Optional[float]
    rsq_test: Optional[float]
    nrmse_train: float
    nrmse_val: Optional[float]
    nrmse_test: Optional[float]
    coefs: np.ndarray
    y_train_pred: np.ndarray
    y_val_pred: Optional[np.ndarray]
    y_test_pred: Optional[np.ndarray]
    y_pred: np.ndarray
    mod: Ridge
    df_int: int
    lambda_: float
    lambda_hp: float
    lambda_max: float
    lambda_min_ratio: float
