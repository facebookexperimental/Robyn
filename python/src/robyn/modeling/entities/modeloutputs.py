# pyre-strict

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import field


@dataclass
class Trial:
    result_hyp_param: pd.DataFrame
    decomp_spend_dist: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    nrmse: float
    decomp_rssd: float
    mape: float
    rsq_train: float
    rsq_val: float
    rsq_test: float
    lambda_: float
    lambda_hp: float
    lambda_max: float
    lambda_min_ratio: float
    pos: int
    nrmse_train: float = 0.0
    nrmse_val: float = 0.0
    nrmse_test: float = 0.0
    elapsed: float = 0.0
    elapsed_accum: float = 0.0
    trial: int = 1
    iter_ng: int = 1
    iter_par: int = 1
    train_size: float = 1.0
    sol_id: str = "1_1_1"
    lift_calibration: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ModelOutputs:
    """
    Represents the overall outputs of the modeling process.

    This class contains the results of all trials, along with metadata about the model run.

    Attributes:
        trials (List[Trial]): List of all trials run during the modeling process.
        train_timestamp (str): Timestamp of when the model was trained.
        cores (int): Number of CPU cores used for training.
        iterations (int): Number of iterations per trial.
        intercept (bool): Whether an intercept was included in the model.
        intercept_sign (str): Sign constraint applied to the intercept.
        nevergrad_algo (str): Nevergrad algorithm used for optimization.
        ts_validation (bool): Whether time series validation was used.
        add_penalty_factor (bool): Whether a penalty factor was added.
        hyper_updated (Dict[str, Any]): Updated hyperparameters.
        hyper_fixed (bool): Whether hyperparameters were fixed.
        convergence (Dict[str, Any]): Convergence information for the optimization process.
        select_id (str): ID of the selected model.
        seed (int): Random seed used for reproducibility.
        hyper_bound_ng (Dict[str, Any]): Hyperparameter bounds for Nevergrad optimization.
        hyper_bound_fixed (Dict[str, Any]): Fixed hyperparameter bounds.
        ts_validation_plot (Optional[str]): Time series validation plot, if applicable.
        all_result_hyp_param (pd.DataFrame): Aggregated hyperparameter results from all trials.
        all_x_decomp_agg (pd.DataFrame): Aggregated decomposition results from all trials.
        all_decomp_spend_dist (pd.DataFrame): Aggregated spend distribution results from all trials.
    """

    trials: List[Trial]
    train_timestamp: str
    cores: int
    iterations: int
    intercept: bool
    intercept_sign: str
    nevergrad_algo: str
    ts_validation: bool
    add_penalty_factor: bool
    hyper_updated: Dict[str, Any]
    hyper_fixed: bool
    select_id: str
    hyper_bound_ng: Dict[str, Any]  # Move non-default arguments before default ones
    hyper_bound_fixed: Dict[str, Any]
    seed: List[int] = field(default_factory=lambda: [123])  # Ensure seed is a list
    convergence: Dict[str, Any] = field(default_factory=dict)  # Default argument
    ts_validation_plot: List[Optional[str]] = field(
        default_factory=list
    )  # Ensure ts_validation_plot is a list
    all_result_hyp_param: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_x_decomp_agg: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_decomp_spend_dist: pd.DataFrame = field(default_factory=pd.DataFrame)
