# pyre-strict

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class Trial:
    """
    Represents the results of a single model trial.

    Attributes:
        result_hyp_param (pd.DataFrame): Hyperparameters and their values for this trial.
        x_decomp_agg (pd.DataFrame): Aggregated decomposition results for this trial.
        lift_calibration (pd.DataFrame): Lift calibration results for this trial.
        decomp_spend_dist (pd.DataFrame): Decomposition spend distribution for this trial.
        nrmse (float): Normalized Root Mean Square Error for this trial.
        decomp_rssd (float): Decomposition Root Sum Squared Distance for this trial.
        mape (float): Mean Absolute Percentage Error for this trial.
    """

    result_hyp_param: pd.DataFrame
    lift_calibration: Optional[pd.DataFrame]
    decomp_spend_dist: Optional[pd.DataFrame]
    nrmse: float
    decomp_rssd: float
    mape: float
    x_decomp_agg: pd.DataFrame
    rsq_train: float
    rsq_val: float
    rsq_test: float
    lambda_: float
    lambda_hp: float
    lambda_max: float
    lambda_min_ratio: float
    pos: int
    elapsed: float
    elapsed_accum: float
    sol_id: str
    trial: int
    iter_ng: int
    iter_par: int
    train_size: float
    solID: str


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
        ts_validation_plot (Any): Time series validation plot (if applicable).
        select_id (str): ID of the selected model.
        seed (int): Random seed used for reproducibility.
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
    convergence: Dict[str, Any]
    ts_validation_plot: Any  # This could be a matplotlib figure or other plot object
    select_id: str
    seed: int
    hyper_bound_ng: Dict[str, Any]  # For hyperBoundNG
    hyper_bound_fixed: Dict[str, Any]  # For hyperBoundFixed
