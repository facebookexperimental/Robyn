# pyre-strict

import numpy as np
import pandas as pd

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class HyperCollectorOutput:
    hyper_list_all: Hyperparameters
    hyper_bound_list_updated: Hyperparameters
    hyper_bound_list_fixed: Hyperparameters
    dt_hyper_fixed_mod: pd.DataFrame
    all_fixed: bool


@dataclass(frozen=True)
class ModelDecompOutput:
    x_decomp_vec: pd.DataFrame
    x_decomp_vec_scaled: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    coefs_out_cat: pd.DataFrame
    media_decomp_immediate: pd.DataFrame
    media_decomp_carryover: pd.DataFrame


class RidgeModelBuilder:

    def __init__(
        self,
        mmm_data: MMMData,
        holiday_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
    ) -> None:
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.calibration_input = calibration_input
        self.hyperparameters = hyperparameters
        self.featurized_mmm_data = featurized_mmm_data

    def build_models(
        self,
        trials_config: TrialsConfig,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        seed: int = 123,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[List[float]] = None,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        adstock: AdstockType = AdstockType.GEOMETRIC,
    ) -> ModelOutputs:
        # Implementation here
        pass

    def _model_train(
        self,
        hyper_collect: Hyperparameters,
        trials_config: TrialsConfig,
        cores: int,
        intercept_sign: str,
        intercept: bool,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        rssd_zero_penalty: bool = True,
        seed: int = 123,
    ) -> ModelOutputs:
        # Implementation here
        pass

    def run_nevergrad_optimization(
        self,
        hyper_collect: Hyperparameters,
        iterations: int,
        cores: int,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        rssd_zero_penalty: bool = True,
        trial: int = 1,
        seed: int = 123,
    ) -> List[Trial]:
        # Implementation here
        pass

    @staticmethod
    def model_decomp(
        coefs: Dict[str, float],
        y_pred: np.ndarray,
        dt_mod_saturated: pd.DataFrame,
        dt_saturated_immediate: pd.DataFrame,
        dt_saturated_carryover: pd.DataFrame,
        dt_mod_roll_wind: pd.DataFrame,
        refresh_added_start: str,
    ) -> ModelDecompOutput:
        # Implementation here
        pass

    @staticmethod
    def _model_refit(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        lambda_: float = 1.0,
        lower_limits: Optional[List[float]] = None,
        upper_limits: Optional[List[float]] = None,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
    ) -> ModelRefitOutput:
        # Implementation here
        pass

    @staticmethod
    def _lambda_seq(
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> np.ndarray:
        # Implementation here
        pass

    @staticmethod
    def _hyper_collector(
        adstock: str,
        all_media: List[str],
        paid_media_spends: List[str],
        organic_vars: List[str],
        prophet_vars: List[str],
        context_vars: List[str],
        dt_mod: pd.DataFrame,
        hyper_in: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        cores: int = 1,
    ) -> HyperCollectorOutput:
        # Implementation here
        pass
