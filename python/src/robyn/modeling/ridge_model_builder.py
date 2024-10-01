import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import nevergrad as ng
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm


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


class RidgeModelBuilder:
    def __init__(
        self,
        mmm_data: MMMData,
        holiday_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
    ):
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
        cores: int = 2,
    ) -> ModelOutputs:
        hyper_collect = self._hyper_collector(
            self.hyperparameters,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        output_models = self._model_train(
            hyper_collect,
            trials_config,
            intercept_sign,
            intercept,
            nevergrad_algo,
            dt_hyper_fixed,
            ts_validation,
            add_penalty_factor,
            objective_weights,
            rssd_zero_penalty,
            seed,
            cores,
        )

        return ModelOutputs(output_models)

    def _model_train(
        self,
        hyper_collect: Dict[str, Any],
        trials_config: TrialsConfig,
        intercept_sign: str,
        intercept: bool,
        nevergrad_algo: NevergradAlgorithm,
        dt_hyper_fixed: Optional[pd.DataFrame],
        ts_validation: bool,
        add_penalty_factor: bool,
        objective_weights: Optional[List[float]],
        rssd_zero_penalty: bool,
        seed: int,
        cores: int,
    ) -> List[Trial]:
        trials = []
        for trial in range(1, trials_config.trials + 1):
            print(f"Running trial {trial} of {trials_config.trials}")
            trial_result = self._run_nevergrad_optimization(
                hyper_collect,
                trials_config.iterations,
                cores,
                nevergrad_algo,
                intercept,
                intercept_sign,
                ts_validation,
                add_penalty_factor,
                objective_weights,
                dt_hyper_fixed,
                rssd_zero_penalty,
                trial,
                seed + trial,
            )
            trials.append(trial_result)
        return trials

    def _run_nevergrad_optimization(
        self,
        hyper_collect: Dict[str, Any],
        iterations: int,
        cores: int,
        nevergrad_algo: NevergradAlgorithm,
        intercept: bool,
        intercept_sign: str,
        ts_validation: bool,
        add_penalty_factor: bool,
        objective_weights: Optional[List[float]],
        dt_hyper_fixed: Optional[pd.DataFrame],
        rssd_zero_penalty: bool,
        trial: int,
        seed: int,
    ) -> Trial:
        np.random.seed(seed)

        param_names = list(hyper_collect["hyper_bound_list_updated"].keys())
        param_bounds = [hyper_collect["hyper_bound_list_updated"][name] for name in param_names]

        # Create a dictionary of parameters for Nevergrad
        instrum_dict = {
            name: ng.p.Scalar(lower=bound[0], upper=bound[1]) for name, bound in zip(param_names, param_bounds)
        }

        # Create the instrumentation
        instrum = ng.p.Instrumentation(**instrum_dict)

        optimizer = ng.optimizers.registry[nevergrad_algo.value](instrum, budget=iterations, num_workers=cores)

        best_loss = float("inf")
        best_params = None

        with tqdm(total=iterations, desc=f"Trial {trial}") as pbar:
            for _ in range(iterations):
                candidate = optimizer.ask()

                # Extract parameters from the candidate
                params = {name: candidate.args[0][name] for name in param_names}

                loss = self._evaluate_model(
                    params, ts_validation, add_penalty_factor, rssd_zero_penalty, objective_weights
                )

                optimizer.tell(candidate, loss)

                if loss < best_loss:
                    best_loss = loss
                    best_params = params

                pbar.update(1)

        return Trial(best_params, best_loss)

    def _prepare_data(self, params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        # Implementation depends on your data structure and transformations
        # This is a placeholder implementation
        X = self.featurized_mmm_data.dt_mod.drop(columns=[self.mmm_data.mmmdata_spec.dep_var])
        y = self.featurized_mmm_data.dt_mod[self.mmm_data.mmmdata_spec.dep_var]
        return X, y

    def _calculate_rssd(self, coefs: np.ndarray, rssd_zero_penalty: bool) -> float:
        rssd = np.sqrt(np.sum(coefs**2))
        if rssd_zero_penalty:
            zero_coef_ratio = np.sum(coefs == 0) / len(coefs)
            rssd *= 1 + zero_coef_ratio
        return rssd

    def _calculate_mape(self, model: Ridge) -> float:
        # Implementation depends on your calibration data structure
        # This is a placeholder implementation
        return 0.0

    def _evaluate_model(
        self,
        params: Dict[str, float],
        ts_validation: bool,
        add_penalty_factor: bool,
        rssd_zero_penalty: bool,
        objective_weights: Optional[List[float]],
    ) -> float:
        # Prepare data
        X, y = self._prepare_data(params)

        if ts_validation:
            train_size = params.get("train_size", 0.8)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None

        # Fit model
        lambda_ = params.get("lambda", 1.0)
        model = Ridge(alpha=lambda_, fit_intercept=True)
        model.fit(X_train, y_train)

        # Calculate metrics
        y_train_pred = model.predict(X_train)
        nrmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2)) / (np.max(y_train) - np.min(y_train))

        if ts_validation:
            y_test_pred = model.predict(X_test)
            nrmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2)) / (np.max(y_test) - np.min(y_test))
        else:
            nrmse_test = None

        decomp_rssd = self._calculate_rssd(model.coef_, rssd_zero_penalty)

        # Calculate MAPE if calibration data is provided
        mape = self._calculate_mape(model) if self.calibration_input else None

        # Combine metrics into a single loss value
        if objective_weights is None:
            objective_weights = [1 / 3, 1 / 3, 1 / 3] if mape is not None else [0.5, 0.5]

        loss = (
            objective_weights[0] * nrmse_train
            + objective_weights[1] * decomp_rssd
            + (objective_weights[2] * mape if mape is not None else 0)
        )

        return loss

    @staticmethod
    def _hyper_collector(
        hyperparameters: Hyperparameters,
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: int,
    ) -> Dict[str, Any]:
        # Implement hyper_collector logic here
        # This should prepare the hyperparameters for optimization

        # Placeholder implementation
        hyper_collect = {
            "hyper_list_all": hyperparameters.hyperparameters,
            "hyper_bound_list_updated": {},
            "hyper_bound_list_fixed": {},
            "dt_hyper_fixed_mod": pd.DataFrame(),
            "all_fixed": False,
        }

        for name, value in hyperparameters.hyperparameters.items():
            if isinstance(value, list) and len(value) == 2:
                hyper_collect["hyper_bound_list_updated"][name] = value
            else:
                hyper_collect["hyper_bound_list_fixed"][name] = value

        if dt_hyper_fixed is not None:
            hyper_collect["dt_hyper_fixed_mod"] = dt_hyper_fixed
            hyper_collect["all_fixed"] = True

        return hyper_collect

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
        model = Ridge(alpha=lambda_, fit_intercept=intercept)
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val) if x_val is not None else None
        y_test_pred = model.predict(x_test) if x_test is not None else None

        rsq_train = r2_score(y_train, y_train_pred)
        rsq_val = r2_score(y_val, y_val_pred) if y_val is not None else None
        rsq_test = r2_score(y_test, y_test_pred) if y_test is not None else None

        nrmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2)) / (np.max(y_train) - np.min(y_train))
        nrmse_val = (
            np.sqrt(np.mean((y_val - y_val_pred) ** 2)) / (np.max(y_val) - np.min(y_val))
            if y_val is not None
            else None
        )
        nrmse_test = (
            np.sqrt(np.mean((y_test - y_test_pred) ** 2)) / (np.max(y_test) - np.min(y_test))
            if y_test is not None
            else None
        )

        return ModelRefitOutput(
            rsq_train=rsq_train,
            rsq_val=rsq_val,
            rsq_test=rsq_test,
            nrmse_train=nrmse_train,
            nrmse_val=nrmse_val,
            nrmse_test=nrmse_test,
            coefs=model.coef_,
            y_train_pred=y_train_pred,
            y_val_pred=y_val_pred,
            y_test_pred=y_test_pred,
            y_pred=(
                np.concatenate([y_train_pred, y_val_pred, y_test_pred])
                if y_val is not None and y_test is not None
                else y_train_pred
            ),
            mod=model,
            df_int=1 if intercept else 0,
        )

    @staticmethod
    def _lambda_seq(
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> np.ndarray:
        lambda_max = np.max(np.abs(np.sum(x * y, axis=0))) / (0.001 * x.shape[0])
        return np.logspace(np.log10(lambda_max * lambda_min_ratio), np.log10(lambda_max), num=seq_len)
