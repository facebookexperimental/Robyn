# pyre-strict

import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import nevergrad as ng
from tqdm import tqdm
from robyn.calibration.calibration import CalibrationEngine
import logging
import time
from datetime import datetime
from robyn.modeling.convergence.convergence import Convergence
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64


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
        self.logger = logging.getLogger(__name__)

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
        start_time = time.time()

        # Initialize hyperparameters
        hyper_collect = self._hyper_collector(
            self.hyperparameters, ts_validation, add_penalty_factor, dt_hyper_fixed, cores
        )

        # Set up objective weights including calibration if available
        if objective_weights is None:
            if self.calibration_input is not None:
                objective_weights = [1 / 3, 1 / 3, 1 / 3]  # NRMSE, RSSD, MAPE
            else:
                objective_weights = [0.5, 0.5]  # NRMSE, RSSD only

        # Run trials
        trials = []
        for trial in range(1, trials_config.trials + 1):
            trial_result = self._run_nevergrad_optimization(
                hyper_collect=hyper_collect,
                iterations=trials_config.iterations,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                intercept=intercept,
                intercept_sign=intercept_sign,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                objective_weights=objective_weights,
                dt_hyper_fixed=dt_hyper_fixed,
                rssd_zero_penalty=rssd_zero_penalty,
                trial=trial,
                seed=seed + trial,
                total_trials=trials_config.trials,
            )
            trials.append(trial_result)

        # Calculate convergence
        convergence = Convergence()
        convergence_results = convergence.calculate_convergence(trials)

        # Aggregate results
        all_result_hyp_param = pd.concat([trial.result_hyp_param for trial in trials], ignore_index=True)
        all_x_decomp_agg = pd.concat([trial.x_decomp_agg for trial in trials], ignore_index=True)
        all_decomp_spend_dist = pd.concat(
            [trial.decomp_spend_dist for trial in trials if trial.decomp_spend_dist is not None], ignore_index=True
        )

        # Create ModelOutputs
        model_outputs = ModelOutputs(
            trials=trials,
            train_timestamp=datetime.now(),
            cores=cores,
            iterations=trials_config.iterations,
            intercept=intercept,
            intercept_sign=intercept_sign,
            nevergrad_algo=nevergrad_algo,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            hyper_updated=hyper_collect["hyper_list_all"],
            hyper_fixed=hyper_collect["all_fixed"],
            convergence=convergence_results,
            select_id=self._select_best_model(trials),
            seed=seed,
            hyper_bound_ng=hyper_collect["hyper_bound_list_updated"],
            hyper_bound_fixed=hyper_collect["hyper_bound_list_fixed"],
            ts_validation_plot=None,
            all_result_hyp_param=all_result_hyp_param,
            all_x_decomp_agg=all_x_decomp_agg,
            all_decomp_spend_dist=all_decomp_spend_dist,
        )

        return model_outputs

    def _select_best_model(self, output_models: List[Trial]) -> str:
        # Extract relevant metrics
        nrmse_values = np.array([trial.nrmse for trial in output_models])
        decomp_rssd_values = np.array([trial.decomp_rssd for trial in output_models])

        # Normalize the metrics
        nrmse_norm = (nrmse_values - np.min(nrmse_values)) / (np.max(nrmse_values) - np.min(nrmse_values))
        decomp_rssd_norm = (decomp_rssd_values - np.min(decomp_rssd_values)) / (
            np.max(decomp_rssd_values) - np.min(decomp_rssd_values)
        )

        # Calculate the combined score (assuming equal weights)
        combined_score = nrmse_norm + decomp_rssd_norm

        # Find the index of the best model (lowest combined score)
        best_index = np.argmin(combined_score)

        # Return the solID of the best model
        return output_models[best_index].result_hyp_param["solID"].values[0]

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
                trials_config.trials,
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
        total_trials: int,
    ) -> Trial:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        np.random.seed(seed)

        param_names = list(hyper_collect["hyper_bound_list_updated"].keys())
        param_bounds = [hyper_collect["hyper_bound_list_updated"][name] for name in param_names]

        instrum_dict = {
            name: ng.p.Scalar(lower=bound[0], upper=bound[1]) for name, bound in zip(param_names, param_bounds)
        }

        instrum = ng.p.Instrumentation(**instrum_dict)

        optimizer = ng.optimizers.registry[nevergrad_algo.value](instrum, budget=iterations, num_workers=cores)

        all_results = []
        start_time = time.time()
        with tqdm(
            total=iterations,
            desc=f"Running trial {trial} of total {total_trials} trials",
            bar_format="{l_bar}{bar}",
            ncols=75,
        ) as pbar:
            for iter_ng in range(iterations):
                candidate = optimizer.ask()
                params = candidate.kwargs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self._evaluate_model(
                        params,
                        ts_validation,
                        add_penalty_factor,
                        rssd_zero_penalty,
                        objective_weights,
                        start_time=start_time,
                        iter_ng=iter_ng,
                        trial=trial,
                    )
                optimizer.tell(candidate, result["loss"])
                result["params"].update(
                    {
                        "solID": f"{trial}_{iter_ng + 1}_1",
                        "ElapsedAccum": result["elapsed_accum"],
                        "trial": trial,
                        "nrmse": result["nrmse"],
                        "decomp.rssd": result["decomp_rssd"],
                        "mape": result["mape"],
                    }
                )
                all_results.append(result)
                pbar.update(1)

        end_time = time.time()
        self.logger.info(f" Finished in {(end_time - start_time) / 60:.2f} mins")

        # Aggregate results from all iterations
        result_hyp_param = pd.DataFrame([r["params"] for r in all_results])
        decomp_spend_dist = pd.concat([r["decomp_spend_dist"] for r in all_results], ignore_index=True)
        x_decomp_agg = pd.concat([r["x_decomp_agg"] for r in all_results], ignore_index=True)

        # Find the best result for single-value metrics
        best_result = min(all_results, key=lambda x: x["loss"])

        return Trial(
            result_hyp_param=result_hyp_param,
            lift_calibration=best_result["lift_calibration"],
            decomp_spend_dist=decomp_spend_dist,
            nrmse=best_result["nrmse"],
            decomp_rssd=best_result["decomp_rssd"],
            mape=best_result["mape"],
            x_decomp_agg=x_decomp_agg,
            rsq_train=best_result["rsq_train"],
            rsq_val=best_result["rsq_val"],
            rsq_test=best_result["rsq_test"],
            lambda_=best_result["lambda_"],
            lambda_hp=best_result["lambda_hp"],
            lambda_max=best_result["lambda_max"],
            lambda_min_ratio=best_result["lambda_min_ratio"],
            pos=best_result["pos"],
            elapsed=best_result["elapsed"],
            elapsed_accum=best_result["elapsed_accum"],
            trial=trial,
            iter_ng=best_result["iter_ng"],
            iter_par=best_result["iter_par"],
            train_size=best_result["params"].get("train_size", 1.0),
            sol_id=best_result["params"]["solID"],
        )

    def _calculate_decomp_spend_dist(
        self, model: Ridge, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]
    ) -> pd.DataFrame:
        paid_media_cols = [col for col in X.columns if col in self.mmm_data.mmmdata_spec.paid_media_spends]
        x_decomp = X[paid_media_cols] * model.coef_[X.columns.get_indexer(paid_media_cols)]

        decomp_spend_dist = pd.DataFrame(
            {
                "rn": paid_media_cols,
                "coef": model.coef_[X.columns.get_indexer(paid_media_cols)],
                "xDecompAgg": x_decomp.sum(),
                "xDecompPerc": x_decomp.sum() / x_decomp.sum().sum(),
                "xDecompMeanNon0": x_decomp[x_decomp > 0].mean(),
                "xDecompMeanNon0Perc": x_decomp[x_decomp > 0].mean() / x_decomp[x_decomp > 0].sum(),
                "xDecompAggRF": x_decomp.sum(),  # You may need to adjust this for refresh
                "xDecompPercRF": x_decomp.sum() / x_decomp.sum().sum(),  # Adjust for refresh
                "xDecompMeanNon0RF": x_decomp[x_decomp > 0].mean(),  # Adjust for refresh
                "xDecompMeanNon0PercRF": x_decomp[x_decomp > 0].mean()
                / x_decomp[x_decomp > 0].sum(),  # Adjust for refresh
                "pos": model.coef_[X.columns.get_indexer(paid_media_cols)] >= 0,
                "mean_spend": X[paid_media_cols].mean(),
                "total_spend": X[paid_media_cols].sum(),
                "spend_share": X[paid_media_cols].sum() / X[paid_media_cols].sum().sum(),
                "spend_share_refresh": X[paid_media_cols].sum() / X[paid_media_cols].sum().sum(),  # Adjust for refresh
                "effect_share": x_decomp.sum() / x_decomp.sum().sum(),
                "effect_share_refresh": x_decomp.sum() / x_decomp.sum().sum(),  # Adjust for refresh
            }
        )

        # Add model performance metrics
        decomp_spend_dist["rsq_train"] = r2_score(y, model.predict(X))
        decomp_spend_dist["rsq_val"] = params.get("rsq_val", 0)
        decomp_spend_dist["rsq_test"] = params.get("rsq_test", 0)
        decomp_spend_dist["nrmse_train"] = np.sqrt(mean_squared_error(y, model.predict(X))) / (y.max() - y.min())
        decomp_spend_dist["nrmse_val"] = params.get("nrmse_val", 0)
        decomp_spend_dist["nrmse_test"] = params.get("nrmse_test", 0)
        decomp_spend_dist["nrmse"] = params.get("nrmse", 0)
        decomp_spend_dist["decomp.rssd"] = params.get("decomp_rssd", 0)
        decomp_spend_dist["mape"] = params.get("mape", 0)
        decomp_spend_dist["lambda"] = params.get("lambda_", 0)
        decomp_spend_dist["lambda_hp"] = params.get("lambda_hp", 0)
        decomp_spend_dist["lambda_max"] = params.get("lambda_max", 0)
        decomp_spend_dist["lambda_min_ratio"] = params.get("lambda_min_ratio", 0)
        decomp_spend_dist["solID"] = params.get("solID", "")
        decomp_spend_dist["trial"] = params.get("trial", 0)
        decomp_spend_dist["iterNG"] = params.get("iter_ng", 0)
        decomp_spend_dist["iterPar"] = params.get("iter_par", 0)

        return decomp_spend_dist

    def _calculate_x_decomp_agg(
        self, model: Ridge, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]
    ) -> pd.DataFrame:
        x_decomp = X * model.coef_
        x_decomp_agg = pd.DataFrame(
            {
                "rn": X.columns,
                "coef": model.coef_,
                "xDecompAgg": x_decomp.sum(),
                "xDecompPerc": x_decomp.sum() / x_decomp.sum().sum(),
                "xDecompMeanNon0": x_decomp[x_decomp > 0].mean(),
                "xDecompMeanNon0Perc": x_decomp[x_decomp > 0].mean() / x_decomp[x_decomp > 0].sum(),
                "xDecompAggRF": x_decomp.sum(),  # You may need to adjust this for refresh
                "xDecompPercRF": x_decomp.sum() / x_decomp.sum().sum(),  # Adjust for refresh
                "xDecompMeanNon0RF": x_decomp[x_decomp > 0].mean(),  # Adjust for refresh
                "xDecompMeanNon0PercRF": x_decomp[x_decomp > 0].mean()
                / x_decomp[x_decomp > 0].sum(),  # Adjust for refresh
                "pos": model.coef_ >= 0,
            }
        )

        # Add model performance metrics and parameters
        x_decomp_agg["train_size"] = params.get("train_size", 1.0)
        x_decomp_agg["rsq_train"] = r2_score(y, model.predict(X))
        x_decomp_agg["rsq_val"] = params.get("rsq_val", 0)
        x_decomp_agg["rsq_test"] = params.get("rsq_test", 0)
        x_decomp_agg["nrmse_train"] = np.sqrt(mean_squared_error(y, model.predict(X))) / (y.max() - y.min())
        x_decomp_agg["nrmse_val"] = params.get("nrmse_val", 0)
        x_decomp_agg["nrmse_test"] = params.get("nrmse_test", 0)
        x_decomp_agg["nrmse"] = params.get("nrmse", 0)
        x_decomp_agg["decomp.rssd"] = params.get("decomp_rssd", 0)
        x_decomp_agg["mape"] = params.get("mape", 0)
        x_decomp_agg["lambda"] = params.get("lambda_", 0)
        x_decomp_agg["lambda_hp"] = params.get("lambda_hp", 0)
        x_decomp_agg["lambda_max"] = params.get("lambda_max", 0)
        x_decomp_agg["lambda_min_ratio"] = params.get("lambda_min_ratio", 0)
        x_decomp_agg["solID"] = params.get("solID", "")
        x_decomp_agg["trial"] = params.get("trial", 0)
        x_decomp_agg["iterNG"] = params.get("iter_ng", 0)
        x_decomp_agg["iterPar"] = params.get("iter_par", 0)

        return x_decomp_agg

    def _prepare_data(self, params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        # Get the dependent variable
        # Check if 'dep_var' is in columns
        if "dep_var" in self.featurized_mmm_data.dt_mod.columns:
            # Rename 'dep_var' to the specified value
            self.featurized_mmm_data.dt_mod = self.featurized_mmm_data.dt_mod.rename(
                columns={"dep_var": self.mmm_data.mmmdata_spec.dep_var}
            )
        y = self.featurized_mmm_data.dt_mod[self.mmm_data.mmmdata_spec.dep_var]

        # Select all columns except the dependent variable
        X = self.featurized_mmm_data.dt_mod.drop(columns=[self.mmm_data.mmmdata_spec.dep_var])

        # Convert date columns to numeric (number of days since the earliest date)
        date_columns = X.select_dtypes(include=["datetime64", "object"]).columns
        for col in date_columns:
            X[col] = pd.to_datetime(X[col], errors="coerce", format="%Y-%m-%d")
            # Fill NaT (Not a Time) values with a default date (e.g., the minimum date in the column)
            min_date = X[col].min()
            X[col] = X[col].fillna(min_date)
            # Convert to days since minimum date, handling potential NaT values
            X[col] = (X[col] - min_date).dt.total_seconds().div(86400).fillna(0).astype(int)

        # One-hot encode categorical variables
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        # Ensure all columns are numeric
        X = X.select_dtypes(include=[np.number])

        # Apply transformations based on hyperparameters
        for media in self.mmm_data.mmmdata_spec.paid_media_spends:
            if f"{media}_thetas" in params:
                X[media] = self._geometric_adstock(X[media], params[f"{media}_thetas"])
            if f"{media}_alphas" in params and f"{media}_gammas" in params:
                X[media] = self._hill_transformation(X[media], params[f"{media}_alphas"], params[f"{media}_gammas"])

        # Handle any remaining NaN or infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
        X = X + 1e-8 * np.random.randn(*X.shape)

        return X, y

    def _geometric_adstock(self, x: pd.Series, theta: float) -> pd.Series:
        y = x.copy()
        for i in range(1, len(x)):
            y.iloc[i] += theta * y.iloc[i - 1]
        return y

    def _hill_transformation(self, x: pd.Series, alpha: float, gamma: float) -> pd.Series:
        x_scaled = (x - x.min()) / (x.max() - x.min())
        return x_scaled**alpha / (x_scaled**alpha + gamma**alpha)

    def _calculate_rssd(self, coefs: np.ndarray, rssd_zero_penalty: bool) -> float:
        rssd = np.sqrt(np.sum(coefs**2))
        if rssd_zero_penalty:
            zero_coef_ratio = np.sum(coefs == 0) / len(coefs)
            rssd *= 1 + zero_coef_ratio
        return rssd

    def _select_best_model(self, output_models: List[Trial]) -> str:
        # Extract relevant metrics
        nrmse_values = np.array([trial.nrmse for trial in output_models])
        decomp_rssd_values = np.array([trial.decomp_rssd for trial in output_models])

        # Normalize the metrics
        nrmse_norm = (nrmse_values - np.min(nrmse_values)) / (np.max(nrmse_values) - np.min(nrmse_values))
        decomp_rssd_norm = (decomp_rssd_values - np.min(decomp_rssd_values)) / (
            np.max(decomp_rssd_values) - np.min(decomp_rssd_values)
        )

        # Calculate the combined score (assuming equal weights)
        combined_score = nrmse_norm + decomp_rssd_norm

        # Find the index of the best model (lowest combined score)
        best_index = np.argmin(combined_score)

        # Return the solID of the best model
        return output_models[best_index].sol_id

    def _calculate_mape(
        self, model: Ridge, dt_raw: pd.DataFrame, hypParamSam: Dict[str, float], wind_start: int, wind_end: int
    ) -> float:
        """
        Calculate MAPE using calibration data, following Robyn's calibration logic
        """
        if self.calibration_input is None:
            return 0.0

        # Initialize calibration engine
        calibration_engine = CalibrationEngine(
            mmm_data=self.mmm_data, hyperparameters=self.hyperparameters, calibration_input=self.calibration_input
        )

        # Calculate MAPE using calibration engine
        lift_collect = calibration_engine.calibrate(
            df_raw=dt_raw,
            hypParamSam=hypParamSam,
            wind_start=wind_start,
            wind_end=wind_end,
            dayInterval=self.mmm_data.mmmdata_spec.day_interval,
            adstock=self.hyperparameters.adstock,
        )

        # Return mean MAPE across all lift studies
        if lift_collect is not None and not lift_collect.empty:
            return float(lift_collect["mape_lift"].mean())
        return 0.0

    def _evaluate_model(
        self,
        params: Dict[str, float],
        ts_validation: bool,
        add_penalty_factor: bool,
        rssd_zero_penalty: bool,
        objective_weights: Optional[List[float]],
        start_time: float,
        iter_ng: int,
        trial: int,
    ) -> Dict[str, Any]:
        train_size = params.get("train_size", 1.0) if ts_validation else 1.0
        X, y = self._prepare_data(params)

        # Split data for time series validation if enabled
        if ts_validation:
            train_idx = int(len(X) * train_size)
            val_test_size = (len(X) - train_idx) // 2

            X_train = X[:train_idx]
            y_train = y[:train_idx]
            X_val = X[train_idx : train_idx + val_test_size]
            y_val = y[train_idx : train_idx + val_test_size]
            X_test = X[train_idx + val_test_size :]
            y_test = y[train_idx + val_test_size :]
        else:
            X_train, y_train = X, y
            X_val = X_test = y_val = y_test = None

        # Fit model
        lambda_ = params.get("lambda", 1.0)
        model = Ridge(alpha=lambda_, fit_intercept=True)
        model.fit(X_train, y_train)

        # Calculate base metrics
        y_train_pred = model.predict(X_train)
        nrmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred)) / (y_train.max() - y_train.min())
        rsq_train = r2_score(y_train, y_train_pred)

        if ts_validation and X_val is not None and X_test is not None:
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            nrmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred)) / (y_val.max() - y_val.min())
            nrmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred)) / (y_test.max() - y_test.min())
            rsq_val = r2_score(y_val, y_val_pred)
            rsq_test = r2_score(y_test, y_test_pred)
            nrmse = nrmse_val
        else:
            nrmse_val = nrmse_test = rsq_val = rsq_test = None
            nrmse = nrmse_train

        # Calculate RSSD
        decomp_rssd = self._calculate_rssd(model.coef_, rssd_zero_penalty)

        # Calculate MAPE using calibration if available
        if self.calibration_input is not None:
            mape = self._calculate_mape(
                model=model,
                dt_raw=self.featurized_mmm_data.dt_mod,
                hypParamSam=params,
                wind_start=self.mmm_data.mmmdata_spec.rolling_window_start_which,
                wind_end=self.mmm_data.mmmdata_spec.rolling_window_end_which,
            )
            calibration_constraint = 0.1  # Default calibration constraint
            if mape > calibration_constraint:
                # Penalize models that don't meet calibration constraint
                decomp_rssd *= 1 + (mape - calibration_constraint)
        else:
            mape = 0.0

        # Calculate final loss based on objectives
        if objective_weights is None:
            objective_weights = [1 / 3, 1 / 3, 1 / 3] if self.calibration_input else [0.5, 0.5]

        loss = (
            objective_weights[0] * nrmse
            + objective_weights[1] * decomp_rssd
            + (objective_weights[2] * mape if self.calibration_input else 0)
        )

        # Prepare result parameters
        result_params = {
            **params,
            "train_size": train_size,
            "rsq_train": rsq_train,
            "rsq_val": rsq_val if rsq_val is not None else 0,
            "rsq_test": rsq_test if rsq_test is not None else 0,
            "nrmse_train": nrmse_train,
            "nrmse_val": nrmse_val if nrmse_val is not None else 0,
            "nrmse_test": nrmse_test if nrmse_test is not None else 0,
            "nrmse": nrmse,
            "decomp.rssd": decomp_rssd,
            "mape": mape,
            "lambda": lambda_,
            "solID": f"{trial}_{iter_ng + 1}_1",
            "trial": trial,
            "iterNG": iter_ng + 1,
            "iterPar": 1,
        }

        # Calculate decomposition results
        decomp_spend_dist = self._calculate_decomp_spend_dist(model, X_train, y_train, result_params)
        x_decomp_agg = self._calculate_x_decomp_agg(model, X_train, y_train, result_params)

        elapsed = time.time() - start_time

        return {
            "loss": loss,
            "params": result_params,
            "nrmse": nrmse,
            "decomp_rssd": decomp_rssd,
            "mape": mape,
            "decomp_spend_dist": decomp_spend_dist,
            "x_decomp_agg": x_decomp_agg,
            "rsq_train": rsq_train,
            "rsq_val": rsq_val if rsq_val is not None else 0,
            "rsq_test": rsq_test if rsq_test is not None else 0,
            "lambda_": lambda_,
            "elapsed": elapsed,
            "elapsed_accum": elapsed,
            "iter_ng": iter_ng + 1,
            "iter_par": 1,
        }

    @staticmethod
    def _hyper_collector(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: int,
    ) -> Dict[str, Any]:
        logger = logging.getLogger(__name__)
        logger.info(f"Collecting hyperparameters for optimization... {hyperparameters_dict}")

        prepared_hyperparameters = hyperparameters_dict["prepared_hyperparameters"]
        hyper_to_optimize = hyperparameters_dict["hyper_to_optimize"]

        hyper_collect = {
            "hyper_list_all": prepared_hyperparameters.hyperparameters,
            "hyper_bound_list_updated": hyper_to_optimize,
            "hyper_bound_list_fixed": {},
            "dt_hyper_fixed_mod": pd.DataFrame(),
            "all_fixed": False,
        }

        # Collect fixed hyperparameters
        for channel, channel_params in prepared_hyperparameters.hyperparameters.items():
            for param_name in ["thetas", "shapes", "scales", "alphas", "gammas", "penalty"]:
                param_value = getattr(channel_params, param_name)
                if param_value is not None and f"{channel}_{param_name}" not in hyper_to_optimize:
                    hyper_collect["hyper_bound_list_fixed"][f"{channel}_{param_name}"] = param_value

        # Handle lambda_ and train_size
        if isinstance(prepared_hyperparameters.lambda_, (int, float)) and "lambda" not in hyper_to_optimize:
            hyper_collect["hyper_bound_list_fixed"]["lambda"] = prepared_hyperparameters.lambda_

        if isinstance(prepared_hyperparameters.train_size, list) and "train_size" not in hyper_to_optimize:
            hyper_collect["hyper_bound_list_fixed"]["train_size"] = prepared_hyperparameters.train_size

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
