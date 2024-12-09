# pyre-strict

import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import nevergrad as ng
from tqdm import tqdm

import logging
import time
from datetime import datetime
from robyn.modeling.convergence.convergence import Convergence
from sklearn.exceptions import ConvergenceWarning
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.model_refit_output import ModelRefitOutput
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm
from robyn.modeling.ridge.ridge_metrics_calculator import (
    RidgeMetricsCalculator,
)
from robyn.modeling.ridge.ridge_evaluate_model import RidgeModelEvaluator
from robyn.modeling.ridge.ridge_data_builder import RidgeDataBuilder


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
        self.ridge_data_builder = RidgeDataBuilder(mmm_data, featurized_mmm_data)
        self.ridge_metrics_calculator = RidgeMetricsCalculator(
            mmm_data, hyperparameters, self.ridge_data_builder
        )
        self.ridge_model_evaluator = RidgeModelEvaluator(
            self.mmm_data,
            self.featurized_mmm_data,
            self.ridge_metrics_calculator,
            self.ridge_data_builder,
        )

        self.logger = logging.getLogger(__name__)

    def build_models(
        self,
        trials_config: TrialsConfig,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        seed: List[int] = [123],
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[List[float]] = None,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        cores: Optional[int] = None,
    ) -> ModelOutputs:
        start_time = time.time()
        # Initialize hyperparameters with flattened structure
        hyper_collect = self.ridge_data_builder._hyper_collector(
            self.hyperparameters,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )
        # Convert datetime to string format matching R's format
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Set up objective weights including calibration if available
        if objective_weights is None:
            if self.calibration_input is not None:
                objective_weights = [1 / 3, 1 / 3, 1 / 3]  # NRMSE, RSSD, MAPE
            else:
                objective_weights = [0.5, 0.5]  # NRMSE, RSSD only
        # Run trials
        trials = []
        for trial in range(1, trials_config.trials + 1):
            trial_result = self.ridge_model_evaluator._run_nevergrad_optimization(
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
                seed=seed[0] + trial,  # Use the first element of the seed list
                total_trials=trials_config.trials,
            )
            trials.append(trial_result)
        # Calculate convergence
        convergence = Convergence()
        convergence_results = convergence.calculate_convergence(trials)

        # Aggregate results with explicit type casting
        all_result_hyp_param = pd.concat(
            [trial.result_hyp_param for trial in trials], ignore_index=True
        )
        all_result_hyp_param = self.ridge_data_builder.safe_astype(
            all_result_hyp_param,
            {
                "sol_id": "str",
                "trial": "int64",
                "iterNG": "int64",
                "iterPar": "int64",
                "nrmse": "float64",
                "decomp.rssd": "float64",
                "mape": "int64",
                "pos": "int64",
                "lambda": "float64",
                "lambda_hp": "float64",
                "lambda_max": "float64",
                "lambda_min_ratio": "float64",
                "rsq_train": "float64",
                "rsq_val": "float64",
                "rsq_test": "float64",
                "nrmse_train": "float64",
                "nrmse_val": "float64",
                "nrmse_test": "float64",
                "ElapsedAccum": "float64",
                "Elapsed": "float64",
            },
        )

        all_x_decomp_agg = pd.concat(
            [trial.x_decomp_agg for trial in trials], ignore_index=True
        )
        all_x_decomp_agg = self.ridge_data_builder.safe_astype(
            all_x_decomp_agg,
            {
                "rn": "str",
                "coef": "float64",
                "xDecompAgg": "float64",
                "xDecompPerc": "float64",
                "xDecompMeanNon0": "float64",
                "xDecompMeanNon0Perc": "float64",
                "xDecompAggRF": "float64",
                "xDecompPercRF": "float64",
                "xDecompMeanNon0RF": "float64",
                "xDecompMeanNon0PercRF": "float64",
                "sol_id": "str",
                "pos": "bool",
                "mape": "int64",
            },
        )

        all_decomp_spend_dist = pd.concat(
            [
                trial.decomp_spend_dist
                for trial in trials
                if trial.decomp_spend_dist is not None
            ],
            ignore_index=True,
        )
        all_decomp_spend_dist = self.ridge_data_builder.safe_astype(
            all_decomp_spend_dist,
            {
                "rn": "str",
                "coef": "float64",
                "total_spend": "float64",
                "mean_spend": "float64",
                "effect_share": "float64",
                "spend_share": "float64",
                "xDecompAgg": "float64",
                "xDecompPerc": "float64",
                "xDecompMeanNon0": "float64",
                "xDecompMeanNon0Perc": "float64",
                "sol_id": "str",
                "pos": "bool",
                "mape": "int64",
                "nrmse": "float64",
                "decomp.rssd": "float64",
                "trial": "int64",
                "iterNG": "int64",
                "iterPar": "int64",
            },
        )
        # Convert hyper_bound_ng from dict to DataFrame
        hyper_bound_ng_df = pd.DataFrame()
        for param_name, bounds in hyper_collect["hyper_bound_list_updated"].items():
            hyper_bound_ng_df.loc[0, param_name] = bounds[0]
            hyper_bound_ng_df[param_name] = hyper_bound_ng_df[param_name].astype(
                "float64"
            )
        if "lambda" in hyper_bound_ng_df.columns:
            hyper_bound_ng_df["lambda"] = hyper_bound_ng_df["lambda"].astype("int64")
        # Create ModelOutputs
        model_outputs = ModelOutputs(
            trials=trials,
            train_timestamp=current_time,
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
            hyper_bound_ng=hyper_bound_ng_df,
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
        nrmse_norm = (nrmse_values - np.min(nrmse_values)) / (
            np.max(nrmse_values) - np.min(nrmse_values)
        )
        decomp_rssd_norm = (decomp_rssd_values - np.min(decomp_rssd_values)) / (
            np.max(decomp_rssd_values) - np.min(decomp_rssd_values)
        )

        # Calculate the combined score (assuming equal weights)
        combined_score = nrmse_norm + decomp_rssd_norm

        # Find the index of the best model (lowest combined score)
        best_index = np.argmin(combined_score)

        # Return the sol_id of the best model (changed from solID)
        return output_models[best_index].result_hyp_param["sol_id"].values[0]

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
            trial_result = self.ridge_model_evaluator._run_nevergrad_optimization(
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
        """Evaluate model with parameter set matching R's implementation exactly"""
        X, y = self._prepare_data(params)
        sol_id = f"{trial}_{iter_ng + 1}_1"
        # After preparing data
        self.logger.debug(f"Data shapes - X: {X.shape}, y: {y.shape}")
        self.logger.debug(f"Sample of X values: {X.head()}")
        self.logger.debug(f"Sample of y values: {y.head()}")

        # Split data using R's approach
        train_size = params.get("train_size", 1.0) if ts_validation else 1.0
        train_idx = int(len(X) * train_size)

        metrics = {}
        if ts_validation:
            val_test_size = (len(X) - train_idx) // 2
            X_train = X.iloc[:train_idx]
            y_train = y.iloc[:train_idx]
            X_val = X.iloc[train_idx : train_idx + val_test_size]
            y_val = y.iloc[train_idx : train_idx + val_test_size]
            X_test = X.iloc[train_idx + val_test_size :]
            y_test = y.iloc[train_idx + val_test_size :]
        else:
            X_train, y_train = X, y
            X_val = X_test = y_val = y_test = None

        x_norm = X_train.to_numpy()
        y_norm = y_train.to_numpy()

        # Calculate lambda using R-matching helper function
        lambda_hp = params.get("lambda", 1.0)
        lambda_, lambda_max = self.ridge_metrics_calculator._calculate_lambda(
            x_norm, y_norm, lambda_hp
        )
        # After calculating lambda
        self.logger.debug(f"Lambda calculation debug:")
        self.logger.debug(f"lambda_hp: {lambda_hp}")
        self.logger.debug(f"lambda_: {lambda_}")
        self.logger.debug(f"lambda_max: {lambda_max}")

        # Scale inputs for model
        model = Ridge(alpha=lambda_ / len(x_norm), fit_intercept=True)
        model.fit(x_norm, y_norm)

        # Calculate metrics using R-style calculations
        y_train_pred = model.predict(x_norm)
        metrics["rsq_train"] = self.ridge_metrics_calculator.calculate_r2_score(
            y_norm, y_train_pred, x_norm.shape[1]
        )
        metrics["nrmse_train"] = self.ridge_metrics_calculator.calculate_nrmse(
            y_norm, y_train_pred
        )

        # Validation and test metrics
        if ts_validation and X_val is not None and X_test is not None:
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            metrics["rsq_val"] = self.ridge_metrics_calculator.calculate_r2_score(
                y_val, y_val_pred, X_val.shape[1]
            )
            metrics["nrmse_val"] = self.ridge_metrics_calculator.calculate_nrmse(
                y_val, y_val_pred
            )

            metrics["rsq_test"] = self.ridge_metrics_calculator.calculate_r2_score(
                y_test, y_test_pred, X_test.shape[1]
            )
            metrics["nrmse_test"] = self.ridge_metrics_calculator.calculate_nrmse(
                y_test, y_test_pred
            )

            metrics["nrmse"] = metrics["nrmse_val"]
        else:
            metrics["rsq_val"] = metrics["rsq_test"] = 0.0
            metrics["nrmse_val"] = metrics["nrmse_test"] = 0.0
            metrics["nrmse"] = metrics["nrmse_train"]

        # Calculate RSSD
        paid_media_cols = [
            col
            for col in X.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]
        decomp_rssd = self.ridge_metrics_calculator._calculate_rssd(
            model, X_train, paid_media_cols, rssd_zero_penalty
        )

        elapsed_time = time.time() - start_time

        # Format hyperparameter names to match R's format
        params_formatted = self._format_hyperparameter_names(params)

        # Update metrics dictionary
        metrics.update(
            {
                "decomp_rssd": float(decomp_rssd),
                "lambda": float(lambda_),
                "lambda_hp": float(lambda_hp),
                "lambda_max": float(lambda_max),
                "lambda_min_ratio": float(0.0001),
                "mape": int(0),  # Cast to int as in R
                "sol_id": str(sol_id),
                "trial": int(trial),
                "iterNG": int(iter_ng + 1),
                "iterPar": int(1),
                "Elapsed": float(elapsed_time),
                "elapsed": float(elapsed_time),
                "elapsed_accum": float(elapsed_time),
            }
        )

        # Calculate decompositions
        x_decomp_agg = self.ridge_metrics_calculator._calculate_x_decomp_agg(
            model, X_train, y_train, {**params_formatted, **metrics}
        )
        decomp_spend_dist = self.ridge_metrics_calculator._calculate_decomp_spend_dist(
            model, X_train, y_train, {**metrics, "params": params_formatted}
        )

        # Calculate loss
        loss = (
            objective_weights[0] * metrics["nrmse"]
            + objective_weights[1] * metrics["decomp_rssd"]
            + (
                objective_weights[2] * metrics["mape"]
                if len(objective_weights) > 2
                else 0
            )
        )
        self.logger.debug(
            f"Model coefficients range: {model.coef_.min()} to {model.coef_.max()}"
        )
        self.logger.debug(f"Sample predictions: {y_train_pred[:5]}")
        self.logger.debug(f"Sample actual values: {y_norm[:5]}")
        return {
            "loss": loss,
            "params": params_formatted,
            **metrics,
            "decomp_spend_dist": decomp_spend_dist,
            "x_decomp_agg": x_decomp_agg,
            "elapsed": elapsed_time,
            "elapsed_accum": elapsed_time,
            "iter_ng": iter_ng + 1,
            "iter_par": 1,
        }
