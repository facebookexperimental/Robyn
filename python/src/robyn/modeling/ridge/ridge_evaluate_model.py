import pandas as pd
import numpy as np
import nevergrad as ng
import time
import warnings
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from typing import Dict, Any, Tuple, Optional, List
from robyn.modeling.entities.modeloutputs import Trial
from robyn.modeling.entities.enums import NevergradAlgorithm
from robyn.modeling.ridge.ridge_metrics_calculator import RidgeMetricsCalculator
import logging


class RidgeModelEvaluator:

    def __init__(
        self,
        mmm_data,
        featurized_mmm_data,
        ridge_metrics_calculator,
        ridge_data_builder,
    ):
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.ridge_metrics_calculator = ridge_metrics_calculator
        self.ridge_data_builder = ridge_data_builder
        self.logger = logging.getLogger(__name__)

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
        param_bounds = [
            hyper_collect["hyper_bound_list_updated"][name] for name in param_names
        ]

        instrum_dict = {
            name: ng.p.Scalar(lower=bound[0], upper=bound[1])
            for name, bound in zip(param_names, param_bounds)
        }

        instrum = ng.p.Instrumentation(**instrum_dict)
        optimizer = ng.optimizers.registry[nevergrad_algo.value](
            instrum, budget=iterations, num_workers=cores
        )

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

                # Important: Convert metrics to correct types
                sol_id = f"{trial}_{iter_ng + 1}_1"
                result["params"].update(
                    {
                        "sol_id": sol_id,
                        "ElapsedAccum": result["elapsed_accum"],
                        "trial": int(trial),
                        "rsq_train": float(result["rsq_train"]),
                        "rsq_val": float(result["rsq_val"]),
                        "rsq_test": float(result["rsq_test"]),
                        "nrmse": float(result["nrmse"]),
                        "nrmse_train": float(result["nrmse_train"]),
                        "nrmse_val": float(result["nrmse_val"]),
                        "nrmse_test": float(result["nrmse_test"]),
                        "decomp.rssd": float(result["decomp_rssd"]),
                        "mape": float(result["mape"]),
                        "lambda": float(
                            result["lambda"]
                        ),  # Critical: Using lambda not lambda_
                        "lambda_hp": float(result["lambda_hp"]),
                        "lambda_max": float(result["lambda_max"]),
                        "lambda_min_ratio": float(result["lambda_min_ratio"]),
                        "iterNG": int(iter_ng + 1),
                        "iterPar": 1,
                    }
                )

                all_results.append(result)
                pbar.update(1)

        end_time = time.time()
        self.logger.info(f" Finished in {(end_time - start_time) / 60:.2f} mins")

        # Aggregate results with explicit dtypes
        result_hyp_param = pd.DataFrame([r["params"] for r in all_results]).astype(
            {
                "sol_id": "str",
                "trial": "int64",
                "iterNG": "int64",
                "iterPar": "int64",
                "nrmse": "float64",
                "decomp.rssd": "float64",
                "mape": "float64",
                "lambda": "float64",
                "lambda_hp": "float64",
                "lambda_max": "float64",
                "lambda_min_ratio": "float64",
                "rsq_train": "float64",
                "rsq_val": "float64",
                "rsq_test": "float64",
            }
        )

        decomp_spend_dist = pd.concat(
            [r["decomp_spend_dist"] for r in all_results], ignore_index=True
        )
        x_decomp_agg = pd.concat(
            [r["x_decomp_agg"] for r in all_results], ignore_index=True
        )

        # Ensure correct dtypes in decomp_spend_dist and x_decomp_agg
        decomp_spend_dist = decomp_spend_dist.astype(
            {
                "rn": "str",
                "coef": "float64",
                "total_spend": "float64",
                "mean_spend": "float64",
                "effect_share": "float64",
                "spend_share": "float64",
                "sol_id": "str",
            }
        )

        x_decomp_agg = x_decomp_agg.astype(
            {
                "rn": "str",
                "coef": "float64",
                "xDecompAgg": "float64",
                "xDecompPerc": "float64",
                "sol_id": "str",
            }
        )

        # Find best result based on loss
        best_result = min(all_results, key=lambda x: x["loss"])
        # Convert values to Series before passing to Trial
        return Trial(
            result_hyp_param=result_hyp_param,
            lift_calibration=best_result.get("lift_calibration", pd.DataFrame()),
            decomp_spend_dist=decomp_spend_dist,
            x_decomp_agg=x_decomp_agg,
            nrmse=pd.Series([float(best_result["nrmse"])]),
            decomp_rssd=pd.Series([float(best_result["decomp_rssd"])]),
            mape=pd.Series([int(best_result["mape"])]),  # Cast to int
            rsq_train=pd.Series([float(best_result["rsq_train"])]),
            rsq_val=pd.Series([float(best_result["rsq_val"])]),
            rsq_test=pd.Series([float(best_result["rsq_test"])]),
            lambda_=pd.Series([float(best_result["lambda"])]),
            lambda_hp=pd.Series([float(best_result["lambda_hp"])]),
            lambda_max=pd.Series([float(best_result["lambda_max"])]),
            lambda_min_ratio=pd.Series([float(best_result["lambda_min_ratio"])]),
            pos=pd.Series([int(best_result.get("pos", 0))]),  # Cast to int
            elapsed=pd.Series([float(best_result["elapsed"])]),
            elapsed_accum=pd.Series([float(best_result["elapsed_accum"])]),
            trial=pd.Series([int(trial)]),
            iter_ng=pd.Series([int(best_result["iter_ng"])]),
            iter_par=pd.Series([int(best_result["iter_par"])]),
            train_size=pd.Series([float(best_result["params"].get("train_size", 1.0))]),
            sol_id=str(best_result["params"]["sol_id"]),
        )

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
        X, y = self.ridge_data_builder._prepare_data(params)
        sol_id = f"{trial}_{iter_ng + 1}_1"
        # After preparing data
        self.logger.debug(f"Data shapes - X: {X.shape}, y: {y.shape}")
        self.logger.debug(f"Sample of X values: {X.head()}")
        self.logger.debug(f"Sample of y values: {y.head()}")

        # Debug is True by default now
        debug = True

        if debug and (iter_ng == 0 or iter_ng % 25 == 0):
            self.logger.debug(
                f"\nEvaluation Debug (trial {trial}, iteration {iter_ng}):"
            )
            self.logger.debug(f"X shape: {X.shape}")
            self.logger.debug(f"y shape: {y.shape}")
            self.logger.debug("Parameters:", params)

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
            x_norm, y_norm, lambda_hp, debug=debug, iteration=iter_ng
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
            y_norm, y_train_pred, debug=debug, iteration=iter_ng
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
            model,
            X_train,
            paid_media_cols,
            rssd_zero_penalty,
            debug=debug,
            iteration=iter_ng,
        )

        elapsed_time = time.time() - start_time

        # Format hyperparameter names to match R's format
        params_formatted = self.ridge_data_builder._format_hyperparameter_names(params)

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
