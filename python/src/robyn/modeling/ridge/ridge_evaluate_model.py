import pandas as pd
import numpy as np
import nevergrad as ng
from nevergrad.optimization.base import Optimizer
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
from robyn.reporting.utils.modeling_debug import debug_model_metrics
from robyn.modeling.ridge.models.ridge_utils import create_ridge_model_rpy2
import json
from datetime import datetime


class RidgeModelEvaluator:

    def __init__(
        self,
        mmm_data,
        featurized_mmm_data,
        ridge_metrics_calculator,
        ridge_data_builder,
        calibration_input=None,
    ):
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.ridge_metrics_calculator = ridge_metrics_calculator
        self.ridge_data_builder = ridge_data_builder
        self.calibration_input = calibration_input
        self.logger = logging.getLogger(__name__)

    def _run_nevergrad_optimization(
        self,
        optimizer: Optimizer,
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
        """Run Nevergrad optimization for ridge regression."""
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        np.random.seed(seed)
        param_names = list(hyper_collect["hyper_bound_list_updated"].keys())

        self.logger.debug(f"Starting optimization with {len(param_names)} parameters")
        self.logger.debug(f"Parameter names: {param_names}")

        start_time = time.time()
        all_results = []

        with tqdm(
            total=iterations, desc=f"Running trial {trial} of {total_trials}"
        ) as pbar:
            for iter_ng in range(iterations):
                candidate = optimizer.ask()
                self.logger.debug(
                    f"Iteration {iter_ng + 1}: Got candidate: {candidate}"
                )
                self.logger.debug(f"Candidate value type: {type(candidate.value)}")
                self.logger.debug(f"Candidate value: {candidate.value}")

                # Since we're using Array instrumentation, candidate.value should be a numpy array
                raw_values = candidate.value
                self.logger.debug(f"Raw values: {raw_values}")

                # Transform values using qunif (like R)
                transformed_params = {}
                for i, name in enumerate(param_names):
                    bounds = hyper_collect["hyper_bound_list_updated"][name]
                    raw_value = raw_values[i]
                    transformed_value = bounds[0] + raw_value * (bounds[1] - bounds[0])
                    transformed_params[name] = transformed_value
                    self.logger.debug(
                        f"Parameter {name}: raw={raw_value}, transformed={transformed_value}"
                    )

                # Log both raw and transformed values
                self.logger.debug(
                    json.dumps(
                        {
                            "step": f"step6_hyperparameter_sampling_iteration_{iter_ng + 1}",
                            "data": {
                                "iteration_info": {
                                    "current_iteration": iter_ng + 1,
                                    "total_iterations": iterations,
                                    "cores": 1,
                                },
                                "sampling": {
                                    "hyper_fixed": False,
                                    "num_samples": 1,
                                    "updated_params": {
                                        "names": param_names,
                                        "bounds": {
                                            name: hyper_collect[
                                                "hyper_bound_list_updated"
                                            ][name]
                                            for name in param_names
                                        },
                                    },
                                },
                                "results": {
                                    "sampled_values": [
                                        [round(v, 4) for v in raw_values]
                                    ],
                                    "final_hyperparams": {
                                        name: [round(transformed_params[name], 4)]
                                        for name in param_names
                                    },
                                },
                            },
                        },
                        indent=2,
                    )
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self._evaluate_model(
                        transformed_params,
                        ts_validation,
                        add_penalty_factor,
                        rssd_zero_penalty,
                        objective_weights,
                        start_time=start_time,
                        iter_ng=iter_ng,
                        total_iterations=iterations,
                        cores=cores,
                        trial=trial,
                    )

                self.logger.debug(
                    f"Evaluation result - NRMSE: {result['nrmse']:.6f}, RSSD: {result.get('decomp_rssd', 0):.6f}"
                )

                if self.calibration_input is not None:
                    optimizer.tell(
                        candidate,
                        (result["nrmse"], result["decomp_rssd"], result["mape"]),
                    )
                    self.logger.debug(f"Told optimizer with multi-objective results")
                else:
                    optimizer.tell(candidate, (result["nrmse"], result["decomp_rssd"]))

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

                if iter_ng == 0 or iter_ng % 10 == 0:
                    self.logger.debug(
                        f"Iteration {iter_ng+1} results - NRMSE: {result['nrmse']:.6f}, RSSD: {result.get('decomp_rssd', 0):.6f}, Loss: {result['loss']:.6f}"
                    )

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
        recommendation = best_result["params"]
        self.logger.debug(
            f"=== Optimization Complete (Trial {trial}/{total_trials}) ==="
        )
        self.logger.debug(f"Best parameters: {recommendation}")
        if (
            hasattr(optimizer, "current_bests")
            and "pessimistic" in optimizer.current_bests
        ):
            self.logger.debug(
                f"Best observed loss: {optimizer.current_bests['pessimistic'].mean}"
            )
        self.logger.debug(
            f"Final performance: NRMSE={best_result['nrmse']:.6f}, RSSD={best_result.get('decomp_rssd', 0):.6f}"
        )
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
        total_iterations: int,
        cores: int,
        trial: int,
    ) -> Dict[str, Any]:
        """Evaluate model with parameter set"""
        # Get transformed data
        transformed_data = self.ridge_data_builder.run_transformations(
            params,
            current_iteration=iter_ng + 1,
            total_iterations=total_iterations,
            cores=cores,
        )
        dt_modSaturated = transformed_data["dt_modSaturated"]

        # Split dep_var and features like R does
        y = dt_modSaturated["dep_var"]
        X = dt_modSaturated.drop(columns=["dep_var"])

        # Continue with existing evaluation logic...
        sol_id = f"{trial}_{iter_ng + 1}_1"

        # Split data using R's approach
        train_size = params.get("train_size", 1.0) if ts_validation else 1.0
        train_idx = int(np.floor(np.quantile(range(len(X)), train_size)))
        val_test_size = int(np.floor((len(X) * (1 - train_size)) / 2))

        metrics = {}
        if ts_validation:
            X_train = X.iloc[:train_idx]
            y_train = y.iloc[:train_idx]
            X_val = X.iloc[train_idx : train_idx + val_test_size]
            y_val = y.iloc[train_idx : train_idx + val_test_size]
            X_test = X.iloc[train_idx + val_test_size :]
            y_test = y.iloc[train_idx + val_test_size :]
        else:
            X_train, y_train = X, y
            X_val = X_test = y_val = y_test = None

        # After splitting data (around line 352)
        # Log step8 data splitting information
        self.logger.debug(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": f"step8_data_split_iteration_{iter_ng + 1}",
                    "data": {
                        "window_info": {
                            "total_rows": len(X),
                            "total_features": X.shape[1],
                            "feature_names": X.columns.tolist(),
                        },
                        "split_params": {
                            "train_size": train_size,
                            "val_size": (1 - train_size) / 2 if ts_validation else None,
                            "test_size": (
                                (1 - train_size) / 2 if ts_validation else None
                            ),
                        },
                        "split_indices": {
                            "train_end": train_idx,
                            "val_end": (
                                train_idx + val_test_size if ts_validation else None
                            ),
                            "test_end": len(X) if ts_validation else None,
                        },
                        "split_shapes": {
                            "x_train": X_train.shape,
                            "y_train": len(y_train),
                            "x_val": X_val.shape if X_val is not None else None,
                            "y_val": len(y_val) if y_val is not None else None,
                            "x_test": X_test.shape if X_test is not None else None,
                            "y_test": len(y_test) if y_test is not None else None,
                        },
                        "data_ranges": {
                            "y_train": {
                                "min": float(y_train.min()),
                                "max": float(y_train.max()),
                                "mean": float(y_train.mean()),
                            },
                            "y_val": (
                                {
                                    "min": float(y_val.min()),
                                    "max": float(y_val.max()),
                                    "mean": float(y_val.mean()),
                                }
                                if y_val is not None
                                else None
                            ),
                            "y_test": (
                                {
                                    "min": float(y_test.min()),
                                    "max": float(y_test.max()),
                                    "mean": float(y_test.mean()),
                                }
                                if y_test is not None
                                else None
                            ),
                        },
                    },
                },
                indent=2,
            )
        )

        x_norm = X_train.to_numpy()
        y_norm = y_train.to_numpy()

        # Initialize lambda sequence if needed
        self.ridge_metrics_calculator.initialize_lambda_sequence(X, y)

        # Get lambda values
        lambda_hp = params.get("lambda", 1.0)
        lambda_ = self.ridge_metrics_calculator.get_lambda_from_hp(lambda_hp)
        lambda_max = self.ridge_metrics_calculator.lambda_max
        lambda_min_ratio = self.ridge_metrics_calculator.lambda_min_ratio

        # Scale inputs for model
        N = len(x_norm)

        # Convert lambda to sklearn alpha using Approach 1: alpha = lambda * N / 2
        # model = create_ridge_model_sklearn(
        #     lambda_value=lambda_, n_samples=N, fit_intercept=True
        # model.fit(x_norm, y_norm)

        # Create and fit the model
        model = create_ridge_model_rpy2(
            lambda_value=lambda_,
            n_samples=N,
            fit_intercept=True,
            standardize=True,
            lower_limits=params.get("lower_limits", None),
            upper_limits=params.get("upper_limits", None),
            penalty_factor=params.get("penalty_factor", None),
        )
        model.fit(x_norm, y_norm)

        # Calculate metrics using R-style calculations
        y_train_pred = model.predict(x_norm)
        metrics["rsq_train"] = self.ridge_metrics_calculator.calculate_r2_score(
            y_norm,
            y_train_pred,
            p=x_norm.shape[1],
            df_int=1 if model.fit_intercept else 0,
        )
        metrics["nrmse_train"] = self.ridge_metrics_calculator.calculate_nrmse(
            y_norm, y_train_pred
        )

        # Validation and test metrics
        if ts_validation and X_val is not None and X_test is not None:
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            n_train = len(y_train)  # Get training set size

            metrics["rsq_val"] = self.ridge_metrics_calculator.calculate_r2_score(
                y_val,
                y_val_pred,
                p=X_val.shape[1],
                df_int=1 if model.fit_intercept else 0,
                n_train=n_train,  # Pass training set size
            )

            metrics["rsq_test"] = self.ridge_metrics_calculator.calculate_r2_score(
                y_test,
                y_test_pred,
                p=X_test.shape[1],
                df_int=1 if model.fit_intercept else 0,
                n_train=n_train,  # Pass training set size
            )

            metrics["nrmse_val"] = self.ridge_metrics_calculator.calculate_nrmse(
                y_val, y_val_pred
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
            debug=True,
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
                "lambda_max": float(lambda_max),  # Now lambda_max is defined
                "lambda_min_ratio": float(
                    lambda_min_ratio
                ),  # Use the ratio from calculator
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
        # Debug model metrics
        lambda_info = {
            "lambda_": lambda_,
            "lambda_max": lambda_max,
            "lambda_min_ratio": lambda_min_ratio,
        }
        # debug_model_metrics(
        #     self.ridge_metrics_calculator,
        #     X_train,
        #     y_train,
        #     lambda_info,
        #     metrics,
        #     iter_ng,
        # )
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

        # Log raw parameters
        self.logger.debug(
            f"\n=== Model Evaluation (Trial {trial}, Iteration {iter_ng}) ==="
        )
        self.logger.debug(f"Raw parameters: {params}")

        # After formatting parameters
        self.logger.debug(f"Formatted parameters: {params_formatted}")

        # After data preparation
        self.logger.debug(
            f"Data splits - train: {X_train.shape}, test: {X_test.shape if X_test is not None else 'None'}"
        )

        # After fitting model
        self.logger.debug(f"Model fitted - elapsed: {elapsed_time:.4f}s")
        self.logger.debug(
            f"Lambda: {lambda_:.4f}, Lambda as proportion of max: {lambda_/lambda_max:.4f}"
        )
        self.logger.debug(
            f"Coefficient stats - min: {model.coef_.min():.6f}, max: {model.coef_.max():.6f}, mean: {model.coef_.mean():.6f}"
        )

        # After metric calculation
        self.logger.debug(
            f"Metrics - NRMSE: {metrics['nrmse']:.6f}, RSQ: {metrics.get('rsq_train', 0):.6f}, RSSD: {metrics.get('decomp_rssd', 0):.6f}"
        )

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
