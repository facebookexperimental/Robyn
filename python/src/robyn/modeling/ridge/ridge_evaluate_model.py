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
from robyn.modeling.ridge.models.ridge_utils import create_ridge_model_rpy2
import json
from datetime import datetime
import random


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
        random.seed(seed)
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
                        intercept_sign=intercept_sign,
                        intercept=intercept,
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
        intercept_sign: str,
        intercept: bool,
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

        # Get sign control parameters
        x_sign, lower_limits, upper_limits, check_factor = self._setup_sign_control(X)
        params["lower_limits"] = lower_limits
        params["upper_limits"] = upper_limits

        # Convert numpy bool_ to Python bool for JSON serialization
        factor_dict = {col: bool(is_factor) for col, is_factor in check_factor.items()}

        # Helper function to format limit values
        def format_limit_value(val):
            if isinstance(val, str):
                return val  # Already a string ("Inf" or "-Inf")
            return str(val) if isinstance(val, float) and np.isinf(val) else float(val)

        # Log step9 sign control
        self.logger.debug(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": f"step9_sign_control_iteration_{iter_ng + 1}",
                    "data": {
                        "variable_types": {
                            "prophet": ["trend", "season", "holiday"],
                            "context": list(self.mmm_data.mmmdata_spec.context_vars),
                            "paid_media": list(
                                self.mmm_data.mmmdata_spec.paid_media_spends
                            ),
                            "organic": list(self.mmm_data.mmmdata_spec.organic_vars),
                        },
                        "signs": x_sign,
                        "factor_variables": {
                            "is_factor": factor_dict,
                            "factor_names": [
                                col
                                for col, is_factor in factor_dict.items()
                                if is_factor
                            ],
                        },
                        "trend_info": (
                            {
                                "location": (
                                    int(X.columns.get_loc("trend"))
                                    if "trend" in X.columns
                                    else None
                                ),
                                "negative_trend": (
                                    bool(X["trend"].sum() < 0)
                                    if "trend" in X.columns
                                    else None
                                ),
                            }
                            if "trend" in X.columns
                            else None
                        ),
                        "constraints": {
                            "lower_limits": [
                                {"value": format_limit_value(val), "variable": str(col)}
                                for col, val in zip(X.columns, lower_limits)
                            ],
                            "upper_limits": [
                                {"value": format_limit_value(val), "variable": str(col)}
                                for col, val in zip(X.columns, upper_limits)
                            ],
                        },
                    },
                },
                indent=2,
            )
        )

        # Initialize lambda sequence if needed
        self.ridge_metrics_calculator.initialize_lambda_sequence(X, y)

        # Get lambda values
        lambda_hp = params.get("lambda", 1.0)
        lambda_ = self.ridge_metrics_calculator.get_lambda_from_hp(lambda_hp)
        lambda_max = self.ridge_metrics_calculator.lambda_max
        lambda_min_ratio = self.ridge_metrics_calculator.lambda_min_ratio

        # Handle penalty factor exactly like R
        if add_penalty_factor:
            penalty_factor = [v for k, v in params.items() if "_penalty" in k]
        else:
            penalty_factor = [1] * x_norm.shape[1]  # rep(1, ncol(x_train))

        # Log ridge regression setup
        self.logger.debug(
            json.dumps(
                {
                    "step": f"step10_ridge_regression_setup_iteration_{iter_ng + 1}",
                    "data": {
                        "lambda_hp": lambda_hp,
                        "lambda_scaled": lambda_,  # Using lambda_ instead of lambda_scaled
                        "penalty_factor": penalty_factor,
                    },
                },
                indent=2,
            )
        )
        # Scale inputs for model
        N = len(x_norm)

        # Create and fit the model
        model = create_ridge_model_rpy2(
            lambda_value=lambda_,
            n_samples=N,
            fit_intercept=True,
            standardize=True,
            intercept_sign=intercept_sign,
            intercept=intercept,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            penalty_factor=penalty_factor,
        )
        # Log data shapes and stats before fitting
        self.logger.debug(
            json.dumps(
                {
                    "step": "step11a_model_refit_data_check",
                    "data": {
                        "shapes": {
                            "x_norm": x_norm.shape,
                            "y_norm": y_norm.shape,
                            "X_val": X_val.shape if X_val is not None else None,
                            "X_test": X_test.shape if X_test is not None else None,
                        },
                        "stats": {
                            "y_norm": {
                                "mean": float(np.mean(y_norm)),
                                "std": float(np.std(y_norm)),
                                "min": float(np.min(y_norm)),
                                "max": float(np.max(y_norm)),
                            },
                            "x_norm_mean": float(np.mean(np.abs(x_norm))),
                        },
                    },
                },
                indent=2,
            )
        )

        model.fit(x_norm, y_norm)

        # Calculate metrics using R-style calculations
        y_train_pred = model.predict(x_norm)

        # Log prediction stats
        self.logger.debug(
            json.dumps(
                {
                    "step": "step11b_model_refit_predictions",
                    "data": {
                        "predictions": {
                            "y_train_pred": {
                                "mean": float(np.mean(y_train_pred)),
                                "std": float(np.std(y_train_pred)),
                                "min": float(np.min(y_train_pred)),
                                "max": float(np.max(y_train_pred)),
                            },
                            "y_norm": {
                                "mean": float(np.mean(y_norm)),
                                "std": float(np.std(y_norm)),
                                "min": float(np.min(y_norm)),
                                "max": float(np.max(y_norm)),
                            },
                        },
                        "r2_components": {
                            "sse": float(np.sum((y_train_pred - y_norm) ** 2)),
                            "sst": float(np.sum((y_norm - np.mean(y_norm)) ** 2)),
                            "n": len(y_norm),
                            "p": x_norm.shape[1],
                            "df_int": model.df_int,
                        },
                    },
                },
                indent=2,
            )
        )
        metrics["rsq_train"] = self.ridge_metrics_calculator.calculate_r2_score(
            y_norm,
            y_train_pred,
            p=x_norm.shape[1],
            df_int=model.df_int,
        )
        metrics["nrmse_train"] = self.ridge_metrics_calculator.calculate_nrmse(
            y_norm, y_train_pred
        )

        # Validation and test metrics
        if ts_validation and X_val is not None and X_test is not None:
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            # Concatenate predictions like R does
            y_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])

            n_train = len(y_train)

            metrics["rsq_val"] = self.ridge_metrics_calculator.calculate_r2_score(
                y_val,
                y_val_pred,
                p=X_val.shape[1],
                df_int=model.df_int,
                n_train=n_train,
            )
            metrics["rsq_test"] = self.ridge_metrics_calculator.calculate_r2_score(
                y_test,
                y_test_pred,
                p=X_test.shape[1],
                df_int=model.df_int,
                n_train=n_train,
            )
            metrics["nrmse_val"] = self.ridge_metrics_calculator.calculate_nrmse(
                y_val, y_val_pred
            )
            metrics["nrmse_test"] = self.ridge_metrics_calculator.calculate_nrmse(
                y_test, y_test_pred
            )
            metrics["nrmse"] = metrics["nrmse_val"]
        else:
            y_pred = y_train_pred  # If no validation, just use training predictions
            metrics["rsq_val"] = metrics["rsq_test"] = 0.0
            metrics["nrmse_val"] = metrics["nrmse_test"] = 0.0
            metrics["nrmse"] = metrics["nrmse_train"]

        # Log ridge regression results
        self.logger.debug(
            json.dumps(
                {
                    "step": f"step11c_model_refit_mod_out_iteration_{iter_ng + 1}",
                    "data": {
                        "rsq_train": metrics["rsq_train"],
                        "rsq_val": metrics["rsq_val"],
                        "rsq_test": metrics["rsq_test"],
                        "nrmse_train": metrics["nrmse_train"],
                        "nrmse_val": metrics["nrmse_val"],
                        "nrmse_test": metrics["nrmse_test"],
                        "coefs": list(model.get_full_coefficients()),
                        "df_int": model.df_int,
                    },
                },
                indent=2,
            )
        )

        paid_media_cols = [
            col
            for col in X.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]

        # Log model decomp inputs
        self.logger.debug(
            json.dumps(
                {
                    "step": "step11d_model_decomp_inputs",
                    "data": {
                        "coefs": model.coef_.tolist(),
                        "y_pred_length": len(y_train_pred),
                        "dt_modSaturated_dims": dt_modSaturated.shape,
                        "dt_saturatedImmediate_dims": transformed_data[
                            "dt_saturatedImmediate"
                        ].shape,
                        "dt_saturatedCarryover_dims": transformed_data[
                            "dt_saturatedCarryover"
                        ].shape,
                        "dt_modRollWind_dims": self.featurized_mmm_data.dt_modRollWind.shape,
                        "refreshAddedStart": str(
                            self.mmm_data.mmmdata_spec.refresh_added_start
                        ),
                        "dt_modSaturated_cols": dt_modSaturated.columns.tolist(),
                        "dt_saturatedImmediate_cols": transformed_data[
                            "dt_saturatedImmediate"
                        ].columns.tolist(),
                        "dt_saturatedCarryover_cols": transformed_data[
                            "dt_saturatedCarryover"
                        ].columns.tolist(),
                        "dt_modRollWind_cols": self.featurized_mmm_data.dt_modRollWind.columns.tolist(),
                    },
                },
                indent=2,
            )
        )
        # Now use the concatenated predictions for decomposition
        decomp_results = self.ridge_metrics_calculator.model_decomp(
            model=model,
            y_pred=y_pred,  # Use concatenated predictions here
            dt_modSaturated=dt_modSaturated,
            dt_saturatedImmediate=transformed_data["dt_saturatedImmediate"],
            dt_saturatedCarryover=transformed_data["dt_saturatedCarryover"],
            dt_modRollWind=self.featurized_mmm_data.dt_modRollWind,
            refreshAddedStart=self.mmm_data.mmmdata_spec.refresh_added_start,
        )

        # Get the components we need
        x_decomp_agg = decomp_results["xDecompAgg"]
        decomp_spend_dist = x_decomp_agg[
            x_decomp_agg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends)
        ]

        # 1. Filter and select media decomposition data
        decomp_spend_dist = x_decomp_agg[
            x_decomp_agg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends)
        ][
            [
                "rn",
                "coef",  # add coef in from x_decomp_agg
                "xDecompAgg",
                "xDecompPerc",
                "xDecompMeanNon0Perc",
                "xDecompMeanNon0",
                "xDecompPercRF",
                "xDecompMeanNon0PercRF",
                "xDecompMeanNon0RF",
            ]
        ]

        # 2. Get spend metrics and join
        spend_metrics = pd.DataFrame(
            {
                "rn": self.ridge_data_builder.spend_metrics["rn"],
                "spend_share": self.ridge_data_builder.spend_metrics["spend_share"],
                "spend_share_refresh": self.ridge_data_builder.spend_metrics[
                    "spend_share_refresh"
                ],
                "mean_spend": self.ridge_data_builder.spend_metrics["mean_spend"],
                "total_spend": self.ridge_data_builder.spend_metrics["total_spend"],
            }
        )

        # 3. Merge spend metrics with decomposition results
        decomp_spend_dist = decomp_spend_dist.merge(spend_metrics, on="rn", how="left")

        # 4. Calculate effect shares
        total_decomp = decomp_spend_dist["xDecompPerc"].sum()
        total_decomp_rf = decomp_spend_dist["xDecompPercRF"].sum()
        decomp_spend_dist["effect_share"] = (
            decomp_spend_dist["xDecompPerc"] / total_decomp
        )
        decomp_spend_dist["effect_share_refresh"] = (
            decomp_spend_dist["xDecompPercRF"] / total_decomp_rf
        )

        # 5. Calculate RSSD based on refresh status
        if not self.mmm_data.mmmdata_spec.refresh_counter:
            # Regular RSSD calculation
            rssd = np.sqrt(
                np.sum(
                    (
                        decomp_spend_dist["effect_share"]
                        - decomp_spend_dist["spend_share"]
                    )
                    ** 2
                )
            )

            # Apply zero penalty if requested
            if rssd_zero_penalty:
                is_zero_effect = np.round(decomp_spend_dist["effect_share"], 4) == 0
                share_zero_effect = np.sum(is_zero_effect) / len(decomp_spend_dist)
                rssd *= 1 + share_zero_effect

        else:
            # Refresh case
            dt_decomp_rf = pd.DataFrame(
                {"rn": x_decomp_agg["rn"], "decomp_perc": x_decomp_agg["xDecompPerc"]}
            )

            # Join with previous decomposition
            dt_decomp_rf = dt_decomp_rf.merge(
                self.previous_decomp[["rn", "xDecompPerc"]].rename(
                    columns={"xDecompPerc": "decomp_perc_prev"}
                ),
                on="rn",
                how="left",
            )

            # Calculate media RSSD
            media_mask = dt_decomp_rf["rn"].isin(
                self.mmm_data.mmmdata_spec.paid_media_spends
            )
            rssd_media = np.sqrt(
                np.mean(
                    (
                        dt_decomp_rf.loc[media_mask, "decomp_perc"]
                        - dt_decomp_rf.loc[media_mask, "decomp_perc_prev"]
                    )
                    ** 2
                )
            )

            # Calculate non-media RSSD
            rssd_non_media = np.sqrt(
                np.mean(
                    (
                        dt_decomp_rf.loc[~media_mask, "decomp_perc"]
                        - dt_decomp_rf.loc[~media_mask, "decomp_perc_prev"]
                    )
                    ** 2
                )
            )

            # Combine them
            rssd = rssd_media + rssd_non_media / (
                1
                - self.mmm_data.mmmdata_spec.refresh_steps
                / self.mmm_data.mmmdata_spec.rolling_window_length
            )

        # 6. Handle zero coefficient case
        if np.isnan(rssd):
            rssd = np.inf
            decomp_spend_dist["effect_share"] = 0

        elapsed_time = time.time() - start_time

        # Format hyperparameter names to match R's format
        params_formatted = self.ridge_data_builder._format_hyperparameter_names(params)

        # Update metrics dictionary
        metrics.update(
            {
                "decomp_rssd": float(rssd),
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
        # Get paid media rows from x_decomp_agg
        paid_media_mask = x_decomp_agg["rn"].isin(
            self.mmm_data.mmmdata_spec.paid_media_spends
        )
        paid_media_rows = x_decomp_agg[paid_media_mask]

        # Add debug logging
        self.logger.debug("Spend metrics structure:")
        self.logger.debug(
            f"total_spend: {self.ridge_data_builder.spend_metrics['total_spend']}"
        )
        self.logger.debug(
            f"mean_spend: {self.ridge_data_builder.spend_metrics['mean_spend']}"
        )
        self.logger.debug(
            f"spend_share: {self.ridge_data_builder.spend_metrics['spend_share']}"
        )

        self.logger.debug("\nPaid media channels:")
        self.logger.debug(
            f"x_decomp_agg filtered rn: {x_decomp_agg.loc[paid_media_mask, 'rn'].tolist()}"
        )
        # Create mapping of channel names to indices
        channel_to_index = {
            channel: idx
            for idx, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends)
        }

        self.logger.debug("Channel to index mapping:")
        self.logger.debug(channel_to_index)
        # Add all required columns with correct types
        decomp_spend_dist = pd.DataFrame(
            {
                "rn": paid_media_rows["rn"],
                "coef": paid_media_rows["coef"],
                "xDecompAgg": paid_media_rows["xDecompAgg"],
                "xDecompPerc": paid_media_rows[
                    "xDecompPerc"
                ],  # Get this from paid_media_rows
                # Use list indices via the mapping
                "total_spend": [
                    self.ridge_data_builder.spend_metrics["total_spend"][
                        channel_to_index[name]
                    ]
                    for name in paid_media_rows["rn"]
                ],
                "mean_spend": [
                    self.ridge_data_builder.spend_metrics["mean_spend"][
                        channel_to_index[name]
                    ]
                    for name in paid_media_rows["rn"]
                ],
                "spend_share": [
                    self.ridge_data_builder.spend_metrics["spend_share"][
                        channel_to_index[name]
                    ]
                    for name in paid_media_rows["rn"]
                ],
                "effect_share": paid_media_rows["xDecompPerc"],
                "sol_id": metrics.get("sol_id", ""),
                "rsq_train": metrics.get("rsq_train", 0),
                "rsq_val": metrics.get("rsq_val", 0),
                "rsq_test": metrics.get("rsq_test", 0),
                "nrmse": metrics.get("nrmse", 0),
                "decomp_rssd": metrics.get("decomp_rssd", 0),
                "mape": metrics.get("mape", 0),
                "lambda": metrics.get("lambda", 0),
                "lambda_hp": metrics.get("lambda_hp", 0),
                "lambda_max": metrics.get("lambda_max", 0),
                "lambda_min_ratio": metrics.get("lambda_min_ratio", 0),
                "trial": metrics.get("trial", 0),
                "iterNG": metrics.get("iterNG", 0),
                "iterPar": metrics.get("iterPar", 0),
                "Elapsed": metrics.get("elapsed", 0),
                "pos": paid_media_rows["pos"],
            }
        )

        # Ensure correct column order
        required_cols = [
            "rn",
            "coef",
            "xDecompAgg",
            "xDecompPerc",
            "total_spend",
            "mean_spend",
            "spend_share",
            "effect_share",
            "sol_id",
            "rsq_train",
            "rsq_val",
            "rsq_test",
            "nrmse",
            "decomp_rssd",
            "mape",
            "lambda",
            "lambda_hp",
            "lambda_max",
            "lambda_min_ratio",
            "trial",
            "iterNG",
            "iterPar",
            "Elapsed",
            "pos",
        ]
        decomp_spend_dist = decomp_spend_dist[required_cols]

        # Format x_decomp_agg with all required columns and metrics
        x_decomp_agg = pd.DataFrame(
            {
                "rn": x_decomp_agg["rn"],
                "coef": x_decomp_agg["coef"],
                "xDecompAgg": x_decomp_agg["xDecompAgg"],
                "xDecompPerc": x_decomp_agg["xDecompPerc"],
                "xDecompMeanNon0": x_decomp_agg["xDecompMeanNon0"],
                "xDecompMeanNon0Perc": x_decomp_agg["xDecompMeanNon0Perc"],
                "xDecompAggRF": x_decomp_agg["xDecompAggRF"],
                "xDecompPercRF": x_decomp_agg["xDecompPercRF"],
                "xDecompMeanNon0RF": x_decomp_agg["xDecompMeanNon0RF"],
                "xDecompMeanNon0PercRF": x_decomp_agg["xDecompMeanNon0PercRF"],
                "pos": x_decomp_agg["pos"],
                # Add all metrics
                "train_size": float(metrics.get("train_size", 1.0)),
                "rsq_train": float(metrics.get("rsq_train", 0)),
                "rsq_val": float(metrics.get("rsq_val", 0)),
                "rsq_test": float(metrics.get("rsq_test", 0)),
                "nrmse_train": float(metrics.get("nrmse_train", 0)),
                "nrmse_val": float(metrics.get("nrmse_val", 0)),
                "nrmse_test": float(metrics.get("nrmse_test", 0)),
                "nrmse": float(metrics.get("nrmse", 0)),
                "decomp.rssd": float(metrics.get("decomp_rssd", 0)),
                "mape": float(metrics.get("mape", 0)),
                "lambda": float(metrics.get("lambda", 0)),
                "lambda_hp": float(metrics.get("lambda_hp", 0)),
                "lambda_max": float(metrics.get("lambda_max", 0)),
                "lambda_min_ratio": float(metrics.get("lambda_min_ratio", 0)),
                "sol_id": str(metrics.get("sol_id", "")),
                "trial": int(metrics.get("trial", 0)),
                "iterNG": int(metrics.get("iterNG", 0)),
                "iterPar": int(metrics.get("iterPar", 0)),
                "Elapsed": float(metrics.get("Elapsed", 0)),
            }
        )

        # Ensure correct column order
        required_cols = [
            "rn",
            "coef",
            "xDecompAgg",
            "xDecompPerc",
            "xDecompMeanNon0",
            "xDecompMeanNon0Perc",
            "xDecompAggRF",
            "xDecompPercRF",
            "xDecompMeanNon0RF",
            "xDecompMeanNon0PercRF",
            "pos",
            "train_size",
            "rsq_train",
            "rsq_val",
            "rsq_test",
            "nrmse_train",
            "nrmse_val",
            "nrmse_test",
            "nrmse",
            "decomp.rssd",
            "mape",
            "lambda",
            "lambda_hp",
            "lambda_max",
            "lambda_min_ratio",
            "sol_id",
            "trial",
            "iterNG",
            "iterPar",
            "Elapsed",
        ]
        x_decomp_agg = x_decomp_agg[required_cols]
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

    def _setup_sign_control(
        self, X: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], List[float], List[float], Dict[str, bool]]:
        """Set up sign control for model variables, matching R's implementation exactly.

        Args:
            X: Feature DataFrame with dep_var already removed

        Returns:
            Tuple containing:
            - x_sign: Dict mapping variable types to their signs
            - lower_limits: List of lower bounds for each variable
            - upper_limits: List of upper bounds for each variable
            - check_factor: Dict mapping column names to boolean indicating if they are factors
        """
        # Define signs grouped by variable type (matching R's structure)
        x_sign = {
            "prophet": ["default"] * 3,  # [trend, season, holiday]
            "context": ["default"] * len(self.mmm_data.mmmdata_spec.context_vars),
            "paid_media": ["positive"]
            * len(self.mmm_data.mmmdata_spec.paid_media_spends),
            "organic": "positive",  # Single string for organic, matching R
        }

        # Check for factor variables
        check_factor = {
            col: pd.api.types.is_categorical_dtype(X[col]) for col in X.columns
        }

        # Initialize limits for prophet vars
        lower_limits = [0] * 3  # trend, season, holiday
        upper_limits = [1] * 3

        # Handle negative trend case
        if "trend" in X.columns and X["trend"].sum() < 0:
            lower_limits[0] = -1
            upper_limits[0] = 0

        # Handle remaining variables
        for col in X.columns[3:]:  # Skip prophet vars
            if check_factor.get(col, False):
                level_n = len(X[col].unique())
                if level_n <= 1:
                    raise ValueError(
                        f"Factor variable {col} must have more than 1 level"
                    )

                # Get variable type and index
                if col in self.mmm_data.mmmdata_spec.context_vars:
                    sign = "default"
                elif col in self.mmm_data.mmmdata_spec.paid_media_spends:
                    sign = "positive"
                else:  # organic
                    sign = "positive"

                if sign == "positive":
                    lower_vec = [0] * (level_n - 1)
                    upper_vec = ["Inf"] * (level_n - 1)  # Match R's "Inf"
                elif sign == "negative":
                    lower_vec = ["-Inf"] * (level_n - 1)
                    upper_vec = [0] * (level_n - 1)
                else:  # default
                    lower_vec = ["-Inf"] * (level_n - 1)
                    upper_vec = ["Inf"] * (level_n - 1)

                lower_limits.extend(lower_vec)
                upper_limits.extend(upper_vec)
            else:
                # Handle non-factor variables
                if col in self.mmm_data.mmmdata_spec.context_vars:
                    lower_limits.append("-Inf")
                    upper_limits.append("Inf")
                else:  # paid_media or organic
                    lower_limits.append(0)
                    upper_limits.append("Inf")

        return x_sign, lower_limits, upper_limits, check_factor
