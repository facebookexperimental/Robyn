import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.linear_model import Ridge
import logging
from robyn.calibration.media_effect_calibration import MediaEffectCalibrator
import os
import json
from datetime import datetime


class RidgeMetricsCalculator:
    def __init__(self, mmm_data, hyperparameters, ridge_data_builder):
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.ridge_data_builder = ridge_data_builder
        self.logger = logging.getLogger(__name__)
        
        # Calculate lambda sequence once during initialization
        self.lambda_max = None
        self.lambda_min_ratio = 0.0001

    def initialize_lambda_sequence(self, X, y):
        """Calculate lambda_max and lambda_min exactly like R"""
        if self.lambda_max is None:
            # Convert inputs to numpy arrays if they're pandas objects
            X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
            y = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
            
            n = X.shape[0]
            
            # Scale X and y
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            y_mean = y.mean()
            y_std = y.std()
            
            X_scaled = (X - X_mean) / X_std
            y_scaled = (y - y_mean) / y_std
            
            # Calculate column sums of X'y directly
            col_sums = X_scaled.T @ y_scaled
            
            # Get max absolute column sum
            max_abs_col_sum = np.max(np.abs(col_sums))
            
            # Calculate lambda_max exactly as R does
            self.lambda_max = max_abs_col_sum * y_std / n * 100  # Base calculation
            
            # Calculate lambda_min using same ratio as R
            self.lambda_min_ratio = 0.0001  # Same as R's default
            self.lambda_min = self.lambda_max * self.lambda_min_ratio

    def get_lambda_from_hp(self, lambda_hp):
        """Convert lambda hyperparameter to actual lambda value using R's method"""
        if self.lambda_max is None:
            raise ValueError("Must call initialize_lambda_sequence first")
        
        # Use linear interpolation like R
        lambda_scaled = self.lambda_min + (self.lambda_max - self.lambda_min) * lambda_hp
        return lambda_scaled

    def debug_model_metrics(self, X, y, lambda_info, metrics, iteration=None):
        """Log debug metrics to JSON file every 10 iterations"""
        if iteration is None or iteration == 1 or iteration % 10 == 0:
            debug_info = {
                "iteration": metrics.get("iterNG", iteration),
                "nrmse_train": metrics.get("nrmse_train"),
                "nrmse_val": metrics.get("nrmse_val"),
                "nrmse_test": metrics.get("nrmse_test"),
                "nrmse": metrics.get("nrmse"),
                "rsq_train": metrics.get("rsq_train"),
                "rsq_val": metrics.get("rsq_val"), 
                "rsq_test": metrics.get("rsq_test"),
                "lambda": metrics.get("lambda"),
                "lambda_max": metrics.get("lambda_max"),
                "lambda_hp": metrics.get("lambda_hp"),
                "decomp_rssd": metrics.get("decomp_rssd"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Read existing logs if file exists
            json_path = os.path.join(os.getcwd(), "python_debug_model_data.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_logs = json.load(f)
            else:
                existing_logs = {}

            # Use iteration number as key if available, otherwise use sequential number
            key = str(iteration) if iteration is not None else str(len(existing_logs) + 1)
            existing_logs[key] = debug_info

            # Write updated logs
            with open(json_path, 'w') as f:
                json.dump(existing_logs, f, indent=2)

    


    def _calculate_decomp_spend_dist(
        self, model: Ridge, X: pd.DataFrame, y: pd.Series, metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate decomposition spend distribution matching R's implementation exactly."""
        paid_media_cols = [
            col
            for col in X.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]


        
        # First pass calculations
        total_media_spend = np.abs(X[paid_media_cols].sum().sum())

        
        all_effects = {}
        all_spends = {}
        
        # Calculate and print individual channel effects
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            spend = np.abs(X[col].sum())
            effect = np.abs(coef * spend)
            all_effects[col] = effect
            all_spends[col] = spend
    
    
    
    
        
        total_effect = np.sum([e for e in all_effects.values()])


        # Second pass to calculate normalized metrics
        results = []
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = float(model.coef_[idx])
            spend = float(np.abs(all_spends[col]))
            effect = float(all_effects[col])

            # Handle non-zero values properly
            non_zero_mask = X[col] != 0
            non_zero_effect = np.abs(
                X[col][non_zero_mask] * coef
            )  # Changed to use absolute value
            non_zero_mean = float(
                non_zero_effect.mean() if len(non_zero_effect) > 0 else 0
            )

            # Calculate normalized shares
            spend_share = (
                float(spend / total_media_spend) if total_media_spend > 0 else 0
            )
            effect_share = float(effect / total_effect) if total_effect > 0 else 0

            result = {
                "rn": str(col),
                "coef": float(coef),
                "xDecompAgg": float(effect),  # This is now positive
                "total_spend": float(spend),
                "mean_spend": float(np.abs(X[col].mean())),
                "spend_share": spend_share,
                "effect_share": effect_share,
                "xDecompPerc": effect_share,
                "xDecompMeanNon0": non_zero_mean,
                "xDecompMeanNon0Perc": float(
                    non_zero_mean
                    / sum(
                        [
                            all_effects[c] / X[c][X[c] != 0].size
                            for c in paid_media_cols
                            if any(X[c] != 0)
                        ]
                    )
                    if any(X[c][X[c] != 0].size > 0 for c in paid_media_cols)
                    else 0
                ),
                "pos": bool(coef >= 0),
                "sol_id": str(metrics.get("sol_id", "")),
            }

            # Add model performance metrics
            for metric_key in [
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
            ]:
                result[metric_key] = float(metrics.get(metric_key, 0))

            result.update(
                {
                    "trial": int(metrics.get("trial", 0)),
                    "iterNG": int(metrics.get("iterNG", 0)),
                    "iterPar": int(metrics.get("iterPar", 0)),
                    "Elapsed": float(metrics.get("elapsed", 0)),
                    "pos": bool(coef >= 0),
                }
            )

            results.append(result)

        df = pd.DataFrame(results)

        # Ensure correct column types and order
        df = df.astype(
            {
                "rn": "str",
                "coef": "float64",
                "xDecompAgg": "float64",
                "total_spend": "float64",
                "mean_spend": "float64",
                "effect_share": "float64",
                "spend_share": "float64",
                "sol_id": "str",
                "pos": "bool",
                "mape": "int64",
                "trial": "int64",
                "iterNG": "int64",
                "iterPar": "int64",
            }
        )

        required_cols = [
            "rn",
            "coef",
            "xDecompAgg",
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

        return df[required_cols]

    def _calculate_rssd(
        self,
        model: Ridge,
        X: pd.DataFrame,
        paid_media_cols: List[str],
        rssd_zero_penalty: bool,
        debug: bool = True,
        iteration: int = 0,
    ) -> float:
        """Calculate RSSD with proper normalization"""
        total_raw_spend = np.sum(np.abs(X[paid_media_cols].sum()))
        effects = []
        spends = []

        # First collect all values
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            raw_spend = np.abs(X[col].sum())
            effect = np.abs(coef * raw_spend)  # Keep absolute effect
            effects.append(effect)
            spends.append(raw_spend)

        # Convert to numpy arrays
        effects = np.array(effects)
        spends = np.array(spends)

        # Calculate totals for normalization
        total_effect = np.sum(effects)
        total_spend = np.sum(spends)

        if total_effect > 0 and total_spend > 0:
            # Normalize proportionally
            effects_norm = effects / total_effect
            spends_norm = spends / total_spend

            # Calculate RSSD
            squared_diff = (effects_norm - spends_norm) ** 2
            rssd = np.sqrt(np.mean(squared_diff))

            if rssd_zero_penalty:
                zero_effects = sum(1 for e in effects if abs(e) < 1e-10)
                if zero_effects > 0:
                    rssd *= 1 + zero_effects / len(effects)

            return float(rssd)

        return float(np.inf)

    def _calculate_mape(
        self,
        model: Ridge,
        dt_raw: pd.DataFrame,
        hypParamSam: Dict[str, float],
        wind_start: int,
        wind_end: int,
    ) -> float:
        """
        Calculate MAPE using calibration data
        """
        if self.calibration_input is None:
            return 0.0

        try:
            # Use the MediaEffectCalibrator for MAPE calculation
            calibration_engine = MediaEffectCalibrator(
                mmm_data=self.mmm_data,
                hyperparameters=self.hyperparameters,
                calibration_input=self.calibration_input,
            )

            # Calculate MAPE using calibration engine
            lift_collect = calibration_engine.calibrate(
                df_raw=dt_raw,
                hypParamSam=hypParamSam,
                wind_start=wind_start,
                wind_end=wind_end,
                dayInterval=1,  # Default to 1 if not specified
                adstock=self.hyperparameters.adstock,
            )

            # Return mean MAPE across all lift studies
            if lift_collect is not None and not lift_collect.empty:
                return float(lift_collect["mape_lift"].mean())
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating MAPE: {str(e)}")
            return 0.0

    def _calculate_x_decomp_agg(
        self, model: Ridge, X: pd.DataFrame, y: pd.Series, metrics: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate x decomposition aggregates matching R's implementation exactly"""
        # Calculate decomposition effects with R-style scaling
        scale_factor = np.mean(np.abs(X)) * np.mean(np.abs(y))
        x_decomp = (X * model.coef_) * scale_factor
        x_decomp_sum = x_decomp.sum().sum()

        results = []
        for col in X.columns:
            coef = model.coef_[list(X.columns).index(col)]
            decomp_values = x_decomp[col]
            decomp_sum = decomp_values.sum()

            # Handle non-zero values with R's approach
            non_zero_mask = decomp_values != 0
            non_zero_values = decomp_values[non_zero_mask]
            non_zero_mean = non_zero_values.mean() if len(non_zero_values) > 0 else 0

            # Calculate total non-zero means across all columns
            total_non_zero_mean = sum(
                [
                    x_decomp[c][x_decomp[c] > 0].mean() if any(x_decomp[c] > 0) else 0
                    for c in X.columns
                ]
            )

            result = {
                "rn": str(col),  # Ensure string type
                "coef": float(coef),  # Ensure float type
                "xDecompAgg": float(decomp_sum),  # Ensure float type
                "xDecompPerc": float(
                    decomp_sum / x_decomp_sum if x_decomp_sum != 0 else 0
                ),
                "xDecompMeanNon0": float(non_zero_mean),
                "xDecompMeanNon0Perc": float(
                    non_zero_mean / total_non_zero_mean
                    if total_non_zero_mean != 0
                    else 0
                ),
                "xDecompAggRF": float(decomp_sum),  # RF version
                "xDecompPercRF": float(
                    decomp_sum / x_decomp_sum if x_decomp_sum != 0 else 0
                ),
                "xDecompMeanNon0RF": float(non_zero_mean),
                "xDecompMeanNon0PercRF": float(
                    non_zero_mean / total_non_zero_mean
                    if total_non_zero_mean != 0
                    else 0
                ),
                "pos": bool(coef >= 0),
            }

            # Add model performance metrics with correct types
            result.update(
                {
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
                    "lambda": float(
                        metrics.get("lambda", 0)
                    ),  # Critical: Using lambda not lambda_
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

            results.append(result)

        df = pd.DataFrame(results)

        # Ensure correct column order and types
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

        df = df[required_cols]
        return df

    def calculate_r2_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int, df_int: int = 1
    ) -> float:
        """Match R's R² calculation exactly"""
        n = len(y_true)

        # Scale data like R
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Calculate base R²
        r2 = 1 - (ss_res / ss_tot)

        # R's adjustment formula
        adj_r2 = 1 - ((1 - r2) * (n - df_int) / (n - n_features - df_int))

        # Match R's scale
        if adj_r2 < 0:
            adj_r2 = -np.sqrt(np.abs(adj_r2))  # R-style negative scaling

        return float(adj_r2)

    def calculate_nrmse(self, y_true: np.ndarray, y_pred: np.ndarray, debug: bool = True) -> float:
        """Calculate NRMSE with detailed debugging"""
        print("\n=== NRMSE Calculation Debug ===")
        
        # Add scale check
        print(f"Data ranges:")
        print(f"y_true range: [{y_true.min():.4f}, {y_true.max():.4f}]")
        print(f"y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        
        # Add distribution check
        print(f"\nDistribution stats:")
        print(f"y_true mean: {y_true.mean():.4f}, std: {y_true.std():.4f}")
        print(f"y_pred mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")
        
        n = len(y_true)
        residuals = y_true - y_pred
        rss = np.sum(residuals**2)
        
        # Print intermediate calculations
        print(f"\nIntermediate values:")
        print(f"n: {n}")
        print(f"RSS: {rss:.4f}")
        print(f"RMSE: {np.sqrt(rss/n):.4f}")
        
        scale = y_true.max() - y_true.min()
        nrmse = np.sqrt(rss/n) / scale if scale > 0 else np.sqrt(rss/n)
        
        print(f"\nFinal values:")
        print(f"Scale (y_range): {scale:.4f}")
        print(f"NRMSE: {nrmse:.4f}")
        print("===========================\n")
        
        return float(nrmse)
    
    def get_lambda_bounds(self):
        """Get lambda bounds for optimization"""
        return [0, 1]  # This matches R's approach where lambda_hp is between 0 and 1

    