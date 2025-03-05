import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from sklearn.linear_model import Ridge
import logging
from robyn.calibration.media_effect_calibration import MediaEffectCalibrator
import json


class RidgeMetricsCalculator:
    def __init__(self, mmm_data, hyperparameters, ridge_data_builder):
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.ridge_data_builder = ridge_data_builder
        self.logger = logging.getLogger(__name__)

        # Calculate lambda sequence once during initialization
        self.lambda_max = None
        self.lambda_min_ratio = 0.0001

    def _calculate_decomp_spend_dist(
        self, model: Ridge, X: pd.DataFrame, y: pd.Series, metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate decomposition spend distribution matching R's implementation exactly."""
        paid_media_cols = [
            col
            for col in X.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]

        # Precompute a mapping from column names to their index in X.columns
        col_to_index = {col: idx for idx, col in enumerate(X.columns)}

        # First pass to calculate total spend and effect for normalization
        total_media_spend = np.abs(X[paid_media_cols].sum().sum())
        all_effects = {}
        all_spends = {}

        # Calculate effects using absolute values for scaling
        for col in paid_media_cols:
            idx = col_to_index[col]
            coef = model.coef_[idx]
            spend = np.abs(X[col].sum())  # Ensure positive spend
            # Use absolute values for effect calculation
            effect = np.abs(coef * spend)  # Changed to use absolute value
            all_effects[col] = effect
            all_spends[col] = spend

        total_effect = np.sum([e for e in all_effects.values()])

        # Precompute the non-zero count per column and denominator for xDecompMeanNon0Perc.
        # The denominator is the sum over paid media channels of (effect / count_non_zero) for channels with non-zero spends.
        count_non_zero = {col: (X[col] != 0).sum() for col in paid_media_cols}
        denom = sum(
            all_effects[col] / count_non_zero[col]
            for col in paid_media_cols
            if count_non_zero[col] > 0
        )
        # Second pass to calculate normalized metrics
        results = []
        for col in paid_media_cols:
            idx = col_to_index[col]
            coef = float(model.coef_[idx])
            spend = float(np.abs(all_spends[col]))
            effect = float(all_effects[col])

            # Handle non-zero values properly
            non_zero_mask = X[col] != 0
            non_zero_values = X.loc[non_zero_mask, col]
            non_zero_effect = np.abs(
                non_zero_values * coef
            )  # Changed to use absolute value
            non_zero_mean = float(
                non_zero_effect.mean() if len(non_zero_effect) > 0 else 0
            )

            # Calculate normalized shares
            spend_share = (
                float(spend / total_media_spend) if total_media_spend > 0 else 0
            )
            effect_share = float(effect / total_effect) if total_effect > 0 else 0
            # Calculate the percentage for non-zero mean; use precomputed denominator.

            xDecompMeanNon0Perc = float(non_zero_mean / denom) if denom > 0 else 0

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
                "xDecompMeanNon0Perc": xDecompMeanNon0Perc,
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

    def get_lambda_from_hp(self, lambda_hp):
        """Convert lambda hyperparameter to actual lambda value using R's method"""
        if self.lambda_max is None:
            raise ValueError("Must call initialize_lambda_sequence first")

        # Use linear interpolation like R
        lambda_scaled = (
            self.lambda_min + (self.lambda_max - self.lambda_min) * lambda_hp
        )
        return lambda_scaled

    def initialize_lambda_sequence(self, X, y, seq_len=100):
        """Calculate lambda sequence exactly like R's lambda_seq function"""
        if self.lambda_max is None:
            # Convert to numpy arrays
            X = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
            y = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

            # R's mysd function
            mysd = lambda x: np.sqrt(np.sum((x - np.mean(x)) ** 2) / len(x))

            # Scale X using mysd (like R's scale function)
            X_scaled = np.zeros_like(X)
            for j in range(X.shape[1]):
                sd = mysd(X[:, j])
                X_scaled[:, j] = (X[:, j] - np.mean(X[:, j])) / sd if sd != 0 else 0

            # Handle NaN values like R
            nan_cols = np.all(np.isnan(X_scaled), axis=0)
            X_scaled[:, nan_cols] = 0

            # Calculate lambda_max (R's way)
            lambda_max = np.max(np.abs(np.sum(X_scaled * y[:, None], axis=0))) / (
                0.001 * len(y)
            )

            # Generate lambda sequence
            log_seq = np.linspace(
                np.log(lambda_max), np.log(lambda_max * self.lambda_min_ratio), seq_len
            )
            lambdas = np.exp(log_seq)

            # Store final values (with 0.1 adjustment like R)
            self.lambda_max = np.max(lambdas) * 0.1  # Apply 0.1 to final lambda_max
            self.lambda_min = (
                self.lambda_max * self.lambda_min_ratio
            )  # Use adjusted lambda_max

            # Log debug info
            self.logger.debug(
                json.dumps(
                    {
                        "step": "step4_lambda_calculation",
                        "data": {
                            "lambda_min_ratio": self.lambda_min_ratio,
                            "lambda_sequence": {
                                "length": seq_len,
                                "min": float(
                                    np.min(lambdas)
                                ),  # Original sequence values
                                "max": float(
                                    np.max(lambdas)
                                ),  # Original sequence values
                                "mean": float(
                                    np.mean(lambdas)
                                ),  # Original sequence values
                            },
                            "final_values": {
                                "lambda_max": float(self.lambda_max),  # Adjusted by 0.1
                                "lambda_min": float(
                                    self.lambda_min
                                ),  # Based on adjusted lambda_max
                            },
                            "input_dimensions": {
                                "x_shape": list(X.shape),
                                "y_length": len(y),
                            },
                        },
                    },
                    indent=2,
                )
            )

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
        x_decomp_sum = x_decomp.to_numpy().sum()  # faster summing over all elements

        # Precompute total non-zero mean across all columns once
        total_non_zero_mean = sum(
            x_decomp[c][x_decomp[c] > 0].mean() if (x_decomp[c] > 0).any() else 0
            for c in X.columns
        )

        results = []
        # Use enumerate to iterate over columns and corresponding coefficients
        for idx, col in enumerate(X.columns):
            coef = model.coef_[idx]
            decomp_values = x_decomp[col]
            decomp_sum = decomp_values.sum()

            # Calculate non-zero mean for this column
            non_zero_mask = decomp_values != 0
            non_zero_mean = (
                decomp_values[non_zero_mask].mean() if non_zero_mask.any() else 0
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
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        p: int,  # number of features
        df_int: int = 1,  # degrees of freedom for intercept
        n_train: Optional[int] = None,  # training set size for validation/test
    ) -> float:
        """
        Calculate R-squared score matching R's implementation exactly.

        Args:
            y_true: True values
            y_pred: Predicted values
            p: Number of features (excluding intercept)
            df_int: Degrees of freedom for intercept (1 if intercept, 0 if not)
            n_train: Size of training set (used for validation/test adjustments)
        """
        n = len(y_true)
        n_adj = n_train if n_train is not None else n

        # Calculate R² components
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Base R²
        r2 = 1 - (ss_res / ss_tot)

        # Adjust R² using n_adj
        adj_r2 = 1 - ((1 - r2) * (n_adj - df_int) / (n_adj - p - df_int))

        # R-style negative scaling
        if adj_r2 < 0:
            adj_r2 = -np.sqrt(np.abs(adj_r2))

        return float(adj_r2)

    def calculate_nrmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate NRMSE matching R's implementation"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Calculate RMSE
        squared_errors = (y_true - y_pred) ** 2
        rmse = np.sqrt(np.mean(squared_errors))

        # Calculate range denominator
        y_range = np.max(y_true) - np.min(y_true)

        if y_range > 0:
            nrmse = rmse / y_range
        else:
            self.logger.warning("y_true range is 0, using rmse as nrmse")
            nrmse = rmse

        return float(nrmse)
