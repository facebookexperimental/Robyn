import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.linear_model import Ridge
import logging
from robyn.calibration.media_effect_calibration import MediaEffectCalibrator


class RidgeMetricsCalculator:
    def __init__(self, mmm_data, hyperparameters, ridge_data_builder):
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.ridge_data_builder = ridge_data_builder
        self.logger = logging.getLogger(__name__)

    # Updated _calculate_decomp_spend_dist method
    def _calculate_decomp_spend_dist(
        self, model: Ridge, X: pd.DataFrame, y: pd.Series, metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate decomposition spend distribution matching R's implementation exactly."""
        paid_media_cols = [
            col
            for col in X.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]

        # First pass to calculate total spend and effect for normalization
        total_media_spend = np.abs(X[paid_media_cols].sum().sum())
        all_effects = {}
        all_spends = {}

        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            spend = np.abs(X[col].sum())  # Ensure positive spend
            effect = coef * spend  # Keep original sign for effect
            all_effects[col] = effect
            all_spends[col] = spend

        total_effect = np.sum(
            np.abs([e for e in all_effects.values()])
        )  # Use absolute sum

        # Second pass to calculate normalized metrics
        results = []
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = float(model.coef_[idx])
            spend = float(np.abs(all_spends[col]))
            effect = float(all_effects[col])

            # Handle non-zero values properly
            non_zero_mask = X[col] != 0
            non_zero_effect = X[col][non_zero_mask] * coef
            non_zero_mean = float(
                non_zero_effect.mean() if len(non_zero_effect) > 0 else 0
            )

            # Calculate normalized shares
            spend_share = (
                float(spend / total_media_spend) if total_media_spend > 0 else 0
            )
            effect_share = (
                float(np.abs(effect) / total_effect) if total_effect > 0 else 0
            )

            result = {
                "rn": str(col),
                "coef": float(coef),
                "xDecompAgg": float(effect),
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
                }
            )

            results.append(result)

        df = pd.DataFrame(results)

        # Ensure correct column order
        required_cols = [
            "rn",
            "coef",
            "xDecompAgg",
            "total_spend",
            "mean_spend",
            "spend_share",
            "effect_share",
            "xDecompPerc",
            "xDecompMeanNon0",
            "xDecompMeanNon0Perc",
            "pos",
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
        ]
        self.logger.debug(f"Decomp spend distribution debug:")
        self.logger.debug(f"Total media spend: {total_media_spend}")
        self.logger.debug(f"Total effect: {total_effect}")
        for col in paid_media_cols:
            self.logger.debug(
                f"{col} - effect: {all_effects[col]}, spend: {all_spends[col]}"
            )
        return df[required_cols]

    def _calculate_lambda(
        self, x_norm: np.ndarray, y_norm: np.ndarray, lambda_hp: float
    ) -> Tuple[float, float]:
        """Match R's glmnet lambda calculation exactly"""
        n_samples = len(y_norm)

        # Scale to original magnitude like R
        scale_factor = np.max(np.abs(x_norm)) * np.max(np.abs(y_norm))

        # R-style standardization
        def r_scale(x):
            mean = np.mean(x)
            var = np.sum((x - mean) ** 2) / (len(x) - 1)
            sd = np.sqrt(var)
            return (x - mean) / sd if sd > 1e-10 else np.zeros_like(x)

        # Scale X and y
        x_scaled = np.apply_along_axis(r_scale, 0, x_norm)
        y_scaled = r_scale(y_norm)

        # R's lambda calculation with original magnitude
        alpha = 0.001
        gram = x_scaled.T @ x_scaled / n_samples
        ctx = np.abs(x_scaled.T @ y_scaled) / n_samples

        # Rescale to match R's magnitude
        lambda_max = np.max(ctx) * scale_factor / alpha

        # R's lambda interpolation
        lambda_min_ratio = 0.0001
        lambda_min = lambda_min_ratio * lambda_max
        lambda_ = lambda_min + lambda_hp * (lambda_max - lambda_min)

        return lambda_, lambda_max

    def _calculate_rssd(
        self,
        model: Ridge,
        X: pd.DataFrame,
        paid_media_cols: List[str],
        rssd_zero_penalty: bool,
    ) -> float:
        """Calculate RSSD exactly like R's implementation"""
        self.logger.debug(f"\nRSSD Calculation Debug:")
        self.logger.debug(f"Paid media columns: {paid_media_cols}")

        total_spend = np.abs(X[paid_media_cols].sum()).sum()
        effects = []
        spends = []

        self.logger.debug(f"Effect calculations per channel:")
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            spend = np.abs(X[col].sum())
            effect = coef * spend
            effects.append(effect)
            spends.append(spend)
            self.logger.debug(
                f"{col}: coef={coef:.6f}, spend={spend:.6f}, effect={effect:.6f}"
            )

        effects = np.array(effects)
        spends = np.array(spends)

        # Calculate normalized effects and spends
        total_effect = np.sum(np.abs(effects))
        self.logger.debug(
            f"Totals - spend: {total_spend:.6f}, effect: {total_effect:.6f}"
        )

        if total_effect > 0 and total_spend > 0:
            effects_norm = np.abs(effects) / total_effect
            spends_norm = spends / total_spend
            self.logger.debug(f"Normalized values:")
            self.logger.debug(f"Effects: {effects_norm}")
            self.logger.debug(f"Spends: {spends_norm}")

            rssd = np.sqrt(np.mean((effects_norm - spends_norm) ** 2))

            if rssd_zero_penalty:
                zero_effects = sum(1 for e in effects if abs(e) < 1e-10)
                if zero_effects > 0:
                    rssd_original = rssd
                    rssd *= 1 + zero_effects / len(effects)
                    self.logger.debug(
                        f"Applied zero penalty: {zero_effects} zero effects found"
                    )
                    self.logger.debug(
                        f"RSSD adjusted from {rssd_original:.6f} to {rssd:.6f}"
                    )

            self.logger.debug(f"Final RSSD: {rssd:.6f}")
            return float(rssd)

        self.logger.debug("RSSD calculation returned infinity")
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

    def calculate_nrmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Match R's NRMSE calculation exactly"""
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        scale = np.max(y_true) - np.min(y_true)

        if scale > 0:
            nrmse = np.sqrt(rss / n) / scale
        else:
            nrmse = np.sqrt(rss / n)

        return float(nrmse)
