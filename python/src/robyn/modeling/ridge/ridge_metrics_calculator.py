import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from sklearn.linear_model import Ridge
import logging
from robyn.calibration.media_effect_calibration import MediaEffectCalibrator
import json


class RidgeMetricsCalculator:
    def __init__(self, mmm_data, hyperparameters):
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.logger = logging.getLogger(__name__)

        # Calculate lambda sequence once during initialization
        self.lambda_max = None
        self.lambda_min_ratio = 0.0001

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

    def calculate_r2_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        p: int,
        df_int: int = 1,
        n_train: Optional[int] = None,
    ) -> float:
        """Calculate R-squared score matching R's implementation exactly."""
        # Match R's SSE calculation order
        sse = np.sum((y_pred - y_true) ** 2)  # Changed order to match R
        y_mean = np.mean(y_true)
        sst = np.sum((y_true - y_mean) ** 2)
        r2 = 1 - (sse / sst)

        if p is not None and df_int is not None:
            n = n_train if n_train is not None else len(y_true)
            rdf = n - p - 1  # R's degrees of freedom calculation
            r2_adj = 1 - (1 - r2) * ((n - df_int) / rdf)
            return float(r2_adj)

        return float(r2)

    def calculate_nrmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate NRMSE matching R's implementation exactly.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: NRMSE value
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Calculate RMSE exactly as R does
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Calculate range using max-min of y_true only (like R)
        y_range = np.max(y_true) - np.min(y_true)

        # Normalize by range
        nrmse = rmse / y_range if y_range > 0 else np.nan

        return float(nrmse)

    def model_decomp(
        self,
        model: Ridge,
        y_pred: np.ndarray,
        dt_modSaturated: pd.DataFrame,
        dt_saturatedImmediate: pd.DataFrame,
        dt_saturatedCarryover: pd.DataFrame,
        dt_modRollWind: pd.DataFrame,
        refreshAddedStart: str,
    ) -> Dict[str, pd.DataFrame]:
        """Implement R's model_decomp function exactly.

        Args:
            model: Fitted Ridge model
            y_pred: Model predictions
            dt_modSaturated: Full model data
            dt_saturatedImmediate: Immediate effects data
            dt_saturatedCarryover: Carryover effects data
            dt_modRollWind: Rolling window data
            refreshAddedStart: Refresh period start date
        """
        # Get y and X like R
        y = dt_modSaturated["dep_var"]
        X = dt_modSaturated.drop(columns=["dep_var"])

        # Get intercept and coefficients
        intercept = model.intercept_
        coefs = model.coef_
        x_names = X.columns
        x_factor = [
            col
            for col, dtype in X.dtypes.items()
            if pd.api.types.is_categorical_dtype(dtype)
        ]

        # Calculate xDecomp like R
        xDecomp = pd.DataFrame(
            {col: X[col] * coef for col, coef in zip(x_names, coefs)}
        )
        xDecomp.insert(0, "intercept", intercept)

        # Create xDecompOut
        xDecompOut = pd.concat(
            [
                pd.DataFrame({"ds": dt_modRollWind["ds"], "y": y, "y_pred": y_pred}),
                xDecomp,
            ],
            axis=1,
        )

        # Calculate media decompositions
        media_cols = dt_saturatedImmediate.columns
        coefs_media = {col: coefs[list(X.columns).index(col)] for col in media_cols}

        mediaDecompImmediate = pd.DataFrame(
            {col: dt_saturatedImmediate[col] * coefs_media[col] for col in media_cols}
        )

        mediaDecompCarryover = pd.DataFrame(
            {col: dt_saturatedCarryover[col] * coefs_media[col] for col in media_cols}
        )

        # Calculate scaled decompositions
        y_hat = xDecomp.sum(axis=1)
        y_hat_scaled = np.abs(xDecomp).sum(axis=1)
        xDecompOutPerc_scaled = np.abs(xDecomp).div(y_hat_scaled, axis=0)
        xDecompOut_scaled = y_hat.multiply(xDecompOutPerc_scaled)

        # Calculate aggregations
        temp = xDecompOut[["intercept"] + list(x_names)]
        xDecompOutAgg = temp.sum()
        xDecompOutAggPerc = xDecompOutAgg / y_hat.sum()

        # Calculate non-zero means
        xDecompOutAggMeanNon0 = temp.apply(
            lambda x: x[x != 0].mean() if (x != 0).any() else 0
        )
        xDecompOutAggMeanNon0Perc = xDecompOutAggMeanNon0 / xDecompOutAggMeanNon0.sum()

        # Handle refresh period calculations
        refresh_mask = dt_modRollWind["ds"] >= refreshAddedStart
        temp_rf = temp[refresh_mask]
        xDecompOutAggRF = temp_rf.sum()
        y_hatRF = y_hat[refresh_mask]
        xDecompOutAggPercRF = xDecompOutAggRF / y_hatRF.sum()
        xDecompOutAggMeanNon0RF = temp_rf.apply(
            lambda x: x[x != 0].mean() if (x != 0).any() else 0
        )
        xDecompOutAggMeanNon0PercRF = (
            xDecompOutAggMeanNon0RF / xDecompOutAggMeanNon0RF.sum()
        )

        # Create final output
        decompOutAgg = pd.DataFrame(
            {
                "rn": ["intercept"] + list(x_names),
                "coef": [intercept] + list(coefs),
                "xDecompAgg": xDecompOutAgg,
                "xDecompPerc": xDecompOutAggPerc,
                "xDecompMeanNon0": xDecompOutAggMeanNon0,
                "xDecompMeanNon0Perc": xDecompOutAggMeanNon0Perc,
                "xDecompAggRF": xDecompOutAggRF,
                "xDecompPercRF": xDecompOutAggPercRF,
                "xDecompMeanNon0RF": xDecompOutAggMeanNon0RF,
                "xDecompMeanNon0PercRF": xDecompOutAggMeanNon0PercRF,
                "pos": [True] + [c >= 0 for c in coefs],
            }
        )

        # Add ds and y to media decomps
        mediaDecompImmediate["ds"] = xDecompOut["ds"]
        mediaDecompImmediate["y"] = xDecompOut["y"]
        mediaDecompCarryover["ds"] = xDecompOut["ds"]
        mediaDecompCarryover["y"] = xDecompOut["y"]

        return {
            "xDecompVec": xDecompOut,
            "xDecompVec.scaled": xDecompOut_scaled,
            "xDecompAgg": decompOutAgg,
            "mediaDecompImmediate": mediaDecompImmediate,
            "mediaDecompCarryover": mediaDecompCarryover,
        }
