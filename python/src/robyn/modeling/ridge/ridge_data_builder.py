import logging
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.signal import lfilter
import json
from datetime import datetime


class RidgeDataBuilder:
    def __init__(self, mmm_data, featurized_mmm_data):
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.logger = logging.getLogger(__name__)
        # Initialize base data during construction
        self.X_base, self.y_base = self._prepare_base_data()
        self.setup_initial_data()  # Run initial setup and logging

    def setup_initial_data(self):
        """One-time setup and logging of environment and spend metrics"""
        # Log step2 environment setup info
        self.logger.debug(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": "step2_environment_setup",
                    "data": {
                        "dt_mod_shape": [
                            len(self.featurized_mmm_data.dt_mod),
                            len(self.featurized_mmm_data.dt_mod.columns),
                        ],
                        "rolling_window": {
                            "start": self.mmm_data.mmmdata_spec.rolling_window_start_which,
                            "end": self.mmm_data.mmmdata_spec.rolling_window_end_which,
                            "length": self.mmm_data.mmmdata_spec.rolling_window_length,
                        },
                        "refresh": {
                            "added_start": str(
                                self.mmm_data.mmmdata_spec.refresh_added_start
                            ),
                            "steps": {},
                        },
                        "variables": {
                            "paid_media": self.mmm_data.mmmdata_spec.paid_media_spends,
                            "organic": self.mmm_data.mmmdata_spec.organic_vars,
                            "context": self.mmm_data.mmmdata_spec.context_vars,
                            "prophet": ["trend", "season", "holiday"],
                        },
                        "signs": {
                            "context": ["default"]
                            * len(self.mmm_data.mmmdata_spec.context_vars),
                            "paid_media": ["positive"]
                            * len(self.mmm_data.mmmdata_spec.paid_media_spends),
                            "prophet": ["default"] * 3,
                            "organic": "positive",
                        },
                        "adstock": "geometric",
                        "optimizer": "TwoPointsDE",
                    },
                },
                indent=2,
            )
        )

        # Calculate and log spend metrics
        self._calculate_spend_metrics()

    def _calculate_spend_metrics(self):
        """Calculate and log spend metrics"""
        start_idx = self.mmm_data.mmmdata_spec.rolling_window_start_which
        end_idx = self.mmm_data.mmmdata_spec.rolling_window_end_which
        dt_input_train = self.mmm_data.data.iloc[start_idx : end_idx + 1]

        paid_media_cols = [
            col
            for col in dt_input_train.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]

        # Calculate initial spend metrics
        total_spend = dt_input_train[paid_media_cols].sum().sum()
        total_spends = []
        mean_spends = []
        spend_shares = []

        for col in paid_media_cols:
            col_total = float(dt_input_train[col].sum())
            col_mean = float(dt_input_train[col].mean())
            col_share = (
                float(round(col_total / total_spend, 4)) if total_spend > 0 else 0
            )

            total_spends.append(col_total)
            mean_spends.append(col_mean)
            spend_shares.append(col_share)

        # Handle refresh metrics
        refresh_spends = total_spends.copy()
        refresh_means = mean_spends.copy()
        refresh_shares = spend_shares.copy()

        if self.mmm_data.mmmdata_spec.refresh_counter > 0:
            refresh_start = pd.to_datetime(
                self.mmm_data.mmmdata_spec.refresh_added_start
            )
            refresh_data = dt_input_train[dt_input_train["ds"] >= refresh_start]
            refresh_total = refresh_data[paid_media_cols].sum().sum()

            refresh_spends = []
            refresh_means = []
            refresh_shares = []

            for col in paid_media_cols:
                col_total = float(refresh_data[col].sum())
                col_mean = float(refresh_data[col].mean())
                col_share = (
                    float(round(col_total / refresh_total, 4))
                    if refresh_total > 0
                    else 0
                )

                refresh_spends.append(col_total)
                refresh_means.append(col_mean)
                refresh_shares.append(col_share)

        # Log step3 spend share calculation
        self.logger.debug(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": "step3_spend_share_calculation",
                    "data": {
                        "input_train_shape": [
                            len(dt_input_train),
                            len(dt_input_train.columns),
                        ],
                        "input_data_columns": dt_input_train.columns.tolist(),
                        "initial_spend_share": {
                            "rn": paid_media_cols,
                            "total_spend": total_spends,
                            "mean_spend": mean_spends,
                            "spend_share": spend_shares,
                            "total_spend_refresh": refresh_spends,
                            "mean_spend_refresh": refresh_means,
                            "spend_share_refresh": refresh_shares,
                        },
                        "refresh_spend_share": {
                            "rn": paid_media_cols,
                            "total_spend": refresh_spends,
                            "mean_spend": refresh_means,
                            "spend_share": refresh_shares,
                        },
                        "final_spend_share": {
                            "rn": paid_media_cols,
                            "total_spend": total_spends,
                            "mean_spend": mean_spends,
                            "spend_share": spend_shares,
                            "total_spend_refresh": refresh_spends,
                            "mean_spend_refresh": refresh_means,
                            "spend_share_refresh": refresh_shares,
                        },
                    },
                },
                indent=2,
            )
        )

    def _prepare_base_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """One-time preparation of base data structure"""
        # Get the dependent variable
        dep_var = self.mmm_data.mmmdata_spec.dep_var
        if dep_var not in self.featurized_mmm_data.dt_mod.columns:
            if "dep_var" in self.featurized_mmm_data.dt_mod.columns:
                self.featurized_mmm_data.dt_mod = (
                    self.featurized_mmm_data.dt_mod.rename(columns={"dep_var": dep_var})
                )
            else:
                raise KeyError(f"Could not find dependent variable column")

        y = self.featurized_mmm_data.dt_mod[dep_var]

        # Select features
        exclude_cols = ["ds", dep_var, "dep_var"]
        exclude_cols = [
            col
            for col in exclude_cols
            if col in self.featurized_mmm_data.dt_mod.columns
        ]
        X = self.featurized_mmm_data.dt_mod.drop(columns=exclude_cols)

        # Handle date columns
        date_columns = X.select_dtypes(include=["datetime64", "object"]).columns
        for col in date_columns:
            X[col] = pd.to_datetime(X[col], errors="coerce", format="%Y-%m-%d")
            min_date = X[col].min()
            X[col] = X[col].fillna(min_date)
            X[col] = (
                (X[col] - min_date).dt.total_seconds().div(86400).fillna(0).astype(int)
            )

        # One-hot encode categorical variables
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        return X, y

    @staticmethod
    def safe_astype(df: pd.DataFrame, type_dict: Dict[str, str]) -> pd.DataFrame:
        """Only cast columns that exist in the DataFrame"""
        existing_cols = {
            col: dtype for col, dtype in type_dict.items() if col in df.columns
        }
        return df.astype(existing_cols) if existing_cols else df

    def _format_hyperparameter_names(
        self, params: Dict[str, float]
    ) -> Dict[str, float]:
        """Format hyperparameter names to match R's naming convention."""
        formatted = {}
        for param_name, value in params.items():
            if param_name == "lambda" or param_name == "train_size":
                formatted[param_name] = value
            else:
                # Split parameter name into media and param type
                # E.g., facebook_S_alphas -> (facebook_S, alphas)
                media, param_type = param_name.rsplit("_", 1)
                if param_type in ["alphas", "gammas", "thetas", "shapes", "scales"]:
                    formatted[f"{media}_{param_type}"] = value
                else:
                    formatted[param_name] = value
        return formatted

    @staticmethod
    def _hyper_collector(
        hyperparameters: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: Optional[int],
    ) -> Dict[str, Any]:
        """
        Collect and organize hyperparameters to match R's structure
        """
        logger = logging.getLogger(__name__)
        logger.info("Collecting hyperparameters for optimization...")
        prepared_hyperparameters = hyperparameters["prepared_hyperparameters"]
        hyper_collect = {
            "hyper_list_all": {},
            "hyper_bound_list_updated": {},
            "hyper_bound_list_fixed": {},
            "all_fixed": False,
        }

        # Adjust hyper_list_all to store lists
        for channel, channel_params in prepared_hyperparameters.hyperparameters.items():
            for param in ["thetas", "alphas", "gammas"]:
                param_value = getattr(channel_params, param, None)
                if param_value is not None:
                    if isinstance(param_value, list) and len(param_value) == 2:
                        param_key = f"{channel}_{param}"
                        hyper_collect["hyper_bound_list_updated"][
                            param_key
                        ] = param_value
                        hyper_collect["hyper_list_all"][
                            f"{channel}_{param}"
                        ] = param_value  # Store as list
                    elif not isinstance(param_value, list):
                        hyper_collect["hyper_bound_list_fixed"][
                            f"{channel}_{param}"
                        ] = param_value
                        hyper_collect["hyper_list_all"][f"{channel}_{param}"] = [
                            param_value,
                            param_value,
                        ]  # Store as list
        # Handle lambda parameter similarly
        if (
            isinstance(prepared_hyperparameters.lambda_, list)
            and len(prepared_hyperparameters.lambda_) == 2
        ):
            hyper_collect["hyper_bound_list_updated"][
                "lambda"
            ] = prepared_hyperparameters.lambda_
            hyper_collect["hyper_list_all"]["lambda"] = prepared_hyperparameters.lambda_
        else:
            hyper_collect["hyper_bound_list_fixed"][
                "lambda"
            ] = prepared_hyperparameters.lambda_
            hyper_collect["hyper_list_all"]["lambda"] = [
                prepared_hyperparameters.lambda_,
                prepared_hyperparameters.lambda_,
            ]
        # Handle train_size similarly
        if ts_validation:
            if (
                isinstance(prepared_hyperparameters.train_size, list)
                and len(prepared_hyperparameters.train_size) == 2
            ):
                hyper_collect["hyper_bound_list_updated"][
                    "train_size"
                ] = prepared_hyperparameters.train_size
                hyper_collect["hyper_list_all"][
                    "train_size"
                ] = prepared_hyperparameters.train_size
            else:
                train_size = [0.5, 0.8]
                hyper_collect["hyper_bound_list_updated"]["train_size"] = train_size
                hyper_collect["hyper_list_all"]["train_size"] = train_size
        else:
            hyper_collect["hyper_list_all"]["train_size"] = [1.0, 1.0]

        # Add debug logging with pretty printing
        logger.debug(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": "step1_hyperparameter_collection",
                    "data": {
                        "hyper_param_names": list(
                            hyper_collect["hyper_list_all"].keys()
                        ),
                        "updated_bounds": {
                            k: {"min": v[0], "max": v[1]}
                            for k, v in hyper_collect[
                                "hyper_bound_list_updated"
                            ].items()
                        },
                        "fixed_bounds": hyper_collect["hyper_bound_list_fixed"],
                        "hyper_count": len(hyper_collect["hyper_list_all"]),
                        "hyper_count_fixed": len(
                            hyper_collect["hyper_bound_list_fixed"]
                        ),
                        "all_fixed": hyper_collect["all_fixed"],
                    },
                },
                indent=2,
            )
        )

        return hyper_collect

    def _geometric_adstock(self, x: pd.Series, theta: float) -> np.ndarray:
        """Apply geometric adstock transformation"""
        # Convert to numpy array if it's a pandas Series
        x_array = x.values if isinstance(x, pd.Series) else x

        # Use lfilter to efficiently compute the geometric transformation
        y = lfilter([1], [1, -theta], x_array)
        return y

    def _hill_transformation(
        self,
        x: np.ndarray,
        alpha: float,
        gamma: float,
        x_marginal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Hill transformation with optional marginal effect calculation"""
        x_array = np.array(x)
        x_scaled = (x_array - x_array.min()) / (x_array.max() - x_array.min())

        if x_marginal is not None:
            # Calculate marginal effect for carryover
            x_marginal_scaled = x_marginal / (x_array.max() - x_array.min())
            result = (
                x_marginal_scaled
                * (alpha * x_scaled ** (alpha - 1) * gamma**alpha)
                / (x_scaled**alpha + gamma**alpha) ** 2
            )
        else:
            # Regular hill transformation
            result = x_scaled**alpha / (x_scaled**alpha + gamma**alpha)

        return result

    def run_transformations(
        self,
        params: Dict[str, float],
        current_iteration: int = 1,
        total_iterations: int = 1,
        cores: int = 1,
    ) -> Dict[str, pd.DataFrame]:
        """Run transformations exactly like R implementation"""
        # Step7a logging (keep this part as is)
        self.logger.debug(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": f"step7a_transformation_inputs_iteration_{current_iteration}",
                    "data": {
                        "input_data": {
                            "dt_mod_shape": [
                                len(self.featurized_mmm_data.dt_mod),
                                len(self.featurized_mmm_data.dt_mod.columns),
                            ],
                            "dt_mod_columns": self.featurized_mmm_data.dt_mod.columns.tolist(),
                            "window_info": {
                                "start": self.mmm_data.mmmdata_spec.rolling_window_start_which,
                                "end": self.mmm_data.mmmdata_spec.rolling_window_end_which,
                                "length": self.mmm_data.mmmdata_spec.rolling_window_length,
                            },
                            "media_variables": self.mmm_data.mmmdata_spec.paid_media_spends,
                            "hyperparameters": params,
                        }
                    },
                },
                indent=2,
            )
        )

        # 1. Remove 'ds' column first
        dt_modAdstocked = self.X_base.drop(
            columns=["ds"] if "ds" in self.X_base.columns else []
        )
        self.logger.debug(
            f"After ds drop - dt_modAdstocked shape: {dt_modAdstocked.shape}"
        )

        # 2. Get window indices - FIXED to match R exactly
        window_start = self.mmm_data.mmmdata_spec.rolling_window_start_which
        window_end = self.mmm_data.mmmdata_spec.rolling_window_end_which
        window_length = self.mmm_data.mmmdata_spec.rolling_window_length

        # Calculate window indices to match R exactly
        window_indices = list(
            range(window_start, window_end + 1)
        )  # Convert to list for consistent indexing

        self.logger.debug(f"Window indices - start: {window_start}, end: {window_end}")
        self.logger.debug(f"Expected window length: {window_length}")
        self.logger.debug(f"Actual window indices length: {len(window_indices)}")
        self.logger.debug(f"First few indices: {window_indices[:5]}")
        self.logger.debug(f"Last few indices: {window_indices[-5:]}")

        # Storage for transformed data
        adstocked_collect = {}
        saturated_total_collect = {}
        saturated_immediate_collect = {}
        saturated_carryover_collect = {}

        # Get media variables
        media_vars = self.mmm_data.mmmdata_spec.paid_media_spends
        self.logger.debug(f"Processing media variables: {media_vars}")

        for var in media_vars:
            # Log progress
            self.logger.debug(f"Processing variable: {var}")

            # 1. Adstocking (whole data)
            m = dt_modAdstocked[var]
            theta = params[f"{var}_thetas"]

            # Apply geometric adstock to full dataset
            input_total = self._geometric_adstock(m, theta)
            input_immediate = m.values
            adstocked_collect[var] = input_total
            input_carryover = input_total - input_immediate

            # Log shapes after adstocking
            self.logger.debug(f"{var} adstocked shape: {len(input_total)}")

            # 2. Saturation (only window data)
            input_total_rw = input_total[window_indices]
            input_carryover_rw = input_carryover[window_indices]

            # Log shapes after windowing
            self.logger.debug(f"{var} windowed shape: {len(input_total_rw)}")

            alpha = params[f"{var}_alphas"]
            gamma = params[f"{var}_gammas"]

            # Apply saturation to windowed data
            sat_total = self._hill_transformation(input_total_rw, alpha, gamma)
            sat_carryover = self._hill_transformation(
                input_total_rw, alpha, gamma, x_marginal=input_carryover_rw
            )

            saturated_total_collect[var] = sat_total
            saturated_carryover_collect[var] = sat_carryover
            saturated_immediate_collect[var] = sat_total - sat_carryover

            # Log shapes after saturation
            self.logger.debug(f"{var} saturated shape: {len(sat_total)}")

        # EXACTLY match R's flow:
        # 1. First update dt_modAdstocked with adstocked values (full data)
        dt_modAdstocked = dt_modAdstocked.drop(columns=media_vars)
        for var, values in adstocked_collect.items():
            dt_modAdstocked[var] = values

        self.logger.debug(
            f"dt_modAdstocked shape before windowing: {dt_modAdstocked.shape}"
        )

        # 2. Then window and create dt_modSaturated (exactly like R)
        dt_modSaturated = dt_modAdstocked.iloc[window_indices].drop(columns=media_vars)
        self.logger.debug(
            f"dt_modSaturated shape after windowing: {dt_modSaturated.shape}"
        )

        # Create DataFrame from saturated_total_collect with explicit index
        saturated_df = pd.DataFrame(
            saturated_total_collect, index=dt_modSaturated.index
        )
        self.logger.debug(f"saturated_df shape before concat: {saturated_df.shape}")

        # Concatenate with matching indices
        dt_modSaturated = pd.concat([dt_modSaturated, saturated_df], axis=1)
        self.logger.debug(
            f"dt_modSaturated shape after concat: {dt_modSaturated.shape}"
        )

        # 3. Create immediate and carryover dataframes (already windowed)
        dt_saturatedImmediate = pd.DataFrame(
            saturated_immediate_collect, index=dt_modSaturated.index
        ).fillna(0)
        dt_saturatedCarryover = pd.DataFrame(
            saturated_carryover_collect, index=dt_modSaturated.index
        ).fillna(0)

        self.logger.debug(f"Immediate shape: {dt_saturatedImmediate.shape}")
        self.logger.debug(f"Carryover shape: {dt_saturatedCarryover.shape}")

        # Window y data using same indices
        self.y_windowed = self.y_base.iloc[window_indices]
        self.logger.debug(f"y_windowed shape: {len(self.y_windowed)}")

        # Additional debug info
        self.logger.debug(f"dt_modSaturated index length: {len(dt_modSaturated.index)}")
        self.logger.debug(f"saturated_df index length: {len(saturated_df.index)}")

        # Verify shapes match
        assert len(dt_modSaturated) == len(self.y_windowed) == window_length, (
            f"Shape mismatch: dt_modSaturated has {len(dt_modSaturated)} rows, "
            f"y_windowed has {len(self.y_windowed)} rows, "
            f"expected {window_length} rows"
        )

        return {
            "dt_modSaturated": dt_modSaturated,
            "dt_saturatedImmediate": dt_saturatedImmediate,
            "dt_saturatedCarryover": dt_saturatedCarryover,
            "y": self.y_windowed,
        }
