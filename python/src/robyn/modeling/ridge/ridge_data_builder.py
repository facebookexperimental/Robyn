import logging
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.signal import lfilter
import json
from datetime import datetime
from robyn.modeling.ridge.ridge_metrics_calculator import RidgeMetricsCalculator


class RidgeDataBuilder:
    def __init__(self, mmm_data, featurized_mmm_data, ridge_metrics_calculator):
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.logger = logging.getLogger(__name__)
        self.ridge_metrics_calculator = ridge_metrics_calculator
        # Initialize base data during construction
        self.X_base, self.y_base = self._prepare_base_data()
        self.setup_initial_data()  # Run initial setup and logging
        self.ridge_metrics_calculator.initialize_lambda_sequence(
            self.X_base, self.y_base, seq_len=100
        )

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
        self.spend_metrics = self._calculate_spend_metrics()

    def _calculate_spend_metrics(self):
        """Calculate and log spend metrics for all media channels"""
        start_idx = self.mmm_data.mmmdata_spec.rolling_window_start_which
        end_idx = self.mmm_data.mmmdata_spec.rolling_window_end_which
        dt_input_train = self.mmm_data.data.iloc[start_idx : end_idx + 1]

        # Get list of paid media columns
        paid_media_cols = [
            col
            for col in dt_input_train.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]

        # Calculate initial spend metrics
        total_spend = (
            dt_input_train[paid_media_cols].sum().sum()
        )  # Grand total across all channels
        total_spends = []  # List to store total spend per individual channel
        mean_spends = []  # List to store mean spend per individual channel
        spend_shares = []  # List to store each channel's share of total spend

        # Calculate metrics for each channel
        for col in paid_media_cols:
            col_total = float(dt_input_train[col].sum())  # Total spend for this channel
            col_mean = float(dt_input_train[col].mean())  # Mean spend for this channel
            col_share = (  # This channel's share of total spend
                float(round(col_total / total_spend, 4)) if total_spend > 0 else 0
            )

            total_spends.append(col_total)
            mean_spends.append(col_mean)
            spend_shares.append(col_share)

        # Initialize refresh metrics with initial values
        refresh_spends = (
            total_spends.copy()
        )  # Total spend per channel for refresh period
        refresh_means = mean_spends.copy()  # Mean spend per channel for refresh period
        refresh_shares = spend_shares.copy()  # Spend shares for refresh period

        # Recalculate metrics for refresh period if needed
        if self.mmm_data.mmmdata_spec.refresh_counter > 0:
            refresh_start = pd.to_datetime(
                self.mmm_data.mmmdata_spec.refresh_added_start
            )
            # Get data only from refresh start date
            refresh_data = dt_input_train[dt_input_train["ds"] >= refresh_start]
            refresh_total = (
                refresh_data[paid_media_cols].sum().sum()
            )  # Grand total for refresh period

            refresh_spends = []
            refresh_means = []
            refresh_shares = []

            # Calculate refresh metrics for each channel
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

        # Log all metrics for debugging
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
        return {
            "rn": paid_media_cols,
            "total_spend": total_spends,
            "mean_spend": mean_spends,
            "spend_share": spend_shares,
            "total_spend_refresh": refresh_spends,
            "mean_spend_refresh": refresh_means,
            "spend_share_refresh": refresh_shares,
        }

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

        # Create a sorted list of parameter names to match R's alphabetical ordering
        param_names = []
        for channel, channel_params in prepared_hyperparameters.hyperparameters.items():
            for param in ["alphas", "gammas", "thetas"]:
                param_value = getattr(channel_params, param, None)
                if param_value is not None:
                    param_names.append(f"{channel}_{param}")

        # Sort parameter names alphabetically to match R
        param_names.sort()

        # Process parameters in alphabetical order
        for param_key in param_names:
            channel, param = param_key.rsplit("_", 1)
            param_value = getattr(
                prepared_hyperparameters.hyperparameters[channel], param
            )

            if isinstance(param_value, list) and len(param_value) == 2:
                hyper_collect["hyper_bound_list_updated"][param_key] = param_value
                hyper_collect["hyper_list_all"][param_key] = param_value
            else:
                hyper_collect["hyper_bound_list_fixed"][param_key] = param_value
                hyper_collect["hyper_list_all"][param_key] = [param_value, param_value]

        # Add lambda parameter (like R)
        hyper_collect["hyper_bound_list_updated"]["lambda"] = [0, 1]  # Lambda_hp bounds
        hyper_collect["hyper_list_all"]["lambda"] = [0, 1]

        # Handle train_size after media parameters
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

    def _geometric_adstock(self, x: pd.Series, theta: float) -> Dict[str, np.ndarray]:
        """Exactly match R's geometric adstock implementation"""
        x_array = x.values if isinstance(x, pd.Series) else x
        x_decayed = np.zeros_like(x_array)
        x_decayed[0] = x_array[0]

        # Calculate x_decayed exactly like R
        for i in range(1, len(x_array)):
            x_decayed[i] = x_array[i] + theta * x_decayed[i - 1]

        # Calculate thetaVecCum exactly like R
        thetaVecCum = np.zeros_like(x_array)
        thetaVecCum[0] = theta
        for i in range(1, len(x_array)):
            thetaVecCum[i] = thetaVecCum[i - 1] * theta

        # Calculate inflation_total
        inflation_total = np.sum(x_decayed) / np.sum(x_array)

        return {
            "x": x_array,
            "x_decayed": x_decayed,
            "thetaVecCum": thetaVecCum,
            "inflation_total": inflation_total,
        }

    def _hill_transformation(
        self,
        x: np.ndarray,
        alpha: float,
        gamma: float,
        x_marginal: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Exactly match R's Hill transformation implementation.

        Args:
            x: Input array
            alpha: Shape parameter
            gamma: Inflection point parameter
            x_marginal: Optional marginal values for carryover effects

        Returns:
            Dictionary containing x_saturated and inflexion point
        """
        x_array = np.array(x)

        # Calculate inflexion point exactly like R
        inflexion = np.max(x_array) * gamma

        if x_marginal is None:
            # Regular hill transformation (exactly like R)
            x_saturated = x_array**alpha / (x_array**alpha + inflexion**alpha)
        else:
            # Marginal effect calculation (exactly like R)
            x_saturated = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)

        return {"x_saturated": x_saturated, "inflexion": inflexion}

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

        # Storage for transformed data
        adstocked_collect = {}
        saturated_total_collect = {}
        saturated_immediate_collect = {}
        saturated_carryover_collect = {}

        # Process media variables
        media_vars = self.mmm_data.mmmdata_spec.paid_media_spends

        # Process each media variable (including newsletter)
        for var in media_vars + ["newsletter"]:

            # 1. Adstocking (whole data)
            input_data = dt_modAdstocked[var].values
            theta = params[f"{var}_thetas"]

            # Get adstocked values with all components
            adstock_result = self._geometric_adstock(input_data, theta)
            input_total = adstock_result["x_decayed"]
            input_immediate = adstock_result["x"]
            adstocked_collect[var] = input_total
            input_carryover = input_total - input_immediate

            # Store inflation metrics (like R)
            if not hasattr(self, "inflation_collect"):
                self.inflation_collect = {}
            self.inflation_collect[f"{var}_inflation"] = adstock_result[
                "inflation_total"
            ]

            # 2. Saturation (only window data)
            input_total_rw = input_total[window_indices]
            input_carryover_rw = input_carryover[window_indices]

            alpha = params[f"{var}_alphas"]
            gamma = params[f"{var}_gammas"]

            # Apply saturation (exactly like R)
            sat_result_total = self._hill_transformation(input_total_rw, alpha, gamma)
            sat_result_carryover = self._hill_transformation(
                input_total_rw, alpha, gamma, x_marginal=input_carryover_rw
            )

            saturated_total_collect[var] = sat_result_total["x_saturated"]
            saturated_carryover_collect[var] = sat_result_carryover["x_saturated"]
            saturated_immediate_collect[var] = (
                sat_result_total["x_saturated"] - sat_result_carryover["x_saturated"]
            )

            # Store inflexion points (like R)
            if not hasattr(self, "inflexion_collect"):
                self.inflexion_collect = {}
            self.inflexion_collect[f"{var}_inflexion"] = sat_result_total["inflexion"]

        # EXACTLY match R's flow:
        # 1. First update dt_modAdstocked with adstocked values (full data)
        dt_modAdstocked = dt_modAdstocked.drop(columns=media_vars)
        for var, values in adstocked_collect.items():
            dt_modAdstocked[var] = values

        # 2. Then window and create dt_modSaturated (exactly like R)
        dt_modSaturated = dt_modAdstocked.iloc[window_indices].copy()

        # Drop media columns before binding (exactly like R)
        dt_modSaturated = dt_modSaturated.drop(columns=media_vars + ["newsletter"])
        for var, values in saturated_total_collect.items():
            dt_modSaturated[var] = values

        # 2. Add dep_var as first column (like R)
        if "dep_var" not in dt_modSaturated.columns:
            dt_modSaturated.insert(0, "dep_var", self.y_base.iloc[window_indices])

        # 3. Create immediate and carryover dataframes
        # These should only contain media variables
        dt_saturatedImmediate = pd.DataFrame(
            saturated_immediate_collect, index=dt_modSaturated.index
        ).fillna(0)
        dt_saturatedCarryover = pd.DataFrame(
            saturated_carryover_collect, index=dt_modSaturated.index
        ).fillna(0)

        # Window y data using same indices
        self.y_windowed = self.y_base.iloc[window_indices]

        # Enhanced step7b logging with detailed transformation info
        self.logger.debug(
            json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": f"step7b_transformation_iteration_{current_iteration}",
                    "data": {
                        "transformation_info": {
                            "dt_modSaturated": {
                                "shape": list(dt_modSaturated.shape),
                                "columns": list(dt_modSaturated.columns),
                                "stats": {
                                    "min": dt_modSaturated.min().to_dict(),
                                    "max": dt_modSaturated.max().to_dict(),
                                    "mean": dt_modSaturated.mean().to_dict(),
                                    "std": dt_modSaturated.std().to_dict(),
                                },
                            },
                            "dt_saturatedImmediate": {
                                "shape": list(dt_saturatedImmediate.shape),
                                "columns": list(dt_saturatedImmediate.columns),
                                "stats": {
                                    "min": dt_saturatedImmediate.min().to_dict(),
                                    "max": dt_saturatedImmediate.max().to_dict(),
                                    "mean": dt_saturatedImmediate.mean().to_dict(),
                                    "std": dt_saturatedImmediate.std().to_dict(),
                                },
                            },
                            "dt_saturatedCarryover": {
                                "shape": list(dt_saturatedCarryover.shape),
                                "columns": list(dt_saturatedCarryover.columns),
                                "stats": {
                                    "min": dt_saturatedCarryover.min().to_dict(),
                                    "max": dt_saturatedCarryover.max().to_dict(),
                                    "mean": dt_saturatedCarryover.mean().to_dict(),
                                    "std": dt_saturatedCarryover.std().to_dict(),
                                },
                            },
                        },
                        "iteration_info": {
                            "current_iteration": current_iteration,
                            "total_iterations": total_iterations,
                            "cores": cores,
                        },
                        "y_info": {
                            "shape": len(self.y_windowed),
                            "stats": {
                                "min": float(self.y_windowed.min()),
                                "max": float(self.y_windowed.max()),
                                "mean": float(self.y_windowed.mean()),
                                "std": float(self.y_windowed.std()),
                            },
                        },
                    },
                },
                indent=2,
            )
        )
        return {
            "dt_modSaturated": dt_modSaturated,  # includes dep_var
            "dt_saturatedImmediate": dt_saturatedImmediate,  # media vars only
            "dt_saturatedCarryover": dt_saturatedCarryover,  # media vars only
            "y": self.y_windowed,  # keep for convenience
        }
