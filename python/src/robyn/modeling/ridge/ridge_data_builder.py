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

    def _prepare_data(self, params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ridge regression, handling dependent variable and excluding date columns"""

        # Get the dependent variable, handling both possible column names
        dep_var = self.mmm_data.mmmdata_spec.dep_var
        if dep_var not in self.featurized_mmm_data.dt_mod.columns:
            # If dep_var column doesn't exist, try 'dep_var'
            if "dep_var" in self.featurized_mmm_data.dt_mod.columns:
                # Rename 'dep_var' to the specified value
                self.featurized_mmm_data.dt_mod = (
                    self.featurized_mmm_data.dt_mod.rename(columns={"dep_var": dep_var})
                )
                y = self.featurized_mmm_data.dt_mod[dep_var]
            else:
                raise KeyError(
                    f"Could not find dependent variable column. Expected either '{dep_var}' or 'dep_var' in columns: {self.featurized_mmm_data.dt_mod.columns.tolist()}"
                )
        else:
            y = self.featurized_mmm_data.dt_mod[dep_var]

        # Select all columns except the dependent variable and date columns
        exclude_cols = ["ds"]  # Always exclude 'ds' if present
        if dep_var in self.featurized_mmm_data.dt_mod.columns:
            exclude_cols.append(dep_var)
        if "dep_var" in self.featurized_mmm_data.dt_mod.columns:
            exclude_cols.append("dep_var")

        # Only drop columns that actually exist in the dataframe
        exclude_cols = [
            col
            for col in exclude_cols
            if col in self.featurized_mmm_data.dt_mod.columns
        ]
        X = self.featurized_mmm_data.dt_mod.drop(columns=exclude_cols)

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
                            "steps": {},  # Fill this if refresh steps exist
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
        # Convert date columns to numeric (number of days since the earliest date)
        date_columns = X.select_dtypes(include=["datetime64", "object"]).columns
        for col in date_columns:
            X[col] = pd.to_datetime(X[col], errors="coerce", format="%Y-%m-%d")
            # Fill NaT (Not a Time) values with a default date (e.g., the minimum date in the column)
            min_date = X[col].min()
            X[col] = X[col].fillna(min_date)
            # Convert to days since minimum date, handling potential NaT values
            X[col] = (
                (X[col] - min_date).dt.total_seconds().div(86400).fillna(0).astype(int)
            )

        # One-hot encode categorical variables
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        # Ensure all columns are numeric
        X = X.select_dtypes(include=[np.number])

        # Apply transformations based on hyperparameters
        for media in self.mmm_data.mmmdata_spec.paid_media_spends:
            if f"{media}_thetas" in params:
                X[media] = self._geometric_adstock(X[media], params[f"{media}_thetas"])
            if f"{media}_alphas" in params and f"{media}_gammas" in params:
                X[media] = self._hill_transformation(
                    X[media], params[f"{media}_alphas"], params[f"{media}_gammas"]
                )

        # Handle any remaining NaN or infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
        X = X + 1e-8 * np.random.randn(*X.shape)

        # Get raw input data window for logging (matches R's dt_inputTrain)
        start_idx = self.mmm_data.mmmdata_spec.rolling_window_start_which
        end_idx = self.mmm_data.mmmdata_spec.rolling_window_end_which
        dt_input_train = self.mmm_data.data.iloc[start_idx : end_idx + 1]

        # Calculate initial spend metrics using training window
        paid_media_cols = [
            col
            for col in dt_input_train.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]

        # Initial spend calculations
        total_spends = []
        mean_spends = []
        spend_shares = []

        total_spend = dt_input_train[paid_media_cols].sum().sum()

        for col in paid_media_cols:
            col_total = float(dt_input_train[col].sum())
            col_mean = float(dt_input_train[col].mean())
            col_share = (
                float(round(col_total / total_spend, 4)) if total_spend > 0 else 0
            )

            total_spends.append(col_total)
            mean_spends.append(col_mean)
            spend_shares.append(col_share)

        # Calculate refresh spend metrics if in refresh mode
        refresh_spends = total_spends.copy()
        refresh_means = mean_spends.copy()
        refresh_shares = spend_shares.copy()

        if self.mmm_data.mmmdata_spec.refresh_counter > 0:
            # Get refresh window data
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

    def _geometric_adstock(self, x: pd.Series, theta: float) -> pd.Series:

        x_array = x.values
        # Use lfilter to efficiently compute the geometric transformation
        y = lfilter([1], [1, -theta], x_array)
        return pd.Series(y, index=x.index)

    def _hill_transformation(
        self, x: pd.Series, alpha: float, gamma: float
    ) -> pd.Series:
        x_scaled = (x - x.min()) / (x.max() - x.min())
        result = x_scaled**alpha / (x_scaled**alpha + gamma**alpha)
        return result
