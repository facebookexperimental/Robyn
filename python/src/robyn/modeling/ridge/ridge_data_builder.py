import logging
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np


class RidgeDataBuilder:
    def __init__(self, mmm_data, featurized_mmm_data):
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.logger = logging.getLogger(__name__)

    def _prepare_data(self, params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ridge regression, excluding date columns"""
        # print("\n=== Data Preparation Debug ===")
        # print("Initial dt_mod shape:", self.featurized_mmm_data.dt_mod.shape)
        # print("Initial dt_mod columns:", self.featurized_mmm_data.dt_mod.columns.tolist())
        # print("Initial dep_var:", self.mmm_data.mmmdata_spec.dep_var)
        
        # Get the dependent variable, handling both possible column names
        dep_var = self.mmm_data.mmmdata_spec.dep_var
        if dep_var not in self.featurized_mmm_data.dt_mod.columns:
            # If dep_var column doesn't exist, try 'dep_var'
            if 'dep_var' in self.featurized_mmm_data.dt_mod.columns:
                y = self.featurized_mmm_data.dt_mod['dep_var']
            else:
                raise KeyError(f"Could not find dependent variable column. Expected either '{dep_var}' or 'dep_var' in columns: {self.featurized_mmm_data.dt_mod.columns.tolist()}")
        else:
            y = self.featurized_mmm_data.dt_mod[dep_var]

        # Select all columns except the dependent variable and date columns
        exclude_cols = ['ds']  # Always exclude 'ds'
        if dep_var in self.featurized_mmm_data.dt_mod.columns:
            exclude_cols.append(dep_var)
        if 'dep_var' in self.featurized_mmm_data.dt_mod.columns:
            exclude_cols.append('dep_var')
        
        X = self.featurized_mmm_data.dt_mod.drop(columns=exclude_cols)

        # print("\nAfter initial selection:")
        # print("X shape:", X.shape)
        # print("X columns:", X.columns.tolist())
        # print("y shape:", y.shape)
        # print("First 3 rows of y:", y.head(3).tolist())

        # Handle any remaining categorical columns (if any)
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns
        # print("\nCategorical columns found:", categorical_columns.tolist())
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        # print("\nAfter transformations:")
        # print("X shape:", X.shape)
        # print("X columns:", X.columns.tolist())
        
        # Apply transformations based on hyperparameters
        for media in self.mmm_data.mmmdata_spec.paid_media_spends:
            if f"{media}_thetas" in params:
                # print(f"\nApplying geometric adstock to {media}")
                X[media] = self._geometric_adstock(X[media], params[f"{media}_thetas"])
            if f"{media}_alphas" in params and f"{media}_gammas" in params:
                # print(f"Applying hill transformation to {media}")
                X[media] = self._hill_transformation(
                    X[media], params[f"{media}_alphas"], params[f"{media}_gammas"]
                )

        # Handle any remaining NaN or infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
        X = X + 1e-8 * np.random.randn(*X.shape)

        # print("\nFinal data shapes:")
        # print("X shape:", X.shape)
        # print("y shape:", y.shape)
        # print("===========================\n")

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
        # Remove train_size from optimization
        if ts_validation:
            # Use fixed train_size (0.7 by default)
            train_size = 0.7  # or get from hyperparameters input
            hyper_collect["hyper_list_all"]["train_size"] = [train_size, train_size]
            hyper_collect["hyper_bound_list_fixed"]["train_size"] = train_size
        else:
            hyper_collect["hyper_list_all"]["train_size"] = [1.0, 1.0]
            hyper_collect["hyper_bound_list_fixed"]["train_size"] = 1.0
        return hyper_collect

    def _geometric_adstock(self, x: pd.Series, theta: float) -> pd.Series:
        # print(f"Before adstock: {x.head()}")
        y = x.copy()
        for i in range(1, len(x)):
            y.iloc[i] += theta * y.iloc[i - 1]
        # print(f"After adstock: {y.head()}")
        return y

    def _hill_transformation(
        self, x: pd.Series, alpha: float, gamma: float
    ) -> pd.Series:
        # Add debug self.logger.debugs
        # print(f"Before hill: {x.head()}")
        x_scaled = (x - x.min()) / (x.max() - x.min())
        result = x_scaled**alpha / (x_scaled**alpha + gamma**alpha)
        # print(f"After hill: {result.head()}")
        return result
