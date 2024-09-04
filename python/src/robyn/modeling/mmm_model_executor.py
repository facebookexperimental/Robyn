import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class MMMModelExecutor:
    def __init__(self):
        pass

    def model_run(
        self,
        mmmdata_collection,
        trials_config,
        seed: int = 123,
        quiet: bool = False,
        cores: Optional[int] = None,
        nevergrad_algo: str = "TwoPointsDE",
        intercept: bool = True,
        intercept_sign: str = "non_negative",
    ) -> Dict[str, Any]:
        t0 = time.time()

        # Print data summary
        if not quiet:
            self._print_data_summary(mmmdata_collection)

        # Initialize hyperparameters
        hyper_collect = self._hyper_collector(mmmdata_collection, trials_config)

        # Print hyperparameter summary
        if not quiet:
            self._print_hyper_summary(hyper_collect)

        # Run trials
        model_output_collection = self._run_trials(
            mmmdata_collection,
            trials_config,
            hyper_collect,
            seed,
            quiet,
            cores,
            nevergrad_algo,
            intercept,
            intercept_sign,
        )

        # Add metadata to model_output_collection
        model_output_collection["train_timestamp"] = time.time()
        model_output_collection["cores"] = cores
        model_output_collection["iterations"] = trials_config.num_iterations_per_trial
        model_output_collection["trials"] = trials_config.num_trials
        model_output_collection["intercept"] = intercept
        model_output_collection["intercept_sign"] = intercept_sign
        model_output_collection["nevergrad_algo"] = nevergrad_algo
        model_output_collection["ts_validation"] = trials_config.timeseries_validation
        model_output_collection["add_penalty_factor"] = trials_config.add_penalty_factor
        model_output_collection["hyper_updated"] = hyper_collect["hyper_list_all"]
        model_output_collection["hyper_fixed"] = hyper_collect["all_fixed"]

        # Check convergence
        if not hyper_collect["all_fixed"]:
            model_output_collection["convergence"] = self._check_convergence(
                model_output_collection
            )
            model_output_collection["ts_validation_plot"] = self._ts_validation(
                model_output_collection
            )

        # Report total timing
        total_run_time = (time.time() - t0) / 60
        if not quiet:
            print(f"Total run time: {total_run_time:.2f} mins")

        return model_output_collection

    def _print_data_summary(self, mmmdata_collection):
        print(
            f"Input data has {len(mmmdata_collection.dt_input)} periods in total: "
            f"{mmmdata_collection.dt_input.ds.min()} to {mmmdata_collection.dt_input.ds.max()}"
        )
        print(
            f"Model is built on rolling window of {mmmdata_collection.rollingWindowLength} periods: "
            f"{mmmdata_collection.window_start} to {mmmdata_collection.window_end}"
        )

    def _hyper_collector(self, mmmdata_collection, trials_config):
        hyper_params = mmmdata_collection.hyperparameters

        hyper_bound_list_updated = {}
        hyper_bound_list_fixed = {}

        for param, value in hyper_params.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                hyper_bound_list_updated[param] = value
            else:
                hyper_bound_list_fixed[param] = value

        all_fixed = len(hyper_bound_list_updated) == 0

        if trials_config.timeseries_validation:
            if "train_size" not in hyper_bound_list_updated:
                hyper_bound_list_updated["train_size"] = [0.5, 0.8]
        else:
            hyper_bound_list_fixed["train_size"] = 1.0

        return {
            "hyper_list_all": hyper_params,
            "hyper_bound_list_updated": hyper_bound_list_updated,
            "hyper_bound_list_fixed": hyper_bound_list_fixed,
            "all_fixed": all_fixed,
        }

    def _print_hyper_summary(self, hyper_collect):
        total_hypers = len(hyper_collect["hyper_list_all"])
        fixed_hypers = len(hyper_collect["hyper_bound_list_fixed"])
        print(
            f"Using {total_hypers} hyperparameters "
            f"({total_hypers - fixed_hypers} to iterate + {fixed_hypers} fixed)"
        )

    def _run_trials(
        self,
        mmmdata_collection,
        trials_config,
        hyper_collect,
        seed,
        quiet,
        cores,
        nevergrad_algo,
        intercept,
        intercept_sign,
    ):
        import nevergrad as ng

        np.random.seed(seed)

        model_output_collection = {
            "trials": [],
            "resultHypParam": [],
            "xDecompAgg": [],
        }

        for trial in range(trials_config.num_trials):
            if not quiet:
                print(f"Running trial {trial + 1} of {trials_config.num_trials}")

            # Initialize Nevergrad optimizer
            param_dict = {
                name: ng.p.Scalar(lower=bounds[0], upper=bounds[1])
                for name, bounds in hyper_collect["hyper_bound_list_updated"].items()
            }
            instrum = ng.p.Instrumentation(**param_dict)
            optimizer = ng.optimizers.registry[nevergrad_algo](
                instrum,
                budget=trials_config.num_iterations_per_trial,
                num_workers=cores or 1,
            )

            results = []
            for _ in tqdm(range(trials_config.num_iterations_per_trial), disable=quiet):
                hyper_params = optimizer.ask()
                model_result = self._run_single_model(
                    mmmdata_collection,
                    hyper_params.value,
                    hyper_collect,
                    intercept,
                    intercept_sign,
                    trials_config.timeseries_validation,
                    trials_config.add_penalty_factor,
                    trials_config.rssd_zero_penalty,
                    trial,
                    seed,
                )

                if model_result is not None:
                    optimizer.tell(
                        hyper_params,
                        (
                            model_result["nrmse"],
                            model_result["decomp_rssd"],
                            model_result.get("mape", 0),
                        ),
                    )
                    results.append(model_result)
                else:
                    optimizer.tell(
                        hyper_params, (float("inf"), float("inf"), float("inf"))
                    )

            if not results:
                raise ValueError(
                    "All model runs failed. Check the error messages above for details."
                )

            best_result = min(results, key=lambda x: x["nrmse"])
            model_output_collection["trials"].append(best_result)
            model_output_collection["resultHypParam"].append(best_result)
            model_output_collection["xDecompAgg"].append(best_result["xDecompAgg"])

        return model_output_collection

    def _run_single_model(
        self,
        mmmdata_collection,
        hyper_params,
        hyper_collect,
        intercept,
        intercept_sign,
        ts_validation,
        add_penalty_factor,
        rssd_zero_penalty,
        trial,
        seed,
    ):
        try:
            # Check and correct the structure of hyper_params
            print("Original hyper_params:", hyper_params)
            if isinstance(hyper_params, tuple):
                # Check if the first element is an empty tuple and the second is a dictionary
                if (
                    len(hyper_params) == 2
                    and not hyper_params[0]
                    and isinstance(hyper_params[1], dict)
                ):
                    hyper_params = hyper_params[1]  # Use the dictionary directly
                else:
                    raise ValueError("Unexpected structure of hyper_params")
            print("Corrected hyper_params:", hyper_params)

            # Ensure 'train_size' is in hyper_params
            if "train_size" not in hyper_params:
                hyper_params["train_size"] = hyper_collect[
                    "hyper_bound_list_fixed"
                ].get("train_size", 0.8)

            # Transform media based on hyperparameters
            transformed_data = self._run_transformations(
                mmmdata_collection, hyper_params
            )

            # Prepare data for modeling (split train/test/validation)
            train_data, val_data, test_data = self._prepare_data(
                transformed_data, hyper_params["train_size"], ts_validation
            )

            # Fit the model
            model, coefs = self._fit_model(
                train_data,
                val_data,
                test_data,
                intercept,
                intercept_sign,
                add_penalty_factor,
                hyper_params,
            )

            # Calculate performance metrics
            nrmse = self._calculate_nrmse(
                model, val_data if ts_validation else train_data
            )
            decomp_rssd = self._calculate_decomp_rssd(
                model, transformed_data, mmmdata_collection, rssd_zero_penalty
            )
            mape = (
                self._calculate_mape(model, mmmdata_collection)
                if mmmdata_collection.calibration_input
                else 0
            )

            # Perform model decomposition
            decomp_result = self._model_decomp(
                model, transformed_data, coefs, mmmdata_collection
            )

            # Prepare and return results
            result = {
                "nrmse": nrmse,
                "decomp_rssd": decomp_rssd,
                "mape": mape,
                "coefs": coefs,
                "xDecompAgg": decomp_result["xDecompAgg"],
            }
            result.update(hyper_params)  # Include hyperparameters in the result

            return result

        except Exception as e:
            print(f"Error in _run_single_model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            return None

    def _run_transformations(self, mmmdata_collection, hyper_params):
        transformed_data = mmmdata_collection.dt_mod.copy()
        print("hyperparams in run_transformations,", hyper_params)
        for media in mmmdata_collection.paid_media_spends:
            try:
                alpha = hyper_params[f"{media}_alphas"]
                gamma = hyper_params[f"{media}_gammas"]
                theta = hyper_params[f"{media}_thetas"]
            except KeyError as e:
                print(f"Missing key in hyper_params: {e}")
                raise
            adstocked = self._apply_adstock(transformed_data[media], theta)
            saturated = self._apply_saturation(adstocked, alpha, gamma)
            transformed_data[f"{media}_transformed"] = saturated
        return transformed_data

    def _apply_adstock(self, series, theta):
        return series.ewm(alpha=1 - theta, adjust=False).mean()

    def _apply_saturation(self, series, alpha, gamma):
        return (
            (1 - np.exp(-alpha * series / series.mean())) / (1 - np.exp(-alpha))
        ) * gamma

    def _prepare_data(self, data, train_size, ts_validation):
        # Debugging: Print original shape
        print("Original data shape:", data.shape)
        # Identify categorical columns
        categorical_cols = [col for col in data.columns if data[col].dtype == "object"]
        print("Categorical columns:", categorical_cols)
        # Apply one-hot encoding
        data = pd.get_dummies(data, columns=categorical_cols)
        print("Data shape after encoding:", data.shape)
        # Drop rows with NaN values
        data = data.dropna()
        print("Data shape after dropping NaNs:", data.shape)
        if ts_validation:
            train_end = int(len(data) * train_size)
            val_end = train_end + int(len(data) * (1 - train_size) / 2)
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            test_data = data.iloc[val_end:]
        else:
            train_data = data
            val_data = test_data = None
        # Debugging: Print final shapes
        print("Train data shape:", train_data.shape)
        print(
            "Validation data shape:", val_data.shape if val_data is not None else "N/A"
        )
        print("Test data shape:", test_data.shape if test_data is not None else "N/A")
        return train_data, val_data, test_data

    def _fit_model(
        self,
        train_data,
        val_data,
        test_data,
        intercept,
        intercept_sign,
        add_penalty_factor,
        hyper_params,
    ):
        X = train_data.drop(["ds", "dep_var"], axis=1)
        y = train_data["dep_var"]

        model = Ridge(alpha=hyper_params.get("lambda", 1.0), fit_intercept=intercept)
        model.fit(X, y)

        coefs = model.coef_
        if intercept:
            coefs = np.insert(coefs, 0, model.intercept_)

        return model, coefs

    def _calculate_nrmse(self, model, data):
        X = data.drop(["ds", "dep_var"], axis=1)
        y_true = data["dep_var"]
        y_pred = model.predict(X)

        mse = mean_squared_error(y_true, y_pred)
        nrmse = np.sqrt(mse) / (y_true.max() - y_true.min())

        return nrmse

    def _calculate_decomp_rssd(
        self, model, data, mmmdata_collection, rssd_zero_penalty
    ):
        X = data.drop(["ds", "dep_var"], axis=1)
        print("Shape of X:", X.shape)
        print("Length of model.coef_:", len(model.coef_))
        # If the model includes an intercept, adjust the coefficients array
        if model.fit_intercept:
            # Concatenate the intercept to the coefficients
            coefficients = np.concatenate([[model.intercept_], model.coef_])
            # Add a column of ones to X for the intercept
            X = np.hstack([np.ones((X.shape[0], 1)), X.values])
        else:
            coefficients = model.coef_
        print("Shape of coefficients:", coefficients.shape)
        # Ensure the dimensions match
        if X.shape[1] != len(coefficients):
            raise ValueError("Mismatch in the number of features and coefficients.")
        # Calculate the decomposition
        decomp = (X * coefficients).sum(axis=0)
        total_effect = decomp.sum()
        effect_share = decomp / total_effect
        spend_share = (
            data[mmmdata_collection.paid_media_spends].sum()
            / data[mmmdata_collection.paid_media_spends].sum().sum()
        )
        rssd = np.sqrt(((effect_share - spend_share) ** 2).sum())
        if rssd_zero_penalty:
            zero_coef_share = (coefficients == 0).mean()
            rssd *= 1 + zero_coef_share
        return rssd

    def _calculate_mape(self, model, mmmdata_collection):
        # Placeholder for MAPE calculation
        return 0

    def _model_decomp(self, model, data, coefs, mmmdata_collection):
        X = data.drop(["ds", "dep_var"], axis=1)
        decomp = (X * coefs[1:]).sum(axis=0)  # Exclude intercept
        total_effect = decomp.sum()

        decomp_agg = pd.DataFrame(
            {
                "variable": X.columns,
                "coef": coefs[1:],
                "effect": decomp,
                "effect_share": decomp / total_effect,
            }
        )

        return {"xDecompAgg": decomp_agg}

    def _check_convergence(self, model_output_collection):
        # Placeholder for convergence check
        return {"converged": False, "message": "Convergence check not implemented"}

    def _ts_validation(self, model_output_collection):
        # Placeholder for time series validation
        return None
