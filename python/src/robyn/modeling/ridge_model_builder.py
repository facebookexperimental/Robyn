# pyre-strict

import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import nevergrad as ng
from tqdm import tqdm
from robyn.calibration.media_effect_calibration import MediaEffectCalibrator
import logging
import time
from datetime import datetime
from robyn.modeling.convergence.convergence import Convergence
from sklearn.exceptions import ConvergenceWarning
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.model_refit_output import ModelRefitOutput
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm


class RidgeModelBuilder:
    def __init__(
        self,
        mmm_data: MMMData,
        holiday_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
    ):
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.calibration_input = calibration_input
        self.hyperparameters = hyperparameters
        self.featurized_mmm_data = featurized_mmm_data
        self.logger = logging.getLogger(__name__)

    def build_models(
        self,
        trials_config: TrialsConfig,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        seed: List[int] = [123],  # Ensure seed is a list
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[List[float]] = None,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        cores: Optional[int] = None,
    ) -> ModelOutputs:
        start_time = time.time()
        # Initialize hyperparameters with flattened structure
        hyper_collect = self._hyper_collector(
            self.hyperparameters,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )
        # Convert datetime to string format matching R's format
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Set up objective weights including calibration if available
        if objective_weights is None:
            if self.calibration_input is not None:
                objective_weights = [1 / 3, 1 / 3, 1 / 3]  # NRMSE, RSSD, MAPE
            else:
                objective_weights = [0.5, 0.5]  # NRMSE, RSSD only
        # Run trials
        trials = []
        for trial in range(1, trials_config.trials + 1):
            trial_result = self._run_nevergrad_optimization(
                hyper_collect=hyper_collect,
                iterations=trials_config.iterations,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                intercept=intercept,
                intercept_sign=intercept_sign,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                objective_weights=objective_weights,
                dt_hyper_fixed=dt_hyper_fixed,
                rssd_zero_penalty=rssd_zero_penalty,
                trial=trial,
                seed=seed[0] + trial,  # Use the first element of the seed list
                total_trials=trials_config.trials,
            )
            trials.append(trial_result)
        # Calculate convergence
        convergence = Convergence()
        convergence_results = convergence.calculate_convergence(trials)
        # Aggregate results
        all_result_hyp_param = pd.concat(
            [trial.result_hyp_param for trial in trials], ignore_index=True
        )
        all_x_decomp_agg = pd.concat(
            [trial.x_decomp_agg for trial in trials], ignore_index=True
        )
        all_decomp_spend_dist = pd.concat(
            [
                trial.decomp_spend_dist
                for trial in trials
                if trial.decomp_spend_dist is not None
            ],
            ignore_index=True,
        )
        # Create ModelOutputs with flattened hyperparameter structure
        model_outputs = ModelOutputs(
            trials=trials,
            train_timestamp=current_time,
            cores=cores,
            iterations=trials_config.iterations,
            intercept=intercept,
            intercept_sign=intercept_sign,
            nevergrad_algo=nevergrad_algo,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            hyper_updated=hyper_collect["hyper_list_all"],  # Using flattened structure
            hyper_fixed=hyper_collect["all_fixed"],
            convergence=convergence_results,
            select_id=self._select_best_model(trials),
            seed=seed,  # Store seed as a list
            hyper_bound_ng=hyper_collect["hyper_bound_list_updated"],
            hyper_bound_fixed=hyper_collect["hyper_bound_list_fixed"],
            ts_validation_plot=None,
            all_result_hyp_param=all_result_hyp_param,
            all_x_decomp_agg=all_x_decomp_agg,
            all_decomp_spend_dist=all_decomp_spend_dist,
        )
        return model_outputs

    def _select_best_model(self, output_models: List[Trial]) -> str:
        # Extract relevant metrics
        nrmse_values = np.array([trial.nrmse for trial in output_models])
        decomp_rssd_values = np.array([trial.decomp_rssd for trial in output_models])

        # Normalize the metrics
        nrmse_norm = (nrmse_values - np.min(nrmse_values)) / (
            np.max(nrmse_values) - np.min(nrmse_values)
        )
        decomp_rssd_norm = (decomp_rssd_values - np.min(decomp_rssd_values)) / (
            np.max(decomp_rssd_values) - np.min(decomp_rssd_values)
        )

        # Calculate the combined score (assuming equal weights)
        combined_score = nrmse_norm + decomp_rssd_norm

        # Find the index of the best model (lowest combined score)
        best_index = np.argmin(combined_score)

        # Return the sol_id of the best model (changed from solID)
        return output_models[best_index].result_hyp_param["sol_id"].values[0]

    def _model_train(
        self,
        hyper_collect: Dict[str, Any],
        trials_config: TrialsConfig,
        intercept_sign: str,
        intercept: bool,
        nevergrad_algo: NevergradAlgorithm,
        dt_hyper_fixed: Optional[pd.DataFrame],
        ts_validation: bool,
        add_penalty_factor: bool,
        objective_weights: Optional[List[float]],
        rssd_zero_penalty: bool,
        seed: int,
        cores: int,
    ) -> List[Trial]:
        trials = []
        for trial in range(1, trials_config.trials + 1):
            trial_result = self._run_nevergrad_optimization(
                hyper_collect,
                trials_config.iterations,
                cores,
                nevergrad_algo,
                intercept,
                intercept_sign,
                ts_validation,
                add_penalty_factor,
                objective_weights,
                dt_hyper_fixed,
                rssd_zero_penalty,
                trial,
                seed + trial,
                trials_config.trials,
            )

            trials.append(trial_result)
        return trials

    def _run_nevergrad_optimization(
        self,
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
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        np.random.seed(seed)

        param_names = list(hyper_collect["hyper_bound_list_updated"].keys())
        param_bounds = [
            hyper_collect["hyper_bound_list_updated"][name] for name in param_names
        ]

        instrum_dict = {
            name: ng.p.Scalar(lower=bound[0], upper=bound[1])
            for name, bound in zip(param_names, param_bounds)
        }

        instrum = ng.p.Instrumentation(**instrum_dict)
        optimizer = ng.optimizers.registry[nevergrad_algo.value](
            instrum, budget=iterations, num_workers=cores
        )

        all_results = []
        start_time = time.time()

        with tqdm(
            total=iterations,
            desc=f"Running trial {trial} of total {total_trials} trials",
            bar_format="{l_bar}{bar}",
            ncols=75,
        ) as pbar:
            for iter_ng in range(iterations):
                candidate = optimizer.ask()
                params = candidate.kwargs

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self._evaluate_model(
                        params,
                        ts_validation,
                        add_penalty_factor,
                        rssd_zero_penalty,
                        objective_weights,
                        start_time=start_time,
                        iter_ng=iter_ng,
                        trial=trial,
                    )

                optimizer.tell(candidate, result["loss"])

                # Update result params with all metrics, using the new 'lambda' name
                result["params"].update(
                    {
                        "sol_id": f"{trial}_{iter_ng + 1}_1",  # Changed from solID to sol_id
                        "ElapsedAccum": result["elapsed_accum"],
                        "trial": trial,
                        "rsq_train": result["rsq_train"],
                        "rsq_val": result["rsq_val"],
                        "rsq_test": result["rsq_test"],
                        "nrmse": result["nrmse"],
                        "nrmse_train": result["nrmse_train"],
                        "nrmse_val": result["nrmse_val"],
                        "nrmse_test": result["nrmse_test"],
                        "decomp.rssd": result["decomp_rssd"],
                        "mape": result["mape"],
                        "lambda": result["lambda"],  # Changed from lambda_ to lambda
                        "lambda_hp": result["lambda_hp"],
                        "lambda_max": result["lambda_max"],
                        "lambda_min_ratio": result["lambda_min_ratio"],
                        "iterNG": iter_ng + 1,
                        "iterPar": 1,
                    }
                )

                all_results.append(result)
                pbar.update(1)

        end_time = time.time()
        self.logger.info(f" Finished in {(end_time - start_time) / 60:.2f} mins")

        # Aggregate results
        result_hyp_param = pd.DataFrame([r["params"] for r in all_results])
        decomp_spend_dist = pd.concat(
            [r["decomp_spend_dist"] for r in all_results], ignore_index=True
        )
        x_decomp_agg = pd.concat(
            [r["x_decomp_agg"] for r in all_results], ignore_index=True
        )

        # Find best result based on loss
        best_result = min(all_results, key=lambda x: x["loss"])

        return Trial(
            result_hyp_param=result_hyp_param,
            lift_calibration=best_result.get("lift_calibration", pd.DataFrame()),
            decomp_spend_dist=decomp_spend_dist,
            x_decomp_agg=x_decomp_agg,
            nrmse=best_result["nrmse"],
            decomp_rssd=best_result["decomp_rssd"],
            mape=best_result["mape"],
            rsq_train=best_result["rsq_train"],
            rsq_val=best_result["rsq_val"],
            rsq_test=best_result["rsq_test"],
            lambda_=best_result["lambda"],  # Changed from lambda_ to lambda
            lambda_hp=best_result["lambda_hp"],
            lambda_max=best_result["lambda_max"],
            lambda_min_ratio=best_result["lambda_min_ratio"],
            pos=best_result.get("pos", 0),
            elapsed=best_result["elapsed"],
            elapsed_accum=best_result["elapsed_accum"],
            trial=trial,
            iter_ng=best_result["iter_ng"],
            iter_par=best_result["iter_par"],
            train_size=best_result["params"].get("train_size", 1.0),
            sol_id=best_result["params"]["sol_id"],  # Changed from solID to sol_id
        )

    def _calculate_decomp_spend_dist(
        self, model: Ridge, X: pd.DataFrame, y: pd.Series, metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate decomposition spend distribution matching R's implementation exactly"""
        paid_media_cols = [
            col
            for col in X.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]

        results = []
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            effect = coef * X[col].sum()
            spend = X[col].sum()
            non_zero_effect = X[col][X[col] > 0] * coef

            result = {
                "rn": col,
                "coef": coef,
                "xDecompAgg": effect,
                "total_spend": spend,
                "mean_spend": X[col].mean(),
                "spend_share": spend / X[paid_media_cols].sum().sum(),
                "effect_share": effect
                / sum([coef * X[c].sum() for c in paid_media_cols]),
                "xDecompPerc": effect
                / sum([coef * X[c].sum() for c in paid_media_cols]),
                "xDecompMeanNon0": (
                    non_zero_effect.mean() if len(non_zero_effect) > 0 else 0
                ),
                "xDecompMeanNon0Perc": (
                    non_zero_effect.mean() if len(non_zero_effect) > 0 else 0
                )
                / sum(
                    [
                        coef * X[c][X[c] > 0].mean() if len(X[c][X[c] > 0]) > 0 else 0
                        for c in paid_media_cols
                    ]
                ),
                "xDecompAggRF": effect,
                "xDecompPercRF": effect
                / sum([coef * X[c].sum() for c in paid_media_cols]),
                "xDecompMeanNon0RF": (
                    non_zero_effect.mean() if len(non_zero_effect) > 0 else 0
                ),
                "xDecompMeanNon0PercRF": (
                    non_zero_effect.mean() if len(non_zero_effect) > 0 else 0
                )
                / sum(
                    [
                        coef * X[c][X[c] > 0].mean() if len(X[c][X[c] > 0]) > 0 else 0
                        for c in paid_media_cols
                    ]
                ),
                "pos": coef >= 0,
                "spend_share_refresh": spend / X[paid_media_cols].sum().sum(),
                "effect_share_refresh": effect
                / sum([coef * X[c].sum() for c in paid_media_cols]),
                "roi_total": effect / spend if spend > 0 else 0,
                "cpa_total": spend / effect if effect > 0 else 0,
            }

            # Add model performance metrics
            result.update(
                {
                    "rsq_train": metrics.get("rsq_train", 0),
                    "rsq_val": metrics.get("rsq_val", 0),
                    "rsq_test": metrics.get("rsq_test", 0),
                    "nrmse_train": metrics.get("nrmse_train", 0),
                    "nrmse_val": metrics.get("nrmse_val", 0),
                    "nrmse_test": metrics.get("nrmse_test", 0),
                    "nrmse": metrics.get("nrmse", 0),
                    "decomp.rssd": metrics.get("decomp_rssd", 0),
                    "mape": metrics.get("mape", 0),
                    "lambda": metrics.get("lambda_", 0),
                    "lambda_hp": metrics.get("lambda_hp", 0),
                    "lambda_max": metrics.get("lambda_max", 0),
                    "lambda_min_ratio": metrics.get("lambda_min_ratio", 0),
                    "sol_id": metrics.get("solID", ""),
                    "trial": metrics.get("trial", 0),
                    "iterNG": metrics.get("iterNG", 0),
                    "iterPar": metrics.get("iterPar", 0),
                    "Elapsed": metrics.get("elapsed", 0),
                }
            )

            results.append(result)

        return pd.DataFrame(results)

    def _prepare_data(self, params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with R-style scaling"""
        # Get the dependent variable
        if "dep_var" in self.featurized_mmm_data.dt_mod.columns:
            self.featurized_mmm_data.dt_mod = self.featurized_mmm_data.dt_mod.rename(
                columns={"dep_var": self.mmm_data.mmmdata_spec.dep_var}
            )
        y = self.featurized_mmm_data.dt_mod[self.mmm_data.mmmdata_spec.dep_var]

        # Select all columns except the dependent variable
        X = self.featurized_mmm_data.dt_mod.drop(
            columns=[self.mmm_data.mmmdata_spec.dep_var]
        )

        # Add debug self.logger.debugs
        self.logger.debug("Before scaling:")
        self.logger.debug("X head 5:", X.head())
        self.logger.debug("y head 5:", y.head())

        # Convert date columns to numeric
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

        # Add small random noise to prevent perfect collinearity
        X = X + 1e-8 * np.random.randn(*X.shape)

        return X, y

    def _geometric_adstock(self, x: pd.Series, theta: float) -> pd.Series:
        # Add debug self.logger.debugs
        self.logger.debug(f"Before adstock: {x.head()}")
        y = x.copy()
        for i in range(1, len(x)):
            y.iloc[i] += theta * y.iloc[i - 1]
        self.logger.debug(f"After adstock: {y.head()}")
        return y

    def _hill_transformation(
        self, x: pd.Series, alpha: float, gamma: float
    ) -> pd.Series:
        # Add debug self.logger.debugs
        self.logger.debug(f"Before hill: {x.head()}")
        x_scaled = (x - x.min()) / (x.max() - x.min())
        result = x_scaled**alpha / (x_scaled**alpha + gamma**alpha)
        self.logger.debug(f"After hill: {result.head()}")
        return result

    def _calculate_rssd(
        self,
        model: Ridge,
        X: pd.DataFrame,
        paid_media_cols: List[str],
        rssd_zero_penalty: bool,
    ) -> float:
        """Calculate RSSD exactly like R's glmnet implementation"""

        # Get coefficients and effects for media variables
        effects = []
        spends = []

        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            effect = coef * X[col].sum()  # Total effect
            spend = X[col].sum()  # Total spend

            effects.append(effect)
            spends.append(spend)

        effects = np.array(effects)
        spends = np.array(spends)

        # R's normalization approach
        effects_norm = effects / np.sum(np.abs(effects))
        spends_norm = spends / np.sum(spends)

        # Calculate RSSD
        rssd = np.sqrt(np.mean((effects_norm - spends_norm) ** 2))

        # Apply zero penalty like R
        if rssd_zero_penalty:
            zero_coef_count = np.sum(np.abs(effects) < 1e-10)
            rssd *= 1 + zero_coef_count / len(effects)

        return rssd

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
        # Calculate decomposition effects
        x_decomp = X * model.coef_

        results = []
        for col in X.columns:
            coef = model.coef_[list(X.columns).index(col)]
            decomp_values = x_decomp[col]

            # Calculate all required metrics
            result = {
                "rn": col,
                "coef": coef,
                "xDecompAgg": decomp_values.sum(),
                "xDecompPerc": (
                    decomp_values.sum() / x_decomp.sum().sum()
                    if x_decomp.sum().sum() != 0
                    else 0
                ),
                "xDecompMeanNon0": (
                    decomp_values[decomp_values > 0].mean()
                    if any(decomp_values > 0)
                    else 0
                ),
                "xDecompMeanNon0Perc": (
                    decomp_values[decomp_values > 0].mean()
                    / sum(
                        [
                            (
                                x_decomp[c][x_decomp[c] > 0].mean()
                                if any(x_decomp[c] > 0)
                                else 0
                            )
                            for c in X.columns
                        ]
                    )
                    if any(decomp_values > 0)
                    else 0
                ),
                # RF (refresh) versions
                "xDecompAggRF": decomp_values.sum(),
                "xDecompPercRF": (
                    decomp_values.sum() / x_decomp.sum().sum()
                    if x_decomp.sum().sum() != 0
                    else 0
                ),
                "xDecompMeanNon0RF": (
                    decomp_values[decomp_values > 0].mean()
                    if any(decomp_values > 0)
                    else 0
                ),
                "xDecompMeanNon0PercRF": (
                    decomp_values[decomp_values > 0].mean()
                    / sum(
                        [
                            (
                                x_decomp[c][x_decomp[c] > 0].mean()
                                if any(x_decomp[c] > 0)
                                else 0
                            )
                            for c in X.columns
                        ]
                    )
                    if any(decomp_values > 0)
                    else 0
                ),
                "pos": coef >= 0,
            }

            # Add model performance metrics
            result.update(
                {
                    "train_size": metrics.get("train_size", 1.0),
                    "rsq_train": metrics.get("rsq_train", 0),
                    "rsq_val": metrics.get("rsq_val", 0),
                    "rsq_test": metrics.get("rsq_test", 0),
                    "nrmse_train": metrics.get("nrmse_train", 0),
                    "nrmse_val": metrics.get("nrmse_val", 0),
                    "nrmse_test": metrics.get("nrmse_test", 0),
                    "nrmse": metrics.get("nrmse", 0),
                    "decomp.rssd": metrics.get("decomp_rssd", 0),
                    "mape": metrics.get("mape", 0),
                    "lambda": metrics.get(
                        "lambda", 0
                    ),  # Using lambda instead of lambda_
                    "lambda_hp": metrics.get("lambda_hp", 0),
                    "lambda_max": metrics.get("lambda_max", 0),
                    "lambda_min_ratio": metrics.get("lambda_min_ratio", 0),
                    "sol_id": metrics.get(
                        "sol_id", ""
                    ),  # Using sol_id instead of solID
                    "trial": metrics.get("trial", 0),
                    "iterNG": metrics.get("iterNG", 0),
                    "iterPar": metrics.get("iterPar", 0),
                    "Elapsed": metrics.get("Elapsed", 0),
                }
            )

            results.append(result)

        return pd.DataFrame(results)

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

    def _evaluate_model(
        self,
        params: Dict[str, float],
        ts_validation: bool,
        add_penalty_factor: bool,
        rssd_zero_penalty: bool,
        objective_weights: Optional[List[float]],
        start_time: float,
        iter_ng: int,
        trial: int,
    ) -> Dict[str, Any]:
        """Evaluate model with parameter set matching R's implementation exactly"""
        X, y = self._prepare_data(params)
        sol_id = f"{trial}_{iter_ng + 1}_1"

        # R-style standardization function
        def r_style_standardize(x):
            if isinstance(x, pd.Series):
                mean = x.mean()
                sd = np.sqrt(np.sum((x - mean) ** 2) / len(x))
                return (x - mean) / (sd if sd > 0 else 1)
            else:  # For numpy arrays
                mean = np.mean(x)
                sd = np.sqrt(np.sum((x - mean) ** 2) / len(x))
                return (x - mean) / (sd if sd > 0 else 1)

        # Scale features R-style
        X_scaled = X.apply(r_style_standardize)
        y_scaled = r_style_standardize(y)

        # Split data using R's approach
        train_size = params.get("train_size", 1.0) if ts_validation else 1.0
        train_idx = int(len(X) * train_size)

        metrics = {}
        if ts_validation:
            val_test_size = (len(X) - train_idx) // 2
            X_train = X_scaled.iloc[:train_idx]
            y_train = y_scaled.iloc[:train_idx]
            X_val = X_scaled.iloc[train_idx : train_idx + val_test_size]
            y_val = y_scaled.iloc[train_idx : train_idx + val_test_size]
            X_test = X_scaled.iloc[train_idx + val_test_size :]
            y_test = y_scaled.iloc[train_idx + val_test_size :]
        else:
            X_train, y_train = X_scaled, y_scaled
            X_val = X_test = y_val = y_test = None

        # Calculate lambda parameters matching R's glmnet
        lambda_max = np.max(np.abs(X_train.to_numpy().T @ y_train.to_numpy())) / len(
            y_train
        )
        lambda_min_ratio = 0.0001
        lambda_hp = params.get("lambda", 1.0)
        lambda_ = lambda_max * lambda_min_ratio + lambda_hp * (
            lambda_max - lambda_max * lambda_min_ratio
        )

        # Add debug self.logger.debugs
        self.logger.debug(f"lambda_max calculation: {lambda_max}")
        self.logger.debug(f"lambda_min_ratio: {lambda_min_ratio}")
        self.logger.debug(f"lambda_hp: {params['lambda']}")
        self.logger.debug(f"final lambda_: {lambda_}")

        # Fit model
        model = Ridge(alpha=lambda_, fit_intercept=True)
        model.fit(X_train, y_train)

        # Scale coefficients back to original scale
        y_std = np.sqrt(np.sum((y - y.mean()) ** 2) / len(y))
        X_std = np.sqrt(np.sum((X - X.mean()) ** 2) / len(X))
        model.coef_ = model.coef_ * (y_std / X_std)

        # Training metrics
        y_train_pred = model.predict(X_train)
        rss_train = np.sum((y_train - y_train_pred) ** 2)
        tss_train = np.sum((y_train - np.mean(y_train)) ** 2)
        metrics["rsq_train"] = 1 - rss_train / tss_train
        metrics["nrmse_train"] = np.sqrt(rss_train / len(y_train)) / (
            y_train.max() - y_train.min()
        )

        # Validation and test metrics
        if ts_validation and X_val is not None and X_test is not None:
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            rss_val = np.sum((y_val - y_val_pred) ** 2)
            tss_val = np.sum((y_val - np.mean(y_val)) ** 2)
            metrics["rsq_val"] = 1 - rss_val / tss_val
            metrics["nrmse_val"] = np.sqrt(rss_val / len(y_val)) / (
                y_val.max() - y_val.min()
            )

            rss_test = np.sum((y_test - y_test_pred) ** 2)
            tss_test = np.sum((y_test - np.mean(y_test)) ** 2)
            metrics["rsq_test"] = 1 - rss_test / tss_test
            metrics["nrmse_test"] = np.sqrt(rss_test / len(y_test)) / (
                y_test.max() - y_test.min()
            )

            metrics["nrmse"] = metrics["nrmse_val"]
        else:
            metrics["rsq_val"] = metrics["rsq_test"] = 0.0
            metrics["nrmse_val"] = metrics["nrmse_test"] = 0.0
            metrics["nrmse"] = metrics["nrmse_train"]

        # Calculate effects and spends for each media variable
        paid_media_cols = [
            col
            for col in X.columns
            if col in self.mmm_data.mmmdata_spec.paid_media_spends
        ]
        media_effects = {}
        media_spends = {}

        # Check positivity of coefficients
        pos = all(
            model.coef_[list(X_train.columns).index(col)] >= 0
            for col in paid_media_cols
        )

        for col in paid_media_cols:
            idx = list(X_train.columns).index(col)
            coef = model.coef_[idx]
            effect = coef * X[col].sum()  # Use unscaled X for effect calculation
            spend = X[col].sum()
            media_effects[col] = effect
            media_spends[col] = spend

        # Calculate RSSD
        total_effect = sum(abs(e) for e in media_effects.values())
        total_spend = sum(media_spends.values())

        effects_norm = {k: v / total_effect for k, v in media_effects.items()}
        spends_norm = {k: v / total_spend for k, v in media_spends.items()}

        effect_spend_diff = [
            effects_norm[col] - spends_norm[col] for col in paid_media_cols
        ]
        decomp_rssd = np.sqrt(np.mean(np.array(effect_spend_diff) ** 2))

        if rssd_zero_penalty:
            zero_effects = sum(1 for e in media_effects.values() if abs(e) < 1e-10)
            decomp_rssd *= 1 + zero_effects / len(media_effects)

        elapsed_time = time.time() - start_time

        # Format hyperparameter names and update metrics
        params_formatted = self._format_hyperparameter_names(params)
        metrics.update(
            {
                "decomp_rssd": decomp_rssd,
                "lambda": lambda_,
                "lambda_hp": lambda_hp,
                "lambda_max": lambda_max,
                "lambda_min_ratio": lambda_min_ratio,
                "mape": 0.0,
                "sol_id": sol_id,
                "trial": trial,
                "iterNG": iter_ng + 1,
                "iterPar": 1,
                "Elapsed": elapsed_time,
                "elapsed": elapsed_time,
                "elapsed_accum": elapsed_time,
                "pos": pos,
            }
        )

        params_formatted.update(
            {
                "sol_id": sol_id,
                "Elapsed": elapsed_time,
                "ElapsedAccum": elapsed_time,
                "pos": pos,
            }
        )

        # Calculate decompositions
        x_decomp_agg = self._calculate_x_decomp_agg(
            model, X, y, {**params_formatted, **metrics}
        )
        decomp_spend_dist = self._calculate_decomp_spend_dist(
            model, X, y, {**metrics, "params": params_formatted}
        )

        # self.logger.debug coefficients for debugging
        self.logger.debug("Model coefficients:")
        self.logger.debug(model.coef_)

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
        return hyper_collect

    @staticmethod
    def _model_refit(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        lambda_: float = 1.0,
        lower_limits: Optional[List[float]] = None,
        upper_limits: Optional[List[float]] = None,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
    ) -> ModelRefitOutput:
        model = Ridge(alpha=lambda_, fit_intercept=intercept)
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val) if x_val is not None else None
        y_test_pred = model.predict(x_test) if x_test is not None else None

        rsq_train = r2_score(y_train, y_train_pred)
        rsq_val = r2_score(y_val, y_val_pred) if y_val is not None else None
        rsq_test = r2_score(y_test, y_test_pred) if y_test is not None else None

        nrmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2)) / (
            np.max(y_train) - np.min(y_train)
        )
        nrmse_val = (
            np.sqrt(np.mean((y_val - y_val_pred) ** 2))
            / (np.max(y_val) - np.min(y_val))
            if y_val is not None
            else None
        )
        nrmse_test = (
            np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            / (np.max(y_test) - np.min(y_test))
            if y_test is not None
            else None
        )

        return ModelRefitOutput(
            rsq_train=rsq_train,
            rsq_val=rsq_val,
            rsq_test=rsq_test,
            nrmse_train=nrmse_train,
            nrmse_val=nrmse_val,
            nrmse_test=nrmse_test,
            coefs=model.coef_,
            y_train_pred=y_train_pred,
            y_val_pred=y_val_pred,
            y_test_pred=y_test_pred,
            y_pred=(
                np.concatenate([y_train_pred, y_val_pred, y_test_pred])
                if y_val is not None and y_test is not None
                else y_train_pred
            ),
            mod=model,
            df_int=1 if intercept else 0,
        )

    def _lambda_seq(
        self, x: pd.DataFrame, y: pd.Series, seq_len: int = 100
    ) -> np.ndarray:
        """Calculate lambda sequence exactly matching R's glmnet implementation"""
        x_np = x.to_numpy()
        y_np = y.to_numpy()
        n = len(y_np)

        # R's stdization uses n (not n-1) in denominator
        def r_scale(x: np.ndarray) -> np.ndarray:
            mu = np.mean(x)
            sigma = np.sqrt(np.sum((x - mu) ** 2) / len(x))  # R-style sd
            return (x - mu) / (sigma if sigma > 0 else 1)

        # Scale x and y like R
        x_scaled = np.column_stack([r_scale(x_np[:, j]) for j in range(x_np.shape[1])])
        y_scaled = r_scale(y_np)

        # R's glmnet lambda calculation
        alpha = 0.001  # Ridge regression default
        dot_prod = np.abs(x_scaled.T @ y_scaled)
        lambda_max = np.max(dot_prod) / (alpha * n)
        lambda_min_ratio = 0.0001

        # Generate sequence with exact R spacing
        log_lambda = np.linspace(
            np.log(lambda_max), np.log(lambda_max * lambda_min_ratio), seq_len
        )
        return np.exp(log_lambda)

    def _calculate_r2_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int, df_int: int = 1
    ) -> float:
        """Calculate R-squared exactly matching R's implementation"""
        n = len(y_true)
        resid_ss = np.sum((y_true - y_pred) ** 2)
        total_ss = np.sum((y_true - np.mean(y_true)) ** 2)

        # R's implementation includes penalty for number of features
        r2 = 1 - (resid_ss / total_ss)

        # R's adjustment formula
        adj_r2 = 1 - ((1 - r2) * (n - df_int) / (n - n_features - df_int))

        # R handles negative R² differently
        if adj_r2 < 0:
            adj_r2 = 1 - (resid_ss / total_ss) * (n - 1) / (n - n_features - 1)

        return adj_r2
