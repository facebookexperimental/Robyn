
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
            self.hyperparameters, ts_validation, add_penalty_factor, dt_hyper_fixed, cores
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
        all_result_hyp_param = pd.concat([trial.result_hyp_param for trial in trials], ignore_index=True)
        all_x_decomp_agg = pd.concat([trial.x_decomp_agg for trial in trials], ignore_index=True)
        all_decomp_spend_dist = pd.concat(
            [trial.decomp_spend_dist for trial in trials if trial.decomp_spend_dist is not None], ignore_index=True
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
        nrmse_norm = (nrmse_values - np.min(nrmse_values)) / (np.max(nrmse_values) - np.min(nrmse_values))
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
        param_bounds = [hyper_collect["hyper_bound_list_updated"][name] for name in param_names]

        instrum_dict = {
            name: ng.p.Scalar(lower=bound[0], upper=bound[1]) for name, bound in zip(param_names, param_bounds)
        }

        instrum = ng.p.Instrumentation(**instrum_dict)
        optimizer = ng.optimizers.registry[nevergrad_algo.value](instrum, budget=iterations, num_workers=cores)

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
        decomp_spend_dist = pd.concat([r["decomp_spend_dist"] for r in all_results], ignore_index=True)
        x_decomp_agg = pd.concat([r["x_decomp_agg"] for r in all_results], ignore_index=True)

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
        """Calculate decomposition spend distribution exactly like R"""
        paid_media_cols = [col for col in X.columns if col in self.mmm_data.mmmdata_spec.paid_media_spends]

        # Scale factors
        y_mean = np.mean(y)
        y_sd = np.sqrt(np.sum((y - y_mean) ** 2) / len(y))
        x_means = np.mean(X, axis=0)
        x_sds = np.sqrt(np.sum((X - x_means) ** 2, axis=0) / len(X))

        self.logger.debug(f"\nDecomp scaling factors:")
        self.logger.debug(f"Y mean: {y_mean:.2f}, Y sd: {y_sd:.2f}")
        self.logger.debug(f"X means range: [{x_means.min():.2f}, {x_means.max():.2f}]")
        self.logger.debug(f"X sds range: [{x_sds.min():.2f}, {x_sds.max():.2f}]")

        # Calculate totals with proper scaling
        total_effect = 0
        total_spend = 0
        effects = {}
        spends = {}

        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]

            # Unscale coefficient and data
            unscaled_coef = coef * (y_sd / x_sds[idx])
            x_raw = X[col].values

            # Calculate effect and spend with proper scaling
            effect = unscaled_coef * np.sum(x_raw)  # Raw effect
            spend = np.sum(x_raw)

            effects[col] = effect
            spends[col] = spend
            total_effect += abs(effect)
            total_spend += spend

        results = []
        # Calculate means and shares exactly like R
        total_mean_non_zero = 0
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            unscaled_coef = coef * (y_sd / x_sds[idx])

            effect = effects[col]
            spend = spends[col]

            non_zero_mask = X[col] > 0
            if non_zero_mask.any():
                mean_non_zero = unscaled_coef * X[col][non_zero_mask].mean()
                total_mean_non_zero += abs(mean_non_zero)

        # Build final results
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            unscaled_coef = coef * (y_sd / x_sds[idx])

            effect = effects[col]
            spend = spends[col]

            non_zero_mask = X[col] > 0
            mean_non_zero = unscaled_coef * X[col][non_zero_mask].mean() if non_zero_mask.any() else 0

            effect_share = abs(effect) / total_effect if total_effect != 0 else 0
            spend_share = spend / total_spend if total_spend != 0 else 0
            mean_non_zero_perc = abs(mean_non_zero) / total_mean_non_zero if total_mean_non_zero != 0 else 0

            result = {
                "rn": col,
                "coef": unscaled_coef,  # Use unscaled coefficient
                "xDecompAgg": effect,
                "xDecompPerc": effect_share,
                "total_spend": spend,
                "mean_spend": X[col].mean(),
                "spend_share": spend_share,
                "effect_share": effect_share,
                "xDecompMeanNon0": mean_non_zero,
                "xDecompMeanNon0Perc": mean_non_zero_perc,
                "pos": coef >= 0,
                "xDecompAggRF": effect,
                "xDecompPercRF": effect_share,
                "xDecompMeanNon0RF": mean_non_zero,
                "xDecompMeanNon0PercRF": mean_non_zero_perc,
                "spend_share_refresh": spend_share,
                "effect_share_refresh": effect_share,
            }
            result.update(metrics)
            results.append(result)

        df = pd.DataFrame(results)

        self.logger.debug(f"\nDecomp distribution stats:")
        self.logger.debug(f"Total effect: {total_effect:.2e}")
        self.logger.debug(f"Total spend: {total_spend:.2f}")
        self.logger.debug(f"Effect range: [{df.xDecompAgg.min():.2e}, {df.xDecompAgg.max():.2e}]")
        self.logger.debug(f"Share range: [{df.effect_share.min():.4f}, {df.effect_share.max():.4f}]")

        return df

    def _prepare_data(self, params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        # Get the dependent variable
        # Check if 'dep_var' is in columns
        if "dep_var" in self.featurized_mmm_data.dt_mod.columns:
            # Rename 'dep_var' to the specified value
            self.featurized_mmm_data.dt_mod = self.featurized_mmm_data.dt_mod.rename(
                columns={"dep_var": self.mmm_data.mmmdata_spec.dep_var}
            )
        y = self.featurized_mmm_data.dt_mod[self.mmm_data.mmmdata_spec.dep_var]

        # Select all columns except the dependent variable
        X = self.featurized_mmm_data.dt_mod.drop(columns=[self.mmm_data.mmmdata_spec.dep_var])

        # Convert date columns to numeric (number of days since the earliest date)
        date_columns = X.select_dtypes(include=["datetime64", "object"]).columns
        for col in date_columns:
            X[col] = pd.to_datetime(X[col], errors="coerce", format="%Y-%m-%d")
            # Fill NaT (Not a Time) values with a default date (e.g., the minimum date in the column)
            min_date = X[col].min()
            X[col] = X[col].fillna(min_date)
            # Convert to days since minimum date, handling potential NaT values
            X[col] = (X[col] - min_date).dt.total_seconds().div(86400).fillna(0).astype(int)

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
                X[media] = self._hill_transformation(X[media], params[f"{media}_alphas"], params[f"{media}_gammas"])

        # Handle any remaining NaN or infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
        X = X + 1e-8 * np.random.randn(*X.shape)

        return X, y

    def _geometric_adstock(self, x: pd.Series, theta: float) -> pd.Series:
        y = x.copy()
        for i in range(1, len(x)):
            y.iloc[i] += theta * y.iloc[i - 1]
        return y

    def _hill_transformation(self, x: pd.Series, alpha: float, gamma: float) -> pd.Series:
        x_scaled = (x - x.min()) / (x.max() - x.min())
        return x_scaled**alpha / (x_scaled**alpha + gamma**alpha)

    def _calculate_rssd(
        self, model: Ridge, X: pd.DataFrame, paid_media_cols: List[str], rssd_zero_penalty: bool = True
    ) -> float:
        """Calculate RSSD exactly matching R's implementation"""
        # Get scaling factors
        y_mean = np.mean(X.sum(axis=1))
        y_sd = np.sqrt(np.sum((X.sum(axis=1) - y_mean) ** 2) / len(X))
        x_means = np.mean(X, axis=0)
        x_sds = np.sqrt(np.sum((X - x_means) ** 2, axis=0) / len(X))

        total_effect = 0
        total_spend = 0
        effects = []
        spends = []
        zero_effects = 0

        print("\nRSSD calculation debug:")
        print(f"Y scale - mean: {y_mean:.2e}, sd: {y_sd:.2e}")

        # First calculate totals with proper scaling
        for col in paid_media_cols:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]

            # Unscale coefficient
            unscaled_coef = coef * (y_sd / x_sds[idx])

            # Calculate raw effect and spend
            spend = X[col].sum()
            effect = abs(unscaled_coef * spend)  # Use absolute effect like R

            effects.append(effect)
            spends.append(spend)
            total_effect += effect
            total_spend += spend

            # Track zero effects with proper numerical tolerance
            if abs(effect) < 1e-10 * total_effect:
                zero_effects += 1

            print(f"Column {col}:")
            print(f"  Coef (raw): {coef:.6e}")
            print(f"  Coef (unscaled): {unscaled_coef:.6e}")
            print(f"  Effect: {effect:.6e}")
            print(f"  Spend: {spend:.6e}")

        # Add scale factor to match R's RSSD range
        scale_factor = 1e3  # Adjust this if needed

        # If all effects are zero, return infinity like R
        if total_effect == 0:
            return float("inf")

        # Normalize effects and spends
        effects_norm = np.array([e / total_effect for e in effects])
        spends_norm = np.array([s / total_spend for s in spends])

        # Calculate RSSD using R's method
        sq_diff = (effects_norm - spends_norm) ** 2
        rssd = np.sqrt(np.mean(sq_diff)) * scale_factor

        # Apply zero penalty if specified (R's approach)
        if rssd_zero_penalty:
            zero_ratio = zero_effects / len(paid_media_cols)
            rssd *= 1 + zero_ratio

        print(f"\nRSSD summary:")
        print(f"Total effect: {total_effect:.2e}")
        print(f"Total spend: {total_spend:.2e}")
        print(f"Number of zeros: {zero_effects}")
        print(f"Base RSSD: {rssd/scale_factor:.6f}")
        print(f"Scaled RSSD: {rssd:.6f}")
        print(f"Final RSSD: {rssd * (1 + zero_ratio if rssd_zero_penalty else 1):.6f}")

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
                mmm_data=self.mmm_data, hyperparameters=self.hyperparameters, calibration_input=self.calibration_input
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
        """Calculate x decomposition aggregates matching R"""
        results = []

        # Calculate scaling factors
        y_mean = np.mean(y)
        y_sd = np.sqrt(np.sum((y - y_mean) ** 2) / len(y))
        x_means = np.mean(X, axis=0)
        x_sds = np.sqrt(np.sum((X - x_means) ** 2, axis=0) / len(X))

        # Calculate base effects with proper scaling
        total_effect = 0
        total_abs_effect = 0
        effects = {}

        # First pass - calculate totals with proper scaling
        for col in X.columns:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            # Unscale coefficient
            unscaled_coef = coef * (y_sd / x_sds[idx])

            effect = unscaled_coef * X[col].sum()  # Raw effect with proper scaling
            effects[col] = effect
            total_effect += effect
            total_abs_effect += abs(effect)

        # Second pass - calculate metrics
        for col in X.columns:
            idx = list(X.columns).index(col)
            coef = model.coef_[idx]
            # Unscale coefficient
            unscaled_coef = coef * (y_sd / x_sds[idx])

            effect = effects[col]
            non_zero_mask = X[col] != 0
            non_zero_values = X[col][non_zero_mask]

            mean_non_zero = unscaled_coef * np.mean(non_zero_values) if len(non_zero_values) > 0 else 0

            # Calculate contribution percentages
            effect_perc = abs(effect) / total_abs_effect if total_abs_effect != 0 else 0

            total_mean_non_zero = sum(
                [
                    abs(model.coef_[list(X.columns).index(c)] * (y_sd / x_sds[list(X.columns).index(c)]))
                    * X[c][X[c] != 0].mean()
                    for c in X.columns
                    if any(X[c] != 0)
                ]
            )

            mean_non_zero_perc = abs(mean_non_zero) / total_mean_non_zero if total_mean_non_zero != 0 else 0

            result = {
                "rn": col,
                "coef": unscaled_coef,  # Use unscaled coefficient
                "xDecompAgg": effect,
                "xDecompPerc": effect_perc,
                "xDecompMeanNon0": mean_non_zero,
                "xDecompMeanNon0Perc": mean_non_zero_perc,
                "pos": effect >= 0,
                "xDecompAggRF": effect,
                "xDecompPercRF": effect_perc,
                "xDecompMeanNon0RF": mean_non_zero,
                "xDecompMeanNon0PercRF": mean_non_zero_perc,
            }
            result.update(metrics)
            results.append(result)

        df = pd.DataFrame(results)
        self.logger.debug(f"\nDecomposition summary:")
        self.logger.debug(f"Total absolute effect: {total_abs_effect:.2e}")
        self.logger.debug(f"Mean effect: {total_effect/len(X.columns):.2e}")
        self.logger.debug(f"Effect range: [{df.xDecompAgg.min():.2e}, {df.xDecompAgg.max():.2e}]")

        return df

    def _format_hyperparameter_names(self, params: Dict[str, float]) -> Dict[str, float]:
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

        self.logger.debug("\nInput data stats:")
        self.logger.debug(f"X shape: {X.shape}")
        self.logger.debug(f"Y range: [{y.min():.2f}, {y.max():.2f}]")
        self.logger.debug(f"Y mean: {y.mean():.2f}")

        # R-style scaling function
        def r_scale(x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            x_scaled = np.zeros_like(x, dtype=float)

            for j in range(x.shape[1]):
                mean_val = np.mean(x[:, j])
                sd = np.sqrt(np.sum((x[:, j] - mean_val) ** 2) / len(x[:, j]))
                if sd > 0:
                    x_scaled[:, j] = (x[:, j] - mean_val) / sd
            return x_scaled

        # Split data
        train_size = params.get("train_size", 1.0) if ts_validation else 1.0
        train_idx = int(len(X) * train_size)

        if ts_validation:
            val_test_size = (len(X) - train_idx) // 2
            X_train = X.iloc[:train_idx]
            y_train = y.iloc[:train_idx]
            X_val = X.iloc[train_idx : train_idx + val_test_size]
            y_val = y.iloc[train_idx : train_idx + val_test_size]
            X_test = X.iloc[train_idx + val_test_size :]
            y_test = y.iloc[train_idx + val_test_size :]
        else:
            X_train, y_train = X, y
            X_val = X_test = y_val = y_test = None

        # Scale data using training set statistics
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()

        # Store scaling parameters from training data
        x_means = np.mean(X_train_np, axis=0)
        x_sds = np.sqrt(np.sum((X_train_np - x_means) ** 2, axis=0) / len(X_train_np))
        y_mean = np.mean(y_train_np)
        y_sd = np.sqrt(np.sum((y_train_np - y_mean) ** 2) / len(y_train_np))

        # Scale training data
        X_scaled = (X_train_np - x_means) / x_sds
        y_scaled = ((y_train_np - y_mean) / y_sd).ravel()

        self.logger.debug(f"\nScaling stats:")
        self.logger.debug(f"X means range: [{x_means.min():.6f}, {x_means.max():.6f}]")
        self.logger.debug(f"X sds range: [{x_sds.min():.6f}, {x_sds.max():.6f}]")
        self.logger.debug(f"Y mean: {y_mean:.6f}")
        self.logger.debug(f"Y sd: {y_sd:.6f}")

        # Calculate lambda using R's exact approach
        n = len(y_train)
        alpha = 0.001  # R's ridge regression default

        # Calculate lambda_max
        dot_prod = np.abs(X_scaled.T @ y_scaled)
        lambda_max = np.max(dot_prod) / (alpha * n)
        lambda_min_ratio = 0.0001
        lambda_hp = params.get("lambda", 1.0)

        # Use R's interpolation formula
        lambda_ = lambda_max * lambda_min_ratio + lambda_hp * (lambda_max - lambda_max * lambda_min_ratio)

        self.logger.debug(f"\nLambda calculation:")
        self.logger.debug(f"Max dot product: {np.max(dot_prod):.6e}")
        self.logger.debug(f"Lambda max: {lambda_max:.6e}")
        self.logger.debug(f"Lambda HP: {lambda_hp:.6f}")
        self.logger.debug(f"Final lambda: {lambda_:.6e}")

        # Fit model
        model = Ridge(alpha=lambda_, fit_intercept=True)
        model.fit(X_scaled, y_scaled)

        # Get predictions and unscale
        y_train_pred = model.predict(X_scaled) * y_sd + y_mean

        # Calculate validation and test predictions if needed
        if ts_validation and X_val is not None and X_test is not None:
            # Scale validation and test data using training set statistics
            X_val_scaled = (X_val.to_numpy() - x_means) / x_sds
            X_test_scaled = (X_test.to_numpy() - x_means) / x_sds

            y_val_pred = model.predict(X_val_scaled) * y_sd + y_mean
            y_test_pred = model.predict(X_test_scaled) * y_sd + y_mean

            # Calculate metrics for all sets
            rss_val = np.sum((y_val - y_val_pred) ** 2)
            rss_test = np.sum((y_test - y_test_pred) ** 2)
            tss_val = np.sum((y_val - np.mean(y_val)) ** 2)
            tss_test = np.sum((y_test - np.mean(y_test)) ** 2)

            metrics = {
                "rsq_val": max(0, 1 - rss_val / tss_val),
                "rsq_test": max(0, 1 - rss_test / tss_test),
                "nrmse_val": np.sqrt(rss_val / len(y_val)) / (np.max(y_val) - np.min(y_val)),
                "nrmse_test": np.sqrt(rss_test / len(y_test)) / (np.max(y_test) - np.min(y_test)),
            }
        else:
            metrics = {"rsq_val": 0.0, "rsq_test": 0.0, "nrmse_val": 0.0, "nrmse_test": 0.0}

        # Calculate training metrics
        rss_train = np.sum((y_train - y_train_pred) ** 2)
        tss_train = np.sum((y_train - np.mean(y_train)) ** 2)

        self.logger.debug(f"\nMetrics calculation:")
        self.logger.debug(f"RSS train: {rss_train:.6e}")
        self.logger.debug(f"TSS train: {tss_train:.6e}")

        metrics.update(
            {
                "rsq_train": max(0, 1 - rss_train / tss_train),
                "nrmse_train": np.sqrt(rss_train / len(y_train)) / (np.max(y_train) - np.min(y_train)),
                "nrmse": (
                    metrics["nrmse_val"]
                    if ts_validation
                    else np.sqrt(rss_train / len(y_train)) / (np.max(y_train) - np.min(y_train))
                ),
            }
        )

        # Calculate RSSD
        paid_media_cols = [col for col in X_train.columns if col in self.mmm_data.mmmdata_spec.paid_media_spends]
        decomp_rssd = self._calculate_rssd(model, X_train, paid_media_cols, rssd_zero_penalty)

        elapsed_time = time.time() - start_time

        # Update metrics dictionary with additional values
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
                "pos": all(model.coef_[list(X_train.columns).index(col)] >= 0 for col in paid_media_cols),
            }
        )

        # Format hyperparameters and calculate decompositions
        params_formatted = self._format_hyperparameter_names(params)
        params_formatted.update(
            {"sol_id": sol_id, "Elapsed": elapsed_time, "ElapsedAccum": elapsed_time, "pos": metrics["pos"]}
        )

        x_decomp_agg = self._calculate_x_decomp_agg(model, X_train, y_train, {**params_formatted, **metrics})
        decomp_spend_dist = self._calculate_decomp_spend_dist(
            model, X_train, y_train, {**metrics, "params": params_formatted}
        )

        # Calculate loss
        loss = (
            objective_weights[0] * metrics["nrmse"]
            + objective_weights[1] * decomp_rssd
            + (objective_weights[2] * metrics["mape"] if len(objective_weights) > 2 else 0)
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
                        hyper_collect["hyper_bound_list_updated"][param_key] = param_value
                        hyper_collect["hyper_list_all"][f"{channel}_{param}"] = param_value  # Store as list
                    elif not isinstance(param_value, list):
                        hyper_collect["hyper_bound_list_fixed"][f"{channel}_{param}"] = param_value
                        hyper_collect["hyper_list_all"][f"{channel}_{param}"] = [
                            param_value,
                            param_value,
                        ]  # Store as list
        # Handle lambda parameter similarly
        if isinstance(prepared_hyperparameters.lambda_, list) and len(prepared_hyperparameters.lambda_) == 2:
            hyper_collect["hyper_bound_list_updated"]["lambda"] = prepared_hyperparameters.lambda_
            hyper_collect["hyper_list_all"]["lambda"] = prepared_hyperparameters.lambda_
        else:
            hyper_collect["hyper_bound_list_fixed"]["lambda"] = prepared_hyperparameters.lambda_
            hyper_collect["hyper_list_all"]["lambda"] = [
                prepared_hyperparameters.lambda_,
                prepared_hyperparameters.lambda_,
            ]
        # Handle train_size similarly
        if ts_validation:
            if isinstance(prepared_hyperparameters.train_size, list) and len(prepared_hyperparameters.train_size) == 2:
                hyper_collect["hyper_bound_list_updated"]["train_size"] = prepared_hyperparameters.train_size
                hyper_collect["hyper_list_all"]["train_size"] = prepared_hyperparameters.train_size
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

        nrmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2)) / (np.max(y_train) - np.min(y_train))
        nrmse_val = (
            np.sqrt(np.mean((y_val - y_val_pred) ** 2)) / (np.max(y_val) - np.min(y_val))
            if y_val is not None
            else None
        )
        nrmse_test = (
            np.sqrt(np.mean((y_test - y_test_pred) ** 2)) / (np.max(y_test) - np.min(y_test))
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

    def _lambda_seq(self, x: np.ndarray, y: np.ndarray, seq_len: int = 100) -> np.ndarray:
        """Calculate lambda sequence exactly matching R's glmnet implementation"""
        n = len(y)
        alpha = 0.001  # R's ridge default

        # R-style standardization
        def r_scale(x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            x_scaled = np.zeros_like(x, dtype=float)

            for j in range(x.shape[1]):
                mean_val = np.mean(x[:, j])
                # Use R's sd calculation
                sd = np.sqrt(np.sum((x[:, j] - mean_val) ** 2) / len(x[:, j]))
                if sd > 0:
                    x_scaled[:, j] = (x[:, j] - mean_val) / sd
            return x_scaled

        # Scale x and y like R
        x_scaled = r_scale(x)
        y_scaled = r_scale(y.reshape(-1, 1)).ravel()

        # Calculate lambda_max using R's method
        dot_prod = np.abs(x_scaled.T @ y_scaled)
        lambda_max = np.max(dot_prod) / (alpha * n)
        lambda_min_ratio = 0.0001

        self.logger.debug(f"\nLambda sequence debug:")
        self.logger.debug(f"lambda_max: {lambda_max:.6e}")
        self.logger.debug(f"min ratio: {lambda_min_ratio}")

        # Generate sequence using R's spacing
        log_lambda = np.linspace(np.log(lambda_max), np.log(lambda_max * lambda_min_ratio), seq_len)
        lambdas = np.exp(log_lambda)

        self.logger.debug(f"Sequence range: [{lambdas.min():.6e}, {lambdas.max():.6e}]")
        return lambdas

    def _calculate_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int, df_int: int = 1) -> float:
        """Calculate R-squared exactly matching R's implementation"""
        n = len(y_true)
        resid_ss = np.sum((y_true - y_pred) ** 2)
        total_ss = np.sum((y_true - np.mean(y_true)) ** 2)

        # self.logger.debug debug info
        self.logger.debug(f"\nR-squared calculation debug:")
        self.logger.debug(f"Residual SS: {resid_ss:.6e}")
        self.logger.debug(f"Total SS: {total_ss:.6e}")
        self.logger.debug(f"N: {n}")
        self.logger.debug(f"N features: {n_features}")

        # R's unadjusted R-squared
        r2 = 1 - (resid_ss / total_ss)

        # R's adjustment formula
        df_error = n - n_features - df_int
        df_total = n - df_int
        adj_r2 = 1 - (1 - r2) * (df_total / df_error)

        self.logger.debug(f"Unadjusted R2: {r2:.6f}")
        self.logger.debug(f"Adjusted R2: {adj_r2:.6f}")

        return max(0, adj_r2)  # R clips negative values to 0
