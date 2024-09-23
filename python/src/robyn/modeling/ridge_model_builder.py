# pyre-strict


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
import nevergrad as ng

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm


@dataclass(frozen=True)
class ModelRefitOutput:
    """
    Contains the results of refitting a model.

    Attributes:
        rsq_train (float): R-squared value for the training set.
        rsq_val (Optional[float]): R-squared value for the validation set, if applicable.
        rsq_test (Optional[float]): R-squared value for the test set, if applicable.
        nrmse_train (float): Normalized Root Mean Square Error for the training set.
        nrmse_val (Optional[float]): Normalized RMSE for the validation set, if applicable.
        nrmse_test (Optional[float]): Normalized RMSE for the test set, if applicable.
        coefs (np.ndarray): Coefficients of the fitted model.
        y_train_pred (np.ndarray): Predicted values for the training set.
        y_val_pred (Optional[np.ndarray]): Predicted values for the validation set, if applicable.
        y_test_pred (Optional[np.ndarray]): Predicted values for the test set, if applicable.
        y_pred (np.ndarray): All predicted values combined.
        mod (Ridge): The fitted Ridge regression model.
        df_int (int): Degrees of freedom for the intercept.
    """

    rsq_train: float
    rsq_val: Optional[float]
    rsq_test: Optional[float]
    nrmse_train: float
    nrmse_val: Optional[float]
    nrmse_test: Optional[float]
    coefs: np.ndarray
    y_train_pred: np.ndarray
    y_val_pred: Optional[np.ndarray]
    y_test_pred: Optional[np.ndarray]
    y_pred: np.ndarray
    mod: Ridge
    df_int: int


@dataclass(frozen=True)
class HyperCollectorOutput:
    """
    Contains the results of collecting hyperparameters.

    Attributes:
        hyper_list_all (Hyperparameters): All hyperparameters, including both fixed and variable.
        hyper_bound_list_updated (Hyperparameters): Updated hyperparameter bounds for optimization.
        hyper_bound_list_fixed (Hyperparameters): Fixed hyperparameters that won't be optimized.
        dt_hyper_fixed_mod (pd.DataFrame): DataFrame of fixed hyperparameters.
        all_fixed (bool): Indicates if all hyperparameters are fixed.
    """

    hyper_list_all: Hyperparameters
    hyper_bound_list_updated: Hyperparameters
    hyper_bound_list_fixed: Hyperparameters
    dt_hyper_fixed_mod: pd.DataFrame
    all_fixed: bool


@dataclass(frozen=True)
class ModelDecompOutput:
    """
    Contains the decomposition output of the model.

    Attributes:
        x_decomp_vec (pd.DataFrame): Decomposed vector of features.
        x_decomp_vec_scaled (pd.DataFrame): Scaled decomposed vector of features.
        x_decomp_agg (pd.DataFrame): Aggregated decomposition of features.
        coefs_out_cat (pd.DataFrame): Coefficients output by category.
        media_decomp_immediate (pd.DataFrame): Immediate media decomposition.
        media_decomp_carryover (pd.DataFrame): Carryover media decomposition.
    """

    x_decomp_vec: pd.DataFrame
    x_decomp_vec_scaled: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    coefs_out_cat: pd.DataFrame
    media_decomp_immediate: pd.DataFrame
    media_decomp_carryover: pd.DataFrame


class RidgeModelBuilder:
    """
    A class for building and managing Ridge regression models for Marketing Mix Modeling (MMM).

    This class handles the entire process of building, training, and analyzing Ridge
    regression models for MMM, including hyperparameter optimization and model decomposition.
    """

    def __init__(
        self,
        mmm_data: MMMData,
        holidays_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
    ) -> None:
        """
        Initialize the RidgeModelBuilder with necessary data and parameters.

        Args:
            mmm_data (MMMData): Marketing Mix Model data.
            holidays_data (HolidaysData): Holiday data for the model.
            calibration_input (CalibrationInput): Calibration input data.
            hyperparameters (Hyperparameters): Hyperparameters for the model.
            featurized_mmm_data (FeaturizedMMMData): Featurized MMM data.
        """
        self.mmm_data = mmm_data
        self.holidays_data = holidays_data
        self.calibration_input = calibration_input
        self.hyperparameters = hyperparameters
        self.featurized_mmm_data = featurized_mmm_data

    def build_models(
        self,
        trials_config: TrialsConfig,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        seed: int = 123,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[List[float]] = None,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        adstock: AdstockType = AdstockType.GEOMETRIC,
        cores: int = 1,
    ) -> ModelOutputs:
        np.random.seed(seed)

        hyper_collect = self._hyper_collector(
            adstock=adstock.value,
            all_media=self.mmm_data.all_media,
            paid_media_spends=self.mmm_data.paid_media_spends,
            organic_vars=self.mmm_data.organic_vars,
            prophet_vars=self.mmm_data.prophet_vars,
            context_vars=self.mmm_data.context_vars,
            dt_mod=self.featurized_mmm_data.dt_mod,
            hyper_in=self.hyperparameters,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            dt_hyper_fixed=dt_hyper_fixed,
            cores=cores,
        )

        model_outputs = self._model_train(
            hyper_collect=hyper_collect,
            trials_config=trials_config,
            cores=cores,
            intercept_sign=intercept_sign,
            intercept=intercept,
            nevergrad_algo=nevergrad_algo,
            dt_hyper_fixed=dt_hyper_fixed,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            objective_weights=objective_weights,
            rssd_zero_penalty=rssd_zero_penalty,
            seed=seed,
        )

        return model_outputs

    def _model_train(
        self,
        hyper_collect: Hyperparameters,
        trials_config: TrialsConfig,
        cores: int,
        intercept_sign: str,
        intercept: bool,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        rssd_zero_penalty: bool = True,
        seed: int = 123,
    ) -> ModelOutputs:
        trials_results = []

        for trial in range(trials_config.trials):
            trial_seed = seed + trial
            trial_results = self.run_nevergrad_optimization(
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
                seed=trial_seed,
            )
            trials_results.extend(trial_results)

        return ModelOutputs(trials=trials_config.trials, iterations=trials_config.iterations, results=trials_results)

    def run_nevergrad_optimization(
        self,
        hyper_collect: Hyperparameters,
        iterations: int,
        cores: int,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        rssd_zero_penalty: bool = True,
        trial: int = 1,
        seed: int = 123,
    ) -> List[Trial]:
        np.random.seed(seed)

        param_count = len(hyper_collect.hyper_bound_list_updated)
        instrumentation = ng.p.Array(shape=(param_count,))
        optimizer = ng.optimizers.registry[nevergrad_algo.value](instrumentation, budget=iterations, num_workers=cores)

        trial_results = []
        for _ in range(iterations):
            candidate = optimizer.ask()
            hyperparams = self._scale_hyperparameters(candidate.value, hyper_collect)

            model_output = self._fit_and_evaluate_model(
                hyperparams,
                ts_validation,
                add_penalty_factor,
                intercept,
                intercept_sign,
                rssd_zero_penalty,
                objective_weights,
            )

            optimizer.tell(candidate, model_output.objective)
            trial_results.append(
                Trial(hyperparameters=hyperparams, performance=model_output.performance, model=model_output.model)
            )

        return trial_results

    @staticmethod
    def model_decomp(
        coefs: Dict[str, float],
        y_pred: np.ndarray,
        dt_mod_saturated: pd.DataFrame,
        dt_saturated_immediate: pd.DataFrame,
        dt_saturated_carryover: pd.DataFrame,
        dt_mod_roll_wind: pd.DataFrame,
        refresh_added_start: str,
    ) -> ModelDecompOutput:
        x_decomp = dt_mod_saturated.copy()
        for col in x_decomp.columns:
            if col in coefs:
                x_decomp[col] *= coefs[col]

        x_decomp_vec = x_decomp.copy()
        x_decomp_vec["y_pred"] = y_pred
        x_decomp_vec_scaled = x_decomp_vec.div(x_decomp_vec.sum(axis=1), axis=0)

        x_decomp_agg = x_decomp.sum()
        x_decomp_agg_perc = x_decomp_agg / x_decomp_agg.sum()

        coefs_out_cat = pd.DataFrame({"variable": coefs.keys(), "coefficient": coefs.values()})

        media_decomp_immediate = dt_saturated_immediate.mul(coefs, axis=1)
        media_decomp_carryover = dt_saturated_carryover.mul(coefs, axis=1)

        return ModelDecompOutput(
            x_decomp_vec=x_decomp_vec,
            x_decomp_vec_scaled=x_decomp_vec_scaled,
            x_decomp_agg=pd.DataFrame(
                {"variable": x_decomp_agg.index, "value": x_decomp_agg.values, "percentage": x_decomp_agg_perc.values}
            ),
            coefs_out_cat=coefs_out_cat,
            media_decomp_immediate=media_decomp_immediate,
            media_decomp_carryover=media_decomp_carryover,
        )

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

        if intercept_sign == "non_negative" and model.intercept_ < 0:
            model.intercept_ = 0
            model = Ridge(alpha=lambda_, fit_intercept=False)
            model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val) if x_val is not None else None
        y_test_pred = model.predict(x_test) if x_test is not None else None

        rsq_train = model.score(x_train, y_train)
        rsq_val = model.score(x_val, y_val) if x_val is not None else None
        rsq_test = model.score(x_test, y_test) if x_test is not None else None

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
                if y_val_pred is not None and y_test_pred is not None
                else y_train_pred
            ),
            mod=model,
            df_int=1 if intercept else 0,
        )

    @staticmethod
    def _lambda_seq(
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> np.ndarray:
        n, p = x.shape
        lambda_max = np.linalg.norm(x.T.dot(y), ord=np.inf) / (n * 0.001)
        return np.logspace(np.log10(lambda_max * lambda_min_ratio), np.log10(lambda_max), num=seq_len)

    @staticmethod
    def _hyper_collector(
        adstock: str,
        all_media: List[str],
        paid_media_spends: List[str],
        organic_vars: List[str],
        prophet_vars: List[str],
        context_vars: List[str],
        dt_mod: pd.DataFrame,
        hyper_in: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        cores: int = 1,
    ) -> HyperCollectorOutput:
        hyper_names = RidgeModelBuilder._get_hyper_names(adstock, all_media, dt_mod.columns)

        hyper_list_all = {}
        hyper_bound_list_updated = {}
        hyper_bound_list_fixed = {}

        for name in hyper_names:
            if name in hyper_in:
                value = hyper_in[name]
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    hyper_bound_list_updated[name] = value
                else:
                    hyper_bound_list_fixed[name] = value
                hyper_list_all[name] = value

        if ts_validation and "train_size" not in hyper_list_all:
            hyper_bound_list_updated["train_size"] = [0.5, 0.8]
            hyper_list_all["train_size"] = [0.5, 0.8]

        if add_penalty_factor:
            for var in dt_mod.columns:
                penalty_name = f"{var}_penalty"
                if penalty_name not in hyper_list_all:
                    hyper_bound_list_updated[penalty_name] = [0, 1]
                    hyper_list_all[penalty_name] = [0, 1]

        all_fixed = len(hyper_bound_list_updated) == 0

        dt_hyper_fixed_mod = pd.DataFrame(hyper_bound_list_fixed, index=[0]) if hyper_bound_list_fixed else None

        return HyperCollectorOutput(
            hyper_list_all=hyper_list_all,
            hyper_bound_list_updated=hyper_bound_list_updated,
            hyper_bound_list_fixed=hyper_bound_list_fixed,
            dt_hyper_fixed_mod=dt_hyper_fixed_mod,
            all_fixed=all_fixed,
        )

    @staticmethod
    def _get_hyper_names(adstock: str, all_media: List[str], all_vars: List[str]) -> List[str]:
        hyper_names = []
        for media in all_media:
            if adstock == "geometric":
                hyper_names.extend([f"{media}_thetas", f"{media}_alphas", f"{media}_gammas"])
            elif adstock in ["weibull_cdf", "weibull_pdf"]:
                hyper_names.extend([f"{media}_shapes", f"{media}_scales", f"{media}_alphas", f"{media}_gammas"])

        hyper_names.extend([f"{var}_penalty" for var in all_vars])
        hyper_names.append("lambda")

        return hyper_names

    def _scale_hyperparameters(
        self, candidate_values: np.ndarray, hyper_collect: HyperCollectorOutput
    ) -> Dict[str, float]:
        """
        Scale the hyperparameters from the Nevergrad optimizer to their actual ranges.

        Args:
            candidate_values (np.ndarray): Candidate values from Nevergrad optimizer.
            hyper_collect (HyperCollectorOutput): Collected hyperparameters.

        Returns:
            Dict[str, float]: Scaled hyperparameters.
        """
        scaled_params = {}
        for i, (name, bounds) in enumerate(hyper_collect.hyper_bound_list_updated.items()):
            scaled_params[name] = bounds[0] + (bounds[1] - bounds[0]) * candidate_values[i]

        # Add fixed hyperparameters
        scaled_params.update(hyper_collect.hyper_bound_list_fixed)

        return scaled_params

    def _fit_and_evaluate_model(
        self,
        hyperparams: Dict[str, float],
        ts_validation: bool,
        add_penalty_factor: bool,
        intercept: bool,
        intercept_sign: str,
        rssd_zero_penalty: bool,
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit the Ridge regression model and evaluate its performance.

        Args:
            hyperparams (Dict[str, float]): Hyperparameters for the model.
            ts_validation (bool): Whether to use time series validation.
            add_penalty_factor (bool): Whether to add penalty factors to the regularization.
            intercept (bool): Whether to include an intercept term in the model.
            intercept_sign (str): Sign constraint for the intercept.
            rssd_zero_penalty (bool): Whether to apply a penalty for zero coefficients in RSSD calculation.
            objective_weights (Optional[Dict[str, float]]): Weights for different objectives in the optimization process.

        Returns:
            Dict[str, Any]: Model output including performance metrics and model object.
        """
        X, y = self._prepare_data(hyperparams)

        if ts_validation:
            train_size = hyperparams.get("train_size", 0.8)
            split_index = int(len(X) * train_size)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None

        lambda_ = hyperparams["lambda"]

        if add_penalty_factor:
            penalty_factor = [hyperparams.get(f"{col}_penalty", 1.0) for col in X.columns]
        else:
            penalty_factor = None

        model_output = self._model_refit(
            X_train,
            y_train,
            X_test,
            y_test,
            lambda_=lambda_,
            intercept=intercept,
            intercept_sign=intercept_sign,
            penalty_factor=penalty_factor,
        )

        rssd = self._calculate_rssd(model_output.coefs, rssd_zero_penalty)

        objective = self._calculate_objective(model_output.rsq_train, model_output.rsq_test, rssd, objective_weights)

        return {
            "model": model_output.mod,
            "performance": {
                "rsq_train": model_output.rsq_train,
                "rsq_test": model_output.rsq_test,
                "nrmse_train": model_output.nrmse_train,
                "nrmse_test": model_output.nrmse_test,
                "rssd": rssd,
            },
            "objective": objective,
        }

    def _prepare_data(self, hyperparams: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the data for modeling, including adstock transformations.

        Args:
            hyperparams (Dict[str, float]): Hyperparameters for the model.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Prepared features (X) and target variable (y).
        """
        dt_mod = self.featurized_mmm_data.dt_mod
        y = dt_mod[self.mmm_data.mmmdata_spec.dep_var]  # Use the dynamic field from MMMDataSpec if set
        X = dt_mod.drop(["ds", self.mmm_data.mmmdata_spec.dep_var], axis=1)

        for media in self.mmm_data.paid_media_spends:
            if self.mmm_data.adstock == AdstockType.GEOMETRIC:
                X[media] = self._geometric_adstock(X[media], hyperparams[f"{media}_thetas"])
            elif self.mmm_data.adstock in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
                X[media] = self._weibull_adstock(
                    X[media], hyperparams[f"{media}_shapes"], hyperparams[f"{media}_scales"], self.mmm_data.adstock
                )

            # Apply saturation (Hill transformation)
            X[media] = self._hill_transformation(
                X[media], hyperparams[f"{media}_alphas"], hyperparams[f"{media}_gammas"]
            )

        return X, y

    @staticmethod
    def _geometric_adstock(x: pd.Series, theta: float) -> pd.Series:
        """Apply geometric adstock transformation."""
        y = x.copy()
        for i in range(1, len(x)):
            y.iloc[i] += theta * y.iloc[i - 1]
        return y

    @staticmethod
    def _weibull_adstock(x: pd.Series, shape: float, scale: float, adstock_type: AdstockType) -> pd.Series:
        """Apply Weibull adstock transformation."""
        L = len(x)
        lag_weights = np.arange(1, L + 1)

        if adstock_type == AdstockType.WEIBULL_CDF:
            adstock_weights = 1 - np.exp(-((lag_weights / scale) ** shape))
        elif adstock_type == AdstockType.WEIBULL_PDF:
            adstock_weights = (
                (shape / scale) * (lag_weights / scale) ** (shape - 1) * np.exp(-((lag_weights / scale) ** shape))
            )
        else:
            raise ValueError(f"Invalid Weibull adstock type: {adstock_type}")

        adstock_weights = adstock_weights / np.sum(adstock_weights)
        return pd.Series(np.convolve(x, adstock_weights[::-1], mode="full")[:L])

    @staticmethod
    def _hill_transformation(x: pd.Series, alpha: float, gamma: float) -> pd.Series:
        """Apply Hill (saturation) transformation."""
        return x**gamma / (x**gamma + alpha**gamma)

    @staticmethod
    def _calculate_rssd(coefs: np.ndarray, rssd_zero_penalty: bool) -> float:
        """Calculate Root Sum Squared Distance (RSSD)."""
        rssd = np.sqrt(np.sum(coefs**2))
        if rssd_zero_penalty:
            zero_coef_ratio = np.sum(coefs == 0) / len(coefs)
            rssd *= 1 + zero_coef_ratio
        return rssd

    @staticmethod
    def _calculate_objective(
        rsq_train: float, rsq_test: Optional[float], rssd: float, objective_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate the objective function value."""
        if objective_weights is None:
            objective_weights = {"rsq_train": 1.0, "rsq_test": 1.0, "rssd": 1.0}

        objective = (
            objective_weights["rsq_train"] * (1 - rsq_train)
            + objective_weights.get("rsq_test", 0) * (1 - (rsq_test or 0))
            + objective_weights["rssd"] * rssd
        )

        return objective
