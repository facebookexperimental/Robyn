# pyre-strict

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm
from sklearn.linear_model import Ridge


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
        holiday_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
    ) -> None:
        """
        Initialize the RidgeModelBuilder with necessary data and parameters.

        Args:
            mmm_data (MMMData): Marketing Mix Model data.
            holiday_data (HolidaysData): Holiday data for the model.
            calibration_input (CalibrationInput): Calibration input data.
            hyperparameters (Hyperparameters): Hyperparameters for the model.
            featurized_mmm_data (FeaturizedMMMData): Featurized MMM data.
        """
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
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
    ) -> ModelOutputs:
        """
        Build and train multiple Ridge regression models based on the given configuration.

        Args:
            trials_config (TrialsConfig): Configuration for the number of trials and iterations.
            dt_hyper_fixed (Optional[pd.DataFrame]): Fixed hyperparameters, if any.
            ts_validation (bool): Whether to use time series validation.
            add_penalty_factor (bool): Whether to add penalty factors to the model.
            seed (int): Random seed for reproducibility.
            rssd_zero_penalty (bool): Whether to apply zero penalty in RSSD calculation.
            objective_weights (Optional[List[float]]): Weights for different objectives in optimization.
            nevergrad_algo (NevergradAlgorithm): Nevergrad algorithm to use for optimization.
            intercept (bool): Whether to include an intercept in the model.
            intercept_sign (str): Sign constraint for the intercept.
            adstock (AdstockType): Type of adstock to use.

        Returns:
            ModelOutputs: The outputs of the built models.
        """
        # Implementation here
        pass

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
        """
        Train the Ridge regression model with the given parameters.

        This method handles the core training process, including hyperparameter optimization
        and model evaluation across multiple trials.

        Args:
            hyper_collect (Hyperparameters): Collected hyperparameters.
            trials_config (TrialsConfig): Configuration for trials and iterations.
            cores (int): Number of CPU cores to use for parallel processing.
            intercept_sign (str): Sign constraint for the intercept.
            intercept (bool): Whether to include an intercept in the model.
            nevergrad_algo (NevergradAlgorithm): Nevergrad algorithm for optimization.
            dt_hyper_fixed (Optional[pd.DataFrame]): Fixed hyperparameters, if any.
            ts_validation (bool): Whether to use time series validation.
            add_penalty_factor (bool): Whether to add penalty factors to the model.
            objective_weights (Optional[Dict[str, float]]): Weights for different objectives.
            rssd_zero_penalty (bool): Whether to apply zero penalty in RSSD calculation.
            seed (int): Random seed for reproducibility.

        Returns:
            ModelOutputs: The outputs of the trained models.
        """
        # Implementation here
        pass

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
        """
        Run Nevergrad optimization for hyperparameter tuning.

        This method uses Nevergrad to optimize hyperparameters over multiple iterations,
        evaluating model performance for each set of hyperparameters.

        Args:
            hyper_collect (Hyperparameters): Collected hyperparameters.
            iterations (int): Number of iterations for optimization.
            cores (int): Number of CPU cores to use for parallel processing.
            nevergrad_algo (NevergradAlgorithm): Nevergrad algorithm for optimization.
            intercept (bool): Whether to include an intercept in the model.
            intercept_sign (str): Sign constraint for the intercept.
            ts_validation (bool): Whether to use time series validation.
            add_penalty_factor (bool): Whether to add penalty factors to the model.
            objective_weights (Optional[Dict[str, float]]): Weights for different objectives.
            dt_hyper_fixed (Optional[pd.DataFrame]): Fixed hyperparameters, if any.
            rssd_zero_penalty (bool): Whether to apply zero penalty in RSSD calculation.
            trial (int): Current trial number.
            seed (int): Random seed for reproducibility.

        Returns:
            List[Trial]: List of results for each iteration of the optimization.
        """
        # Implementation here
        pass

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
        """
        Perform model decomposition to analyze feature contributions.

        This method decomposes the model's predictions to understand the contribution
        of each feature, including immediate and carryover effects for media variables.

        Args:
            coefs (Dict[str, float]): Model coefficients.
            y_pred (np.ndarray): Predicted values.
            dt_mod_saturated (pd.DataFrame): Saturated model data.
            dt_saturated_immediate (pd.DataFrame): Immediate effects data.
            dt_saturated_carryover (pd.DataFrame): Carryover effects data.
            dt_mod_roll_wind (pd.DataFrame): Rolling window data.
            refresh_added_start (str): Start date for refresh period.

        Returns:
            ModelDecompOutput: Decomposition results.
        """
        # Implementation here
        pass

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
        """
        Refit the Ridge regression model with given parameters.

        This method refits the model using the provided data and parameters, calculating
        various performance metrics for train, validation, and test sets.

        Args:
            x_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target.
            x_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation target.
            x_test (Optional[np.ndarray]): Test features.
            y_test (Optional[np.ndarray]): Test target.
            lambda_ (float): Regularization strength.
            lower_limits (Optional[List[float]]): Lower bounds for coefficients.
            upper_limits (Optional[List[float]]): Upper bounds for coefficients.
            intercept (bool): Whether to fit an intercept.
            intercept_sign (str): Sign constraint for the intercept.

        Returns:
            ModelRefitOutput: Results of the refitted model.
        """
        # Implementation here
        pass

    @staticmethod
    def _lambda_seq(
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> np.ndarray:
        """
        Generate a sequence of lambda values for regularization.

        This method creates a sequence of lambda values to be used in regularization,
        based on the input data and specified parameters.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Target variable.
            seq_len (int): Length of the lambda sequence.
            lambda_min_ratio (float): Minimum ratio for lambda.

        Returns:
            np.ndarray: Sequence of lambda values.
        """
        # Implementation here
        pass

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
        """
        Collect and organize hyperparameters for model optimization.

        This method gathers hyperparameters from various sources, organizes them into
        fixed and variable sets, and prepares them for use in model optimization.

        Args:
            adstock (str): Type of adstock to use.
            all_media (List[str]): List of all media variables.
            paid_media_spends (List[str]): List of paid media spend variables.
            organic_vars (List[str]): List of organic variables.
            prophet_vars (List[str]): List of Prophet model variables.
            context_vars (List[str]): List of context variables.
            dt_mod (pd.DataFrame): Modified input data.
            hyper_in (Dict[str, Any]): Input hyperparameters.
            ts_validation (bool): Whether to use time series validation.
            add_penalty_factor (bool): Whether to add penalty factors.
            dt_hyper_fixed (Optional[pd.DataFrame]): Fixed hyperparameters, if any.
            cores (int): Number of CPU cores to use.

        Returns:
            HyperCollectorOutput: Organized hyperparameters and related information.
        """
        # Implementation here
        pass
