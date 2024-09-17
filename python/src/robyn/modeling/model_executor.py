# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from robyn.modeling.base_model_executor import BaseModelExecutor
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.enums import NevergradAlgorithm, Models
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.ridge_model_builder import RidgeModelBuilder


class ModelExecutor(BaseModelExecutor):
    """
    Concrete implementation of the model executor for marketing mix models.

    This class extends BaseModelExecutor and implements the model_run method
    to execute specific types of marketing mix models, particularly the Ridge
    regression model. It serves as the main entry point for running models
    with various configurations and hyperparameters.
    """

    def model_run(
        self,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        refresh: bool = False,
        seed: int = 123,
        cores: Optional[int] = None,
        trials_config: Optional[TrialsConfig] = None,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        outputs: bool = False,
        model_name: Models = Models.RIDGE,
    ) -> ModelOutputs:
        """
        Execute the Robyn model run with specified parameters.

        This method orchestrates the entire model running process, including
        hyperparameter optimization, model training, and output generation.
        It currently supports Ridge regression models and can be extended
        to support other model types in the future.

        Args:
            dt_hyper_fixed (Optional[Dict[str, Any]]): Fixed hyperparameters for the model.
                If provided, these values will not be optimized.
            ts_validation (bool): Whether to use time series validation.
                If True, the data will be split into train, validation, and test sets.
            add_penalty_factor (bool): Whether to add penalty factors to the regularization.
                This can help in handling multicollinearity.
            refresh (bool): Whether to refresh the model, typically used in iterative modeling processes.
            seed (int): Random seed for reproducibility of results.
            cores (Optional[int]): Number of CPU cores to use for parallel processing.
                If None, will use a default value based on system capabilities.
            trials_config (Optional[TrialsConfig]): Configuration for multiple trials of model training.
            rssd_zero_penalty (bool): Whether to apply a penalty for zero coefficients in RSSD calculation.
            objective_weights (Optional[Dict[str, float]]): Weights for different objectives in the optimization process.
            nevergrad_algo (NevergradAlgorithm): The Nevergrad algorithm to use for hyperparameter optimization.
            intercept (bool): Whether to include an intercept term in the model.
            intercept_sign (str): Sign constraint for the intercept ('non_negative' or 'unconstrained').
            outputs (bool): Whether to generate additional outputs beyond the standard model results.
            model_name (Models): The type of model to run. Currently, only RIDGE is supported.

        Returns:
            ModelOutputs: The outputs of the model run, including trained model, performance metrics,
                          and various analysis results.

        Raises:
            NotImplementedError: If a model other than Ridge regression is specified.

        Note:
            This method is the core of the ModelExecutor class and ties together various
            components of the Robyn framework. It's designed to be flexible and extensible
            to accommodate future model types and configurations.
        """
        if model_name == Models.RIDGE:
            model_builder = RidgeModelBuilder(
                self.mmmdata,
                self.holidays_data,
                self.calibration_input,
                self.hyperparameters,
                self.featurized_mmm_data,
            )
            return model_builder.build_models(
                trials_config=trials_config,
                dt_hyper_fixed=dt_hyper_fixed,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                seed=seed,
                rssd_zero_penalty=rssd_zero_penalty,
                objective_weights=objective_weights,
                nevergrad_algo=nevergrad_algo,
                intercept=intercept,
                intercept_sign=intercept_sign,
            )
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented yet.")
