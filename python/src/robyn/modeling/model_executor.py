# model_executor.py
# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import os
import logging

from robyn.common.common_util import CommonUtils
from robyn.common.common_util import CommonUtils
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def model_run(
        self,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        refresh: bool = False,
        seed: List[int] = [123],
        cores: int = None,
        trials_config: Optional[TrialsConfig] = None,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        outputs: bool = False,
        model_name: Models = Models.RIDGE,
        lambda_control: Optional[float] = None,
    ) -> ModelOutputs:
        """
        Execute the Robyn model run with specified parameters.
        """
        self.logger.info("Starting model execution with model_name=%s", model_name)
        self.logger.debug(
            "Model configuration - ts_validation=%s, add_penalty_factor=%s, seed=%s, "
            "rssd_zero_penalty=%s, intercept=%s, intercept_sign=%s",
            ts_validation,
            add_penalty_factor,
            seed,
            rssd_zero_penalty,
            intercept,
            intercept_sign,
        )

        try:
            self._validate_input()
            self.logger.debug("Input validation successful")

            cores = CommonUtils.get_cores_available(cores)
            self.logger.debug("Using %d cores for processing", cores)

            prepared_hyperparameters = self._prepare_hyperparameters(
                dt_hyper_fixed, add_penalty_factor, ts_validation
            )
            self.logger.debug("Hyperparameters prepared: %s", prepared_hyperparameters)

            if model_name == Models.RIDGE:
                self.logger.info("Initializing Ridge model builder")
                model_builder = RidgeModelBuilder(
                    self.mmmdata,
                    self.holidays_data,
                    self.calibration_input,
                    prepared_hyperparameters,
                    self.featurized_mmm_data,
                )

                self.logger.info("Building models with configured parameters")
                model_outputs = model_builder.build_models(
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
                    cores=cores,
                )
                self.logger.info("Model building completed successfully")

                if outputs:
                    self.logger.debug("Generating additional outputs")
                    additional_outputs = self._generate_additional_outputs(
                        model_outputs
                    )
                    model_outputs.update(additional_outputs)
                    self.logger.debug(
                        "Additional outputs generated: %s", additional_outputs
                    )

                self.logger.info("Model execution completed successfully")
                return model_outputs
            else:
                error_msg = f"Model {model_name} is not implemented yet"
                self.logger.error(error_msg)
                raise NotImplementedError(error_msg)

        except Exception as e:
            self.logger.error("Error during model execution: %s", str(e), exc_info=True)
            raise

    def _generate_additional_outputs(
        self, model_outputs: ModelOutputs
    ) -> Dict[str, Any]:
        """
        Generate additional outputs based on the model results.
        """
        self.logger.debug("Starting additional outputs generation")
        additional_outputs = {}

        try:
            if hasattr(model_outputs, "trials") and model_outputs.trials > 1:
                self.logger.debug(
                    "Calculating average performance across %d trials",
                    model_outputs.trials,
                )
                avg_performance = (
                    sum(trial.performance for trial in model_outputs.results)
                    / model_outputs.trials
                )
                additional_outputs["average_performance"] = avg_performance
                self.logger.debug("Average performance calculated: %f", avg_performance)

            self.logger.debug(
                "Additional outputs generated successfully: %s", additional_outputs
            )
            return additional_outputs

        except Exception as e:
            self.logger.error(
                "Error generating additional outputs: %s", str(e), exc_info=True
            )
            raise

    def _validate_input(self):
        """
        Validates the input data before model execution.
        """
        self.logger.debug("Starting input validation")
        try:
            super()._validate_input()
            self.logger.debug(
                "Input validation successful - MMM data shape: %s, "
                "Holidays data shape: %s",
                getattr(self.mmmdata, "shape", "N/A"),
                getattr(self.holidays_data, "shape", "N/A"),
            )
        except Exception as e:
            self.logger.error("Input validation failed: %s", str(e), exc_info=True)
            raise
