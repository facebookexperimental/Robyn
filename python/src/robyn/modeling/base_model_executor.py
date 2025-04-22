# base_model_executor.py
# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import numpy as np
import logging
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.data.entities.enums import AdstockType
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.transformations.transformations import Transformation
import nevergrad as ng

logger = logging.getLogger(__name__)


class BaseModelExecutor(ABC):
    """
    Abstract base class for executing marketing mix models.

    This class defines the interface for model executors and provides common
    initialization logic for different types of models.
    """

    def __init__(
        self,
        mmmdata: MMMData,
        holidays_data: HolidaysData,
        hyperparameters: Hyperparameters,
        calibration_input: CalibrationInput,
        featurized_mmm_data: FeaturizedMMMData,
    ) -> None:
        """
        Initialize the BaseModelExecutor.

        Args:
            mmmdata (MMMData): Marketing Mix Model data.
            holidays_data (HolidaysData): Holiday data for the model.
            hyperparameters (Hyperparameters): Model hyperparameters.
            calibration_input (CalibrationInput): Calibration input data.
            featurized_mmm_data (FeaturizedMMMData): Featurized MMM data.
        """
        logger.info("Initializing BaseModelExecutor")
        logger.debug(
            "Input data shapes - MMMData: %s, Holidays: %s",
            mmmdata.data.shape if hasattr(mmmdata, "data") else "None",
            (
                len(holidays_data.holidays)
                if hasattr(holidays_data, "holidays")
                else "None"
            ),
        )

        self.mmmdata = mmmdata
        self.holidays_data = holidays_data
        self.hyperparameters = hyperparameters
        self.calibration_input = calibration_input
        self.featurized_mmm_data = featurized_mmm_data
        self.transformation = Transformation(mmmdata)

        logger.debug(
            "Initialized with %d paid media variables, %d context variables, %d organic variables",
            len(mmmdata.mmmdata_spec.paid_media_vars),
            len(mmmdata.mmmdata_spec.context_vars),
            len(mmmdata.mmmdata_spec.organic_vars),
        )

    @abstractmethod
    def model_run(
        self,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        refresh: bool = False,
        seed: List[int] = [123],
        cores: Optional[int] = None,
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
        pass

    def _validate_input(self) -> None:
        """
        Validate the input data and parameters.

        Raises:
            ValueError: If any required data or parameter is missing or invalid.
        """
        logger.debug("Validating input parameters")

        validation_errors = []
        if self.mmmdata is None:
            validation_errors.append("MMMData is required")
        if self.holidays_data is None:
            validation_errors.append("HolidaysData is required")
        if self.hyperparameters is None:
            validation_errors.append("Hyperparameters are required")
        if self.featurized_mmm_data is None:
            validation_errors.append("FeaturizedMMMData is required")

        if validation_errors:
            error_msg = "; ".join(validation_errors)
            logger.error("Input validation failed: %s", error_msg)
            raise ValueError(error_msg)

        logger.info("Input validation successful")

    def _prepare_hyperparameters(
        self,
        dt_hyper_fixed: Optional[Dict[str, Any]],
        add_penalty_factor: bool,
        ts_validation: bool,
    ) -> Dict[str, Any]:
        logger.info("Preparing hyperparameters")
        logger.debug(
            "Initial hyperparameters config: fixed=%s, add_penalty=%s, ts_validation=%s",
            bool(dt_hyper_fixed),
            add_penalty_factor,
            ts_validation,
        )

        prepared_hyperparameters = self.hyperparameters.copy()

        if dt_hyper_fixed:
            logger.debug(
                "Updating with fixed hyperparameters: %s", dt_hyper_fixed.keys()
            )
            for key, value in dt_hyper_fixed.items():
                if prepared_hyperparameters.has_channel(key):
                    channel_params = prepared_hyperparameters.get_hyperparameter(key)
                    for param in [
                        "thetas",
                        "shapes",
                        "scales",
                        "alphas",
                        "gammas",
                        "penalty",
                    ]:
                        if param in value:
                            logger.debug("Setting %s.%s = %s", key, param, value[param])
                            setattr(channel_params, param, value[param])

        if add_penalty_factor:
            logger.debug("Adding penalty factors to all channels")
            for channel in prepared_hyperparameters.hyperparameters:
                channel_params = prepared_hyperparameters.get_hyperparameter(channel)
                channel_params.penalty = [0, 1]

        if ts_validation and not prepared_hyperparameters.train_size:
            logger.debug("Setting default train_size for time series validation")
            prepared_hyperparameters.train_size = [0.5, 0.8]

        hyper_to_optimize = {}
        for channel, channel_params in prepared_hyperparameters.hyperparameters.items():
            for param in ["thetas", "shapes", "scales", "alphas", "gammas"]:
                values = getattr(channel_params, param)
                if isinstance(values, list) and len(values) == 2:
                    hyper_to_optimize[f"{channel}_{param}"] = values
                    logger.debug(
                        "Added %s_%s to optimization parameters", channel, param
                    )

        if (
            isinstance(prepared_hyperparameters.lambda_, list)
            and len(prepared_hyperparameters.lambda_) == 2
        ):
            hyper_to_optimize["lambda"] = prepared_hyperparameters.lambda_
            logger.debug("Added lambda to optimization parameters")

        if (
            isinstance(prepared_hyperparameters.train_size, list)
            and len(prepared_hyperparameters.train_size) == 2
        ):
            hyper_to_optimize["train_size"] = prepared_hyperparameters.train_size
            logger.debug("Added train_size to optimization parameters")

        logger.info(
            "Completed hyperparameter preparation with %d parameters to optimize",
            len(hyper_to_optimize),
        )

        return {
            "prepared_hyperparameters": prepared_hyperparameters,
            "hyper_to_optimize": hyper_to_optimize,
        }
