# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData


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
        self.mmmdata = mmmdata
        self.holidays_data = holidays_data
        self.hyperparameters = hyperparameters
        self.calibration_input = calibration_input
        self.featurized_mmm_data = featurized_mmm_data

    @abstractmethod
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
        Execute the model run.

        This abstract method should be implemented by subclasses to define the
        specific logic for running the marketing mix model.

        Args:
            dt_hyper_fixed (Optional[Dict[str, Any]]): Fixed hyperparameters.
            ts_validation (bool): Whether to use time series validation.
            add_penalty_factor (bool): Whether to add penalty factors.
            refresh (bool): Whether to refresh the model.
            seed (int): Random seed for reproducibility.
            cores (Optional[int]): Number of CPU cores to use.
            trials_config (Optional[TrialsConfig]): Configuration for trials.
            rssd_zero_penalty (bool): Whether to apply zero penalty in RSSD calculation.
            objective_weights (Optional[Dict[str, float]]): Weights for objectives.
            nevergrad_algo (NevergradAlgorithm): Nevergrad algorithm to use.
            intercept (bool): Whether to include an intercept.
            intercept_sign (str): Sign constraint for the intercept.
            outputs (bool): Whether to generate additional outputs.
            model_name (Models): Name of the model to use.

        Returns:
            ModelOutputs: The outputs of the model run.
        """
        pass
