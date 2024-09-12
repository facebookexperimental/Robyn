# pyre-strict

from typing import Optional, Dict, Any

from robyn.modeling.base_model_executor import BaseModelExecutor
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.enums import NevergradAlgorithm, Models
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.ridge_model_builder import RidgeModelBuilder


class ModelExecutor(BaseModelExecutor):

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
        Run the Robyn model with the specified parameters.

        Args:
            dt_hyper_fixed: Fixed hyperparameters.
            ts_validation: Enable time-series validation.
            add_penalty_factor: Add penalty factor.
            refresh: Refresh the model.
            seed: Random seed.
            cores: Number of cores to use.
            trials_config: Configuration for trials.
            rssd_zero_penalty: Enable RSSD zero penalty.
            objective_weights: Objective weights.
            nevergrad_algo: Nevergrad algorithm to use.
            intercept: Include intercept term.
            intercept_sign: Sign of the intercept term.
            outputs: Output results.
            model_name: Model to use.

        Returns:
            ModelOutputs: The results of the model run.
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
