#pyre-strict

from typing import Optional, Dict, Any

from robyn.modeling.base_model_executor import BaseModelExecutor
from robyn.modeling.trials_config import TrialsConfig
from robyn.modeling.enums import NevergradAlgorithm, Models
from robyn.modeling.model_outputs import ModelOutputs

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
            InputCollect: Input data collection.
            dt_hyper_fixed: Fixed hyperparameters.
            json_file: JSON file path.
            ts_validation: Enable time-series validation.
            add_penalty_factor: Add penalty factor.
            refresh: Refresh the model.
            seed: Random seed.
            quiet: Suppress output.
            cores: Number of cores to use.
            trials: Number of trials.
            iterations: Number of iterations.
            rssd_zero_penalty: Enable RSSD zero penalty.
            objective_weights: Objective weights.
            nevergrad_algo: Nevergrad algorithm to use.
            intercept: Include intercept term.
            intercept_sign: Sign of the intercept term.
            lambda_control: Lambda control value.
            outputs: Output results.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    #Call build_models from model_builder.py
    #Evaluator, clustering, and plotting
