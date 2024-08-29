# mmm_model_executor.py
from typing import Any, Dict, Optional, Tuple

import numpy as np
from robyn.modeling.entities.mmmdata_collection import MMMDataCollection
from robyn.modeling.entities.modeloutput import ModelOutput, ResultHypParam, XDecompAgg
from robyn.modeling.entities.modeloutput_collection import ModelOutputCollection
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.mmm_pareto_optimizer import ParetoOptimizer
from robyn.modeling.model_evaluation import ModelEvaluator
from scipy.optimize import minimize


class MMMModelExecutor:
    def __init__(self):
        self.model_evaluator = ModelEvaluator()
        self.pareto_optimizer = ParetoOptimizer()

    def model_run(
        self,
        mmmdata_collection: MMMDataCollection,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        json_file: Optional[str] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        refresh: bool = False,
        seed: int = 123,
        quiet: bool = False,
        cores: Optional[int] = None,
        trials: int = 5,
        iterations: int = 2000,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        nevergrad_algo: str = "TwoPointsDE",
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        lambda_control: Optional[float] = None,
        outputs: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> ModelOutputCollection:
        trials_config = TrialsConfig(
            num_trials=trials,
            num_iterations_per_trial=iterations,
            timeseries_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
        )
        kwargs.pop("trials_config", None)
        model_output = self.robyn_train(
            mmmdata_collection=mmmdata_collection,
            trials_config=trials_config,
            dt_hyper_fixed=dt_hyper_fixed,
            json_file=json_file,
            refresh=refresh,
            seed=seed,
            quiet=quiet,
            cores=cores,
            rssd_zero_penalty=rssd_zero_penalty,
            objective_weights=objective_weights,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            lambda_control=lambda_control,
            *args,
            **kwargs,
        )
        if outputs:
            pass
        return model_output

    def robyn_train(
        self,
        mmmdata_collection: MMMDataCollection,
        trials_config: TrialsConfig,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        json_file: Optional[str] = None,
        refresh: bool = False,
        seed: int = 123,
        quiet: bool = False,
        cores: Optional[int] = None,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        nevergrad_algo: str = "TwoPointsDE",
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        lambda_control: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> ModelOutputCollection:
        model_output_collection = ModelOutputCollection()
        for trial in range(trials_config.num_trials):
            trial_result = self.run_nevergrad_optimization(
                mmmdata_collection=mmmdata_collection,
                iterations=trials_config.num_iterations_per_trial,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                intercept_sign=intercept_sign,
                intercept=intercept,
                ts_validation=trials_config.timeseries_validation,
                add_penalty_factor=trials_config.add_penalty_factor,
                objective_weights=objective_weights,
                dt_hyper_fixed=dt_hyper_fixed,
                rssd_zero_penalty=rssd_zero_penalty,
                refresh=refresh,
                trial=trial,
                seed=seed,
                quiet=quiet,
                *args,
                **kwargs,
            )
            model_output_collection.update(**trial_result)
        return model_output_collection

    def run_nevergrad_optimization(
        self,
        mmmdata_collection: MMMDataCollection,
        iterations: int,
        cores: Optional[int],
        nevergrad_algo: str,
        intercept_sign: str,
        intercept: bool = True,
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        rssd_zero_penalty: bool = True,
        refresh: bool = False,
        trial: int = 1,
        seed: int = 123,
        quiet: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the nevergrad optimization.
        """
        np.random.seed(seed)  # For reproducibility

        # Example objective function: minimize the sum of squares of parameters
        def objective(params):
            return np.sum(params**2)

        # Initial parameters (random start)
        initial_params = np.random.rand(5)  # Example: 5 parameters
        # Example optimization using scipy's minimize
        result = minimize(
            objective,
            initial_params,
            method="Nelder-Mead",
            options={"maxiter": iterations},
        )
        # Create realistic ResultHypParam based on optimization results
        result_hyp_param = ResultHypParam(
            solID=f"{trial}_{iterations}_{seed}",
            nrmse=result.fun,  # Use the function value as an example nrmse
            decomp_rssd=np.random.rand(),  # Dummy value
            mape=np.random.rand(),  # Dummy value
            rsq_train=np.random.rand(),  # Dummy value
            rsq_val=np.random.rand(),  # Dummy value
            rsq_test=np.random.rand(),  # Dummy value
            nrmse_train=np.random.rand(),  # Dummy value
            nrmse_val=np.random.rand(),  # Dummy value
            nrmse_test=np.random.rand(),  # Dummy value
            lambda_max=np.max(result.x),  # Maximum lambda value
            lambda_min_ratio=0.1,  # Example ratio
            iterNG=iterations,
            iterPar=0,  # Example parallel iteration count
            ElapsedAccum=0.0,  # Accumulated time
            Elapsed=0.0,  # Elapsed time for this call
            pos=0,  # Example position index
            error_score=np.random.rand(),  # Dummy error score
            lambda_=np.mean(result.x),  # Average lambda value
            iterations=iterations,
            trial=trial,
        )
        # Create realistic XDecompAgg based on optimization results
        x_decomp_agg = XDecompAgg(
            solID=result_hyp_param.solID,
            rn="example_media_channel",
            coef=np.mean(
                result.x
            ),  # Use the mean of the optimized parameters as an example coefficient
            decomp=np.var(result.x),  # Use the variance as an example decomp value
            total_spend=np.sum(result.x),  # Total spend as the sum of parameters
            mean_spend=np.mean(result.x),  # Mean spend
            roi_mean=np.mean(result.x) / np.var(result.x),  # Example ROI calculation
            roi_total=np.sum(result.x) / np.var(result.x),  # Total ROI
            cpa_total=np.sum(result.x) / np.mean(result.x),  # Example CPA calculation
        )
        # Example ModelOutput with realistic data
        model_output = ModelOutput(
            trials=[result_hyp_param],  # Include the result as a trial
            metadata={"seed": seed},
            seed=seed,
        )
        print("Model output from nevergrad optimization:", model_output)
        return {
            "resultHypParam": result_hyp_param,
            "xDecompAgg": x_decomp_agg,
            "model_output": model_output,
        }
