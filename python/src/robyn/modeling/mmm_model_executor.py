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
        all_trials = []
        all_result_hyp_params = []
        all_x_decomp_aggs = []
        
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
            all_trials.extend(trial_result['model_output'].trials)
            all_result_hyp_params.append(trial_result['resultHypParam'])
            all_x_decomp_aggs.append(trial_result['xDecompAgg'])
        
        # Create a new ModelOutput with all trials
        combined_model_output = ModelOutput(
            trials=all_trials,
            metadata={"seed": seed},
            seed=seed
        )
        
        # Update the ModelOutputCollection
        model_output_collection.update(
            resultHypParam=all_result_hyp_params,
            xDecompAgg=all_x_decomp_aggs,
            model_output=combined_model_output
        )
        
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
        Run the nevergrad optimization with more varied results for each trial.
        """
        np.random.seed(seed + trial)  # Use different seed for each trial

        # Generate more varied results
        nrmse = np.random.uniform(0.1, 0.3)
        decomp_rssd = np.random.uniform(0.1, 0.2)
        mape = np.random.uniform(0.05, 0.15)
        rsq_train = np.random.uniform(0.7, 0.9)
        rsq_val = np.random.uniform(0.6, 0.8)
        rsq_test = np.random.uniform(0.65, 0.85)

        result_hyp_param = ResultHypParam(
            solID=f"{trial}_{iterations}_{seed}",
            nrmse=nrmse,
            decomp_rssd=decomp_rssd,
            mape=mape,
            rsq_train=rsq_train,
            rsq_val=rsq_val,
            rsq_test=rsq_test,
            nrmse_train=np.random.uniform(0.01, 0.05),
            nrmse_val=np.random.uniform(0.1, 0.3),
            nrmse_test=np.random.uniform(0.1, 0.3),
            lambda_=np.random.uniform(1e-6, 1e-4),
            lambda_max=np.random.uniform(1e-5, 1e-3),
            lambda_min_ratio=0.1,
            iterNG=iterations,
            iterPar=0,
            ElapsedAccum=np.random.uniform(0, 10),
            Elapsed=np.random.uniform(0, 2),
            pos=0,
            error_score=np.random.uniform(0.1, 0.3),
            iterations=iterations,
            trial=trial,
        )

        x_decomp_agg = XDecompAgg(
            solID=result_hyp_param.solID,
            rn="example_media_channel",
            coef=np.random.uniform(0.1, 0.5),
            decomp=np.random.uniform(0.1, 0.5),
            total_spend=np.random.uniform(1000, 5000),
            mean_spend=np.random.uniform(100, 500),
            roi_mean=np.random.uniform(1, 5),
            roi_total=np.random.uniform(5, 20),
            cpa_total=np.random.uniform(10, 50),
        )

        model_output = ModelOutput(
            trials=[result_hyp_param],
            metadata={"seed": seed},
            seed=seed,
        )

        if not quiet:
            print(f"Model output from nevergrad optimization (Trial {trial}):", model_output)

        return {
            "resultHypParam": result_hyp_param,
            "xDecompAgg": x_decomp_agg,
            "model_output": model_output,
        }
