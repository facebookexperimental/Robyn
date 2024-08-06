#pyre-strict

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ResultHypParam:
    solID: str
    nrmse: float
    decomp_rssd: float
    mape: float
    rsq_train: float
    rsq_val: Optional[float]
    rsq_test: Optional[float]
    nrmse_train: float
    nrmse_val: Optional[float]
    nrmse_test: Optional[float]
    lambda_: float
    lambda_max: float
    lambda_min_ratio: float
    iterations: int
    trial: int
    iterNG: int
    iterPar: int
    ElapsedAccum: float
    Elapsed: float
    pos: float
    error_score: float
    # Add other hyperparameters as needed

@dataclass(frozen=True)
class XDecompAgg:
    solID: str
    rn: str
    coef: float
    decomp: float
    total_spend: float
    mean_spend: float
    roi_mean: float
    roi_total: float
    cpa_total: float

@dataclass(frozen=True)
class DecompSpendDist:
    rn: str
    coefs: float
    xDecompAgg: float
    xDecompPerc: float
    xDecompMeanNon0: float
    xDecompMeanNon0Perc: float
    pos: float
    total_spend: float
    mean_spend: float
    spend_share: float
    effect_share: float
    roi_mean: float
    roi_total: float
    cpa_mean: float
    cpa_total: float
    solID: str
    trial: int
    iterNG: int
    iterPar: int

@dataclass(frozen=True)
class ResultCollect:
    resultHypParam: List[ResultHypParam]
    xDecompAgg: List[XDecompAgg]
    decompSpendDist: List[DecompSpendDist]
    elapsed_min: float

    @classmethod
    def create(cls, result_collect_dict: Dict[str, Any]) -> 'ResultCollect':
        return cls(
            resultHypParam=[ResultHypParam(**rhp) for rhp in result_collect_dict['resultHypParam']],
            xDecompAgg=[XDecompAgg(**xda) for xda in result_collect_dict['xDecompAgg']],
            decompSpendDist=[DecompSpendDist(**dsd) for dsd in result_collect_dict['decompSpendDist']],
            elapsed_min=result_collect_dict['elapsed.min']
        )

@dataclass(frozen=True)
class Trial:
    resultCollect: ResultCollect
    hyperBoundNG: Dict[str, List[float]]
    hyperBoundFixed: Dict[str, List[float]]
    trial: int
    name: str

    @classmethod
    def create(cls, trial_dict: Dict[str, Any]) -> 'Trial':
        return cls(
            resultCollect=ResultCollect.create(trial_dict['resultCollect']),
            hyperBoundNG=trial_dict['hyperBoundNG'],
            hyperBoundFixed=trial_dict['hyperBoundFixed'],
            trial=trial_dict['trial'],
            name=trial_dict['name']
        )

@dataclass(frozen=True)
class Metadata:
    hyper_fixed: bool
    bootstrap: Optional[Any]
    refresh: bool
    train_timestamp: float
    cores: int
    iterations: int
    trials: int
    intercept: bool
    intercept_sign: str
    nevergrad_algo: str
    ts_validation: bool
    add_penalty_factor: bool
    hyper_updated: Dict[str, List[float]]

@dataclass(frozen=True)
class OutputModels:
    trials: List[Trial]
    metadata: Metadata
    seed: int
    __class__: str = "robyn_models"

    @classmethod
    def create(cls, output_models_dict: Dict[str, Any]) -> 'OutputModels':
        trials = [
            Trial(
                resultCollect=trial['resultCollect'],
                hyperBoundNG=trial['hyperBoundNG'],
                hyperBoundFixed=trial['hyperBoundFixed'],
                trial=trial['trial'],
                name=trial['name']
            )
            for trial in output_models_dict['trials']
        ]

        metadata = Metadata(
            hyper_fixed=output_models_dict['metadata']['hyper_fixed'],
            bootstrap=output_models_dict['metadata'].get('bootstrap'),
            refresh=output_models_dict['metadata']['refresh'],
            train_timestamp=output_models_dict['metadata']['train_timestamp'],
            cores=output_models_dict['metadata']['cores'],
            iterations=output_models_dict['metadata']['iterations'],
            trials=output_models_dict['metadata']['trials'],
            intercept=output_models_dict['metadata']['intercept'],
            intercept_sign=output_models_dict['metadata']['intercept_sign'],
            nevergrad_algo=output_models_dict['metadata']['nevergrad_algo'],
            ts_validation=output_models_dict['metadata']['ts_validation'],
            add_penalty_factor=output_models_dict['metadata']['add_penalty_factor'],
            hyper_updated=output_models_dict['metadata']['hyper_updated']
        )

        return cls(
            trials=trials,
            metadata=metadata,
            seed=output_models_dict['seed']
        )
