from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass(frozen=True)
class OutputCollect:
    resultHypParam: pd.DataFrame
    xDecompAgg: pd.DataFrame
    mediaVecCollect: pd.DataFrame
    xDecompVecCollect: pd.DataFrame
    resultCalibration: Optional[pd.DataFrame]
    allSolutions: List[str]
    allPareto: Dict[str, Any]
    calibration_constraint: float
    OutputModels: OutputModels
    cores: int
    iterations: int
    trials: List[Any]
    intercept_sign: str
    nevergrad_algo: str
    add_penalty_factor: bool
    seed: int
    UI: Optional[Any]
    pareto_fronts: int
    hyper_fixed: bool
    plot_folder: str
    convergence: Optional[Dict[str, Any]] = None
    ts_validation_plot: Optional[Any] = None
    selectID: Optional[str] = None
    hyper_updated: Optional[Dict[str, List[float]]] = None
    runTime: Optional[float] = None

    @classmethod
    def create(cls, output_collect_dict: Dict[str, Any]) -> 'OutputCollect':
        return cls(
            resultHypParam=output_collect_dict['resultHypParam'],
            xDecompAgg=output_collect_dict['xDecompAgg'],
            mediaVecCollect=output_collect_dict['mediaVecCollect'],
            xDecompVecCollect=output_collect_dict['xDecompVecCollect'],
            resultCalibration=output_collect_dict.get('resultCalibration'),
            allSolutions=output_collect_dict['allSolutions'],
            allPareto=output_collect_dict['allPareto'],
            calibration_constraint=output_collect_dict['calibration_constraint'],
            OutputModels=OutputModels.create(output_collect_dict['OutputModels']),
            cores=output_collect_dict['cores'],
            iterations=output_collect_dict['iterations'],
            trials=output_collect_dict['trials'],
            intercept_sign=output_collect_dict['intercept_sign'],
            nevergrad_algo=output_collect_dict['nevergrad_algo'],
            add_penalty_factor=output_collect_dict['add_penalty_factor'],
            seed=output_collect_dict['seed'],
            UI=output_collect_dict.get('UI'),
            pareto_fronts=output_collect_dict['pareto_fronts'],
            hyper_fixed=output_collect_dict['hyper_fixed'],
            plot_folder=output_collect_dict['plot_folder'],
            convergence=output_collect_dict.get('convergence'),
            ts_validation_plot=output_collect_dict.get('ts_validation_plot'),
            selectID=output_collect_dict.get('selectID'),
            hyper_updated=output_collect_dict.get('hyper_updated'),
            runTime=output_collect_dict.get('runTime')
        )

    def __post_init__(self):
        object.__setattr__(self, '__class__', 'robyn_outputs')
