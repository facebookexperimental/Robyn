from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


#TODO: Logical grouping of attributes needs to be done. dataclasses should be created for nested logical grouped attributes.
@dataclass(frozen=True)
class ModelOutputCollection:
    # Group 1: Model Results
    # These are the primary outputs of the Robyn model
    resultHypParam: pd.DataFrame
    xDecompAgg: pd.DataFrame
    mediaVecCollect: pd.DataFrame
    xDecompVecCollect: pd.DataFrame
    resultCalibration: Optional[pd.DataFrame]

    # Group 2: Model Solutions
    # Information about the model solutions and Pareto optimization
    allSolutions: List[str]
    allPareto: Dict[str, Any]
    calibration_constraint: float
    pareto_fronts: int
    selectID: Optional[str]

    # Group 3: Model Configuration
    # Parameters and settings used for the model run
    modeloutput_collection: ModelOutputCollection
    cores: int
    iterations: int
    trials: List[Any]
    intercept_sign: str
    nevergrad_algo: str
    add_penalty_factor: bool
    seed: int
    hyper_fixed: bool
    hyper_updated: Optional[Dict[str, List[float]]]

    # Group 4: Output and Visualization
    # Information related to output storage and visualization
    plot_folder: str
    UI: Optional[Any]
    convergence: Optional[Dict[str, Any]]
    ts_validation_plot: Optional[Any]

    # Group 5: Performance Metrics
    # Metrics related to model performance and runtime
    runTime: Optional[float]

    @classmethod
    def create(cls, model_output_collection_dict: Dict[str, Any]) -> ModelOutputCollection:
        return cls(
            # Group 1: Model Results
            resultHypParam=model_output_collection_dict['resultHypParam'],
            xDecompAgg=model_output_collection_dict['xDecompAgg'],
            mediaVecCollect=model_output_collection_dict['mediaVecCollect'],
            xDecompVecCollect=model_output_collection_dict['xDecompVecCollect'],
            resultCalibration=model_output_collection_dict.get('resultCalibration'),

            # Group 2: Model Solutions
            allSolutions=model_output_collection_dict['allSolutions'],
            allPareto=model_output_collection_dict['allPareto'],
            calibration_constraint=model_output_collection_dict['calibration_constraint'],
            pareto_fronts=model_output_collection_dict['pareto_fronts'],
            selectID=model_output_collection_dict.get('selectID'),

            # Group 3: Model Configuration
            model_output_collection=model_output_collection_dict['model_output_collection'],
            cores=model_output_collection_dict['cores'],
            iterations=model_output_collection_dict['iterations'],
            trials=model_output_collection_dict['trials'],
            intercept_sign=model_output_collection_dict['intercept_sign'],
            nevergrad_algo=model_output_collection_dict['nevergrad_algo'],
            add_penalty_factor=model_output_collection_dict['add_penalty_factor'],
            seed=model_output_collection_dict['seed'],
            hyper_fixed=model_output_collection_dict['hyper_fixed'],
            hyper_updated=model_output_collection_dict.get('hyper_updated'),

            # Group 4: Output and Visualization
            plot_folder=model_output_collection_dict['plot_folder'],
            UI=model_output_collection_dict.get('UI'),
            convergence=model_output_collection_dict.get('convergence'),
            ts_validation_plot=model_output_collection_dict.get('ts_validation_plot'),

            # Group 5: Performance Metrics
            runTime=model_output_collection_dict.get('runTime')
        )

    def __post_init__(self):
        object.__setattr__(self, '__class__', 'robyn_outputs')

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Group 1: Model Results
            'resultHypParam': self.resultHypParam,
            'xDecompAgg': self.xDecompAgg,
            'mediaVecCollect': self.mediaVecCollect,
            'xDecompVecCollect': self.xDecompVecCollect,
            'resultCalibration': self.resultCalibration,

            # Group 2: Model Solutions
            'allSolutions': self.allSolutions,
            'allPareto': self.allPareto,
            'calibration_constraint': self.calibration_constraint,
            'pareto_fronts': self.pareto_fronts,
            'selectID': self.selectID,

            # Group 3: Model Configuration
            'model_output': self.model_output,
            'cores': self.cores,
            'iterations': self.iterations,
            'trials': self.trials,
            'intercept_sign': self.intercept_sign,
            'nevergrad_algo': self.nevergrad_algo,
            'add_penalty_factor': self.add_penalty_factor,
            'seed': self.seed,
            'hyper_fixed': self.hyper_fixed,
            'hyper_updated': self.hyper_updated,

            # Group 4: Output and Visualization
            'plot_folder': self.plot_folder,
            'UI': self.UI,
            'convergence': self.convergence,
            'ts_validation_plot': self.ts_validation_plot,

            # Group 5: Performance Metrics
            'runTime': self.runTime
        }
