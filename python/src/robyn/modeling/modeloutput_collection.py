from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class ModelOutputCollection:
    """
    A data class to store a collection of models data.

    Attributes:
        resultHypParam (Dict[str, Any]): Resulting hyperparameters.
        xDecompAgg (Dict[str, Any]): Decomposed aggregated data.
        mediaVecCollect (List[Any]): Media vector collection.
        xDecompVecCollect (List[Any]): Decomposed vector collection.
        resultCalibration (Optional[Dict[str, Any]]): Resulting calibration data.
        allSolutions (List[Any]): All solutions.
        allPareto (List[Any]): All Pareto fronts.
        calibration_constraint (Any): Calibration constraint.
        ModelsData (Dict[str, Any]): Models data.
        cores (int): Number of cores used.
        iterations (int): Number of iterations.
        trials (int): Number of trials.
        intercept_sign (str): Sign of the intercept.
        nevergrad_algo (str): Nevergrad algorithm used.
        add_penalty_factor (bool): Whether to add a penalty factor or not.
        seed (int): Seed used.
        UI (Optional[Any]): User interface data.
        pareto_fronts (List[Any]): Pareto fronts.
        hyper_fixed (Dict[str, Any]): Fixed hyperparameters.
        plot_folder (str): Plot folder path.
    """
    resultHypParam: Dict[str, Any]
    xDecompAgg: Dict[str, Any]
    mediaVecCollect: List[Any]
    xDecompVecCollect: List[Any]
    resultCalibration: Optional[Dict[str, Any]]
    allSolutions: List[Any]
    allPareto: List[Any]
    calibration_constraint: Any
    model_output: ModelOutput
    modelrun_trials_config: TrialsConfig
    cores: int
    intercept_sign: str
    nevergrad_algo: str
    seed: int
    UI: Optional[Any]
    pareto_fronts: List[Any]
    hyper_fixed: Dict[str, Any]
    plot_folder: str
