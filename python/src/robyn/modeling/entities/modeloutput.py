from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelOutput:
    """
    A data object to store metadata about the output models.

    Attributes:
        metadata (Dict[str, any]): A dictionary containing metadata about the output models.
    """

    def __init__(self):
        self.metadata: Dict[str, any] = {}
        self.trials: Dict[str, any] = {}

    def update_metadata(self, 
                        hyper_fixed: Dict[str, any], 
                        bootstrap: bool, 
                        refresh: bool, 
                        train_timestamp: float, 
                        cores: int, 
                        iterations: int, 
                        trials: int, 
                        intercept: bool, 
                        intercept_sign: str, 
                        nevergrad_algo: str, 
                        ts_validation: bool, 
                        add_penalty_factor: bool, 
                        hyper_updated: Dict[str, any]) -> None:
        """
        Update the metadata dictionary.

        Args:
            hyper_fixed (Dict[str, any]): Hyperparameters that are fixed.
            bootstrap (bool): Whether to use bootstrap or not.
            refresh (bool): Whether to refresh or not.
            train_timestamp (float): Timestamp of the training.
            cores (int): Number of cores used.
            iterations (int): Number of iterations.
            trials (int): Number of trials.
            intercept (bool): Whether to include an intercept or not.
            intercept_sign (str): Sign of the intercept.
            nevergrad_algo (str): Nevergrad algorithm used.
            ts_validation (bool): Whether to use time series validation or not.
            add_penalty_factor (bool): Whether to add a penalty factor or not.
            hyper_updated (Dict[str, any]): Updated hyperparameters.
        """
        self.metadata['hyper_fixed'] = hyper_fixed
        self.metadata['bootstrap'] = bootstrap
        self.metadata['refresh'] = refresh
        self.metadata['train_timestamp'] = train_timestamp
        self.metadata['cores'] = cores
        self.metadata['iterations'] = iterations
        self.metadata['trials'] = trials
        self.metadata['intercept'] = intercept
        self.metadata['intercept_sign'] = intercept_sign
        self.metadata['nevergrad_algo'] = nevergrad_algo
        self.metadata['ts_validation'] = ts_validation
        self.metadata['add_penalty_factor'] = add_penalty_factor
        self.metadata['hyper_updated'] = hyper_updated
