from dataclasses import dataclass
from typing import Optional


@dataclass
class TrialsConfig:
    """
    Configuration for model trials.

    This class defines the parameters for running multiple trials of the model.

    Attributes:
        trials (int): The number of trials to run. Each trial is an independent
            model fitting process with its own set of hyperparameters.
        iterations (int): The number of iterations to run for each trial. This
            determines how many times the optimization algorithm will attempt
            to improve the model within each trial.
    """

    trials: int
    iterations: int
