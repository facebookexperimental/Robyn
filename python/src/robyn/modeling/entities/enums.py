from enum import Enum, auto


class NevergradAlgorithm(Enum):
    """
    Enumeration of available Nevergrad optimization algorithms.

    These algorithms are used in the hyperparameter optimization process.
    """

    DE = auto()  # Differential Evolution
    TWO_POINTS_DE = auto()  # Two-Points Differential Evolution
    ONE_PLUS_ONE = auto()  # One Plus One
    DOUBLE_FAST_GA_DISCRETE_ONE_PLUS_ONE = auto()  # Double Fast GA Discrete One Plus One
    DISCRETE_ONE_PLUS_ONE = auto()  # Discrete One Plus One
    PORTFOLIO_DISCRETE_ONE_PLUS_ONE = auto()  # Portfolio Discrete One Plus One
    NAIVE_TBPSA = auto()  # Naive TBPSA
    CGA = auto()  # Compact Genetic Algorithm
    RANDOM_SEARCH = auto()  # Random Search


class Models(Enum):
    """
    Enumeration of available model types.

    This enum can be expanded to include other model types as they are implemented.
    """

    RIDGE = auto()  # Ridge Regression
    # Add other model types as needed
