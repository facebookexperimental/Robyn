from enum import auto, Enum


class NevergradAlgorithm(Enum):
    """
    Enumeration of available Nevergrad optimization algorithms.
    These algorithms are used in the hyperparameter optimization process.
    """

    DE = "DE"  # Differential Evolution
    TWO_POINTS_DE = "TwoPointsDE"  # Two-Points Differential Evolution
    ONE_PLUS_ONE = "OnePlusOne"  # One Plus One
    DOUBLE_FAST_GA_DISCRETE_ONE_PLUS_ONE = (
        "DoubleFastGADiscreteOnePlusOne"  # Double Fast GA Discrete One Plus One
    )
    DISCRETE_ONE_PLUS_ONE = "DiscreteOnePlusOne"  # Discrete One Plus One
    PORTFOLIO_DISCRETE_ONE_PLUS_ONE = (
        "PortfolioDiscreteOnePlusOne"  # Portfolio Discrete One Plus One
    )
    NAIVE_TBPSA = "NaiveTBPSA"  # Naive TBPSA
    CGA = "CGA"  # Compact Genetic Algorithm
    RANDOM_SEARCH = "RandomSearch"  # Random Search


class Models(Enum):
    """
    Enumeration of available model types.

    This enum can be expanded to include other model types as they are implemented.
    """

    RIDGE = auto()  # Ridge Regression
    # Add other model types as needed
