from enum import Enum

class DependentVarType(str, Enum):
    """
    Enum class for dependent variable types.

    Attributes:
    REVENUE (str): Revenue type.
    CONVERSION (str): Conversion type.
    """
    REVENUE = "revenue"
    CONVERSION = "conversion"

class AdstockType(str, Enum):
    """
    Enum class for adstock types.

    Attributes:
    GEOMETRIC (str): Geometric adstock type.
    WEIBULL (str): Weibull adstock type.
    """
    GEOMETRIC = "geometric"
    WEIBULL = "weibull"

class SaturationType(str, Enum):
    """
    Enum class for saturation types.

    Attributes:
    MICHAELIS_MENTEN (str): Michaelis-Menten saturation type.
    LOGISTIC (str): Logistic saturation type.
    """
    MICHAELIS_MENTEN = "michaelis_menten"
    LOGISTIC = "logistic"

class ProphetVariableType(str, Enum):
    """
    Enum class for Prophet variable types.

    Attributes:
    TREND (str): Trend variable type.
    SEASON (str): Seasonal variable type.
    MONTHLY (str): Monthly variable type.
    WEEKDAY (str): Weekday variable type.
    HOLIDAY (str): Holiday variable type.
    """
    TREND = "trend"
    SEASON = "season"
    MONTHLY = "monthly"
    WEEKDAY = "weekday"
    HOLIDAY = "holiday"

class PaidMediaSigns(Enum):
    """
    Enum class for paid media signs.

    Attributes:
    POSITIVE (str): Positive sign.
    NEGATIVE (str): Negative sign.
    DEFAULT (str): Default sign.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    DEFAULT = "default"

class ProphetSigns(Enum):
    """
    Enum class for prophet signs.

    Attributes:
    POSITIVE (str): Positive sign.
    NEGATIVE (str): Negative sign.
    DEFAULT (str): Default sign.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    DEFAULT = "default"

class HyperParameterNames(Enum):
    """
    Enum class for hyperparameter names.

    Attributes:
    THETAS (str): Thetas hyperparameter.
    SHAPES (str): Shapes hyperparameter.
    SCALES (str): Scales hyperparameter.
    ALPHAS (str): Alphas hyperparameter.
    GAMMAS (str): Gammas hyperparameter.
    PENALTY (str): Penalty hyperparameter.
    """
    THETAS = "thetas"
    SHAPES = "shapes"
    SCALES = "scales"
    ALPHAS = "alphas"
    GAMMAS = "gammas"
    PENALTY = "penalty"

class CalibrationScope(Enum):
    IMMEDIATE = "immediate"
    TOTAL = "total"
