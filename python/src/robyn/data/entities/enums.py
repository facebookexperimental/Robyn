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
    WEIBULL_CDF = "weibull_cdf"
    WEIBULL_PDF = "weibull_pdf"

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

class OrganicSigns(Enum):
    """
    Enum class for Organic variables signs.

    Attributes:
    POSITIVE (str): Positive sign.
    NEGATIVE (str): Negative sign.
    DEFAULT (str): Default sign.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    DEFAULT = "default"

class ContextSigns(Enum):
    """
    Enum class for Context variables signs.

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

class CalibrationScope(Enum):
    IMMEDIATE = "immediate"
    TOTAL = "total"
