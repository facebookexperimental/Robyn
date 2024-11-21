# pyre-strict
# Enum classes for different types of variables, model parameters etc.

from enum import Enum
from typing import List


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
    Enumeration class representing different types of adstock models.
    Attributes:
        GEOMETRIC (str): Represents the geometric adstock model.
        WEIBULL (str): Represents the Weibull adstock model.
        WEIBULL_CDF (str): Represents the Weibull cumulative distribution function adstock model.
        WEIBULL_PDF (str): Represents the Weibull probability density function adstock model.
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
    """
    Enumeration representing the calibration scope.

    Attributes:
        IMMEDIATE (str): Represents the immediate calibration scope.
        TOTAL (str): Represents the total calibration scope.
    """

    IMMEDIATE = "immediate"
    TOTAL = "total"


class PlotType(Enum):
    """
    Enumeration of available plot types for the OnePagerReporter.
    """

    SPEND_EFFECT = "spend_effect"
    WATERFALL = "waterfall"
    FITTED_VS_ACTUAL = "fitted_vs_actual"
    BOOTSTRAP = "bootstrap"
    ADSTOCK = "adstock"
    IMMEDIATE_CARRYOVER = "immediate_carryover"
    RESPONSE_CURVES = "response_curves"
    DIAGNOSTIC = "diagnostic"
