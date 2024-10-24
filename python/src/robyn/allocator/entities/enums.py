from enum import Enum, auto


class OptimizationScenario(str, Enum):
    """Available optimization scenarios for budget allocation."""

    MAX_RESPONSE = "max_response"  # Maximize response while keeping budget constant
    TARGET_EFFICIENCY = "target_efficiency"  # Optimize spend based on target ROAS/CPA


class ConstrMode(str, Enum):
    """Constraint modes for optimization."""

    EQUALITY = "eq"
    INEQUALITY = "ineq"


class UseCase(str, Enum):
    """Defines different use cases for allocation."""

    ALL_HISTORICAL_VEC = "all_historical_vec"
    SELECTED_HISTORICAL_VEC = "selected_historical_vec"
    TOTAL_METRIC_DEFAULT_RANGE = "total_metric_default_range"
    TOTAL_METRIC_SELECTED_RANGE = "total_metric_selected_range"
    UNIT_METRIC_DEFAULT_LAST_N = "unit_metric_default_last_n"
    UNIT_METRIC_SELECTED_DATES = "unit_metric_selected_dates"
