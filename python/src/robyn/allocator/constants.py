"""Constants for the budget allocator module."""

# Optimization Scenarios
SCENARIO_MAX_RESPONSE = "max_response"
SCENARIO_TARGET_EFFICIENCY = "target_efficiency"

# Optimization Algorithms
ALGO_SLSQP_AUGLAG = "SLSQP_AUGLAG"
ALGO_MMA_AUGLAG = "MMA_AUGLAG"

# Constraint Modes
CONSTRAINT_MODE_EQ = "eq"
CONSTRAINT_MODE_INEQ = "ineq"

# Dependent Variable Types
DEP_VAR_TYPE_REVENUE = "revenue"
DEP_VAR_TYPE_CONVERSION = "conversion"

# Default Values
DEFAULT_MAX_EVAL = 100000
DEFAULT_CONSTRAINT_MULTIPLIER = 3.0
DEFAULT_CHANNEL_CONSTRAINT_LOW = 0.7
DEFAULT_CHANNEL_CONSTRAINT_UP = 1.2

# Date Range Options
DATE_RANGE_ALL = "all"
DATE_RANGE_LAST = "last"
