import numpy as np
import pandas as pd
import pytest
from robyn.modeling.pareto.pareto_utils import ParetoUtils


@pytest.fixture
def test_data():
    return pd.DataFrame(
        {
            "nrmse_test": [0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
            "nrmse_train": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            "decomp.rssd": [0.2, 0.3, 0.4, 0.5, 0.6, np.inf],
            "mape": [0.05, 0.1, 0.15, 0.2, 0.25, np.nan],
        }
    )


def test_calculate_errors_scores_default(test_data):
    result = ParetoUtils.calculate_errors_scores(test_data)
    expected = np.array([0.0, 0.1443376, 0.2886751, 0.4330127, 0.5773503, 0.4714045])
    np.testing.assert_almost_equal(result, expected, decimal=2)


def test_calculate_errors_scores_custom_balance(test_data):
    result = ParetoUtils.calculate_errors_scores(test_data, balance=[2, 1, 1])
    expected = np.array([0.0, 0.1545743, 0.3091487, 0.4637231, 0.6182975, 0.5590170])
    np.testing.assert_almost_equal(result, expected, decimal=2)


def test_calculate_errors_scores_ts_validation_false(test_data):
    result = ParetoUtils.calculate_errors_scores(test_data, ts_validation=False)
    expected = np.array([0.0, 0.1361372, 0.2722745, 0.4082483, 0.5443118, 0.4714045])
    np.testing.assert_almost_equal(result, expected, decimal=2)


def test_calculate_errors_scores_all_infinite():
    inf_data = pd.DataFrame(
        {
            "nrmse_test": [np.inf, np.inf],
            "nrmse_train": [np.inf, np.inf],
            "decomp.rssd": [np.inf, np.inf],
            "mape": [np.inf, np.inf],
        }
    )
    result = ParetoUtils.calculate_errors_scores(inf_data)
    expected = np.array([0.0, 0.0])
    np.testing.assert_almost_equal(result, expected, decimal=2)


def test_calculate_errors_scores_single_row():
    single_row = pd.DataFrame(
        {
            "nrmse_test": [0.1],
            "nrmse_train": [0.15],
            "decomp.rssd": [0.2],
            "mape": [0.05],
        }
    )
    result = ParetoUtils.calculate_errors_scores(single_row)
    expected = np.array(0.0763763)
    np.testing.assert_almost_equal(result, expected, decimal=7)


def test_min_max_norm():
    series = pd.Series([1, 2, 3, 4, 5])
    normalized = ParetoUtils.min_max_norm(series)
    expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    pd.testing.assert_series_equal(normalized, expected)


def test_min_max_norm_constant_values():
    series = pd.Series([5, 5, 5])
    normalized = ParetoUtils.min_max_norm(series)
    expected = pd.Series(
        [5, 5, 5]
    )  # Should return the same values since they are constant
    pd.testing.assert_series_equal(normalized, expected)


def test_min_max_norm_single_value():
    series = pd.Series([10])
    normalized = ParetoUtils.min_max_norm(series)
    expected = pd.Series([10])  # Should return the same value since there's only one
    pd.testing.assert_series_equal(normalized, expected)
