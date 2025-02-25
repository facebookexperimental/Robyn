import pytest
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from robyn.modeling.ridge.ridge_data_builder import RidgeDataBuilder


def original_geometric_adstock(x: pd.Series, theta: float) -> pd.Series:
    y = x.copy()
    for i in range(1, len(x)):
        y.iloc[i] += theta * y.iloc[i - 1]
    return y


@pytest.mark.parametrize("theta", [0, 0.5, 0.8, 1])
def test_geometric_adstock(theta):
    x = pd.Series(np.random.rand(10_000))  # Random test data

    # Instantiate the RidgeDataBuilder object (without requiring real data)
    dummy_data = None
    ridge_builder = RidgeDataBuilder(dummy_data, dummy_data)

    # Call the actual function from the RidgeDataBuilder instance
    optimized = ridge_builder._geometric_adstock(x, theta)

    # Compute the expected output using the original function
    original = original_geometric_adstock(x, theta)

    assert np.allclose(
        original, optimized, atol=1e-6
    ), f"Mismatch found for theta={theta}"
