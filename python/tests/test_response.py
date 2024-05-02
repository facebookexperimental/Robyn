import pytest
import numpy as np
import pandas as pd
from src.response import robyn_response, which_usecase

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_response():
    # Define the input parameters for the function
    InputCollect = None
    OutputCollect = None
    json_file = None
    robyn_object = None
    select_build = None
    select_model = None
    metric_name = None
    metric_value = None
    date_range = None
    dt_hyppar = None
    dt_coef = None
    quiet = False

    # Call the function
    result = robyn_response(InputCollect, OutputCollect, json_file, robyn_object, select_build, select_model,
                            metric_name, metric_value, date_range, dt_hyppar, dt_coef, quiet)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert 'metric_name' in result
    assert 'date' in result
    assert 'input_total' in result
    assert 'input_carryover' in result
    assert 'input_immediate' in result
    assert 'response_total' in result
    assert 'response_carryover' in result
    assert 'response_immediate' in result
    assert 'usecase' in result
    assert 'plot' in result

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_response():
    # Define the input parameters for the function
    InputCollect = None
    OutputCollect = None
    json_file = None
    robyn_object = None
    select_build = None
    select_model = None
    metric_name = None
    metric_value = None
    date_range = None
    dt_hyppar = None
    dt_coef = None
    quiet = False

    # Call the function
    result = robyn_response(InputCollect, OutputCollect, json_file, robyn_object, select_build, select_model,
                            metric_name, metric_value, date_range, dt_hyppar, dt_coef, quiet)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert 'metric_name' in result
    assert 'date' in result
    assert 'input_total' in result
    assert 'input_carryover' in result
    assert 'input_immediate' in result
    assert 'response_total' in result
    assert 'response_carryover' in result
    assert 'response_immediate' in result
    assert 'usecase' in result
    assert 'plot' in result

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_which_usecase():
    # Test case 1: pd.isnull(metric_value) and pd.isnull(date_range)
    metric_value = None
    date_range = None
    assert which_usecase(metric_value, date_range) == "all_historical_vec"

    # Test case 2: pd.isnull(metric_value) and not pd.isnull(date_range)
    metric_value = None
    date_range = [2022-01-01, 2022-12-31]
    assert which_usecase(metric_value, date_range) == "selected_historical_vec"

    # Test case 3: len(metric_value) == 1 and pd.isnull(date_range)
    metric_value = [1]
    date_range = None
    assert which_usecase(metric_value, date_range) == "total_metric_default_range"

    # Test case 4: len(metric_value) == 1 and not pd.isnull(date_range)
    metric_value = [1]
    date_range = [2022-01-01, 2022-12-31]
    assert which_usecase(metric_value, date_range) == "total_metric_selected_range"

    # Test case 5: len(metric_value) > 1 and pd.isnull(date_range)
    metric_value = [1, 2, 3]
    date_range = None
    assert which_usecase(metric_value, date_range) == "unit_metric_default_last_n"

    # Test case 6: len(metric_value) > 1 and not pd.isnull(date_range)
    metric_value = [1, 2, 3]
    date_range = [2022-01-01, 2022-12-31]
    assert which_usecase(metric_value, date_range) == "unit_metric_selected_dates"

    # Test case 7: len(date_range) == 1 and date_range[0] == "all"
    metric_value = [1, 2, 3]
    date_range = ["all"]
    assert which_usecase(metric_value, date_range) == "all_historical_vec"

# Run the tests
pytest.main(["-v", "--tb=line", "~/Documents/GitHub/Robyn/python/python/response.py"])