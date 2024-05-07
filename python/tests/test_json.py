import pytest
from src.robyn.json import robyn_write, robyn_read, robyn_recreate, robyn_chain
from src.robyn.inputs import robyn_inputs
from src.robyn.outputs import robyn_outputs
import pandas as pd
import os
import re
import json

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_write_valid_inputs():
    # Define the input parameters for the function
    InputCollect = robyn_inputs()
    OutputCollect = robyn_outputs()
    select_model = "model1"
    dir = "/path/to/directory"
    export = True
    quiet = False
    pareto_df = pd.DataFrame({'solID': [1, 2, 3], 'cluster': [1, 2, 3]})

    # Call the function
    result = robyn_write(InputCollect, OutputCollect, select_model, dir, export, quiet, pareto_df)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert "InputCollect" in result
    assert "ModelsCollect" in result
    assert "ExportedModel" in result
    assert result["ExportedModel"]["select_model"] == select_model
    assert result["OutputCollect"]["all_sols"].equals(pareto_df)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_write_invalid_inputs():
    # Define the input parameters for the function
    InputCollect = "invalid_input"
    OutputCollect = "invalid_output"
    select_model = 123
    dir = "/path/to/nonexistent/directory"
    export = True
    quiet = False
    pareto_df = pd.DataFrame({'solID': [1, 2, 3], 'cluster': [1, 2, 3]})

    # Call the function and expect a ValueError
    with pytest.raises(ValueError):
        robyn_write(InputCollect, OutputCollect, select_model, dir, export, quiet, pareto_df)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_read_valid_inputs():
    # Define the input parameters for the function
    json_file = "/path/to/json/file.json"
    step = 1
    quiet = False

    # Call the function
    result = robyn_read(json_file, step, quiet)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert "InputCollect" in result
    assert "ModelsCollect" in result
    assert "ExportedModel" in result
    assert isinstance(result["InputCollect"], list)
    assert isinstance(result["ModelsCollect"], list)
    assert isinstance(result["ExportedModel"], list)
    assert len(result["InputCollect"]) > 0
    assert not quiet

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_read_invalid_inputs():
    # Define the input parameters for the function
    json_file = 123
    step = 1
    quiet = False

    # Call the function and expect a ValueError
    with pytest.raises(ValueError):
        robyn_read(json_file, step, quiet)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_read_nonexistent_file():
    # Define the input parameters for the function
    json_file = "/path/to/nonexistent/file.json"
    step = 1
    quiet = False

    # Call the function and expect a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        robyn_read(json_file, step, quiet)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_read():
    # Define the input parameters for the function
    x = {
        'InputCollect': {
            'date_var': '2022-01-01',
            'dep_var': 'sales',
            'dep_var_type': 'continuous',
            'paid_media_vars': ['tv', 'radio'],
            'paid_media_spends': ['1000', '500'],
            'context_vars': ['weather', 'holiday'],
            'organic_vars': ['social_media', 'email'],
            'prophet': True,
            'unused_vars': ['var1', 'var2'],
            'window_start': '2022-01-01',
            'window_end': '2022-01-31',
            'rollingWindowStartWhich': 1,
            'rollingWindowEndWhich': 31,
            'intervalType': 'day',
            'calibration_input': None,
            'custom_params': {'param1': 0.5, 'param2': 0.8},
            'adstock': 0.5
        },
        'ExportedModel': None
    }

    # Call the function
    result = robyn_read(x)

    # Perform assertions on the output
    assert isinstance(result, pd.Series)
    assert len(result) == 0

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_write_valid_inputs():
    # Define the input parameters for the function
    InputCollect = robyn_inputs()
    OutputCollect = robyn_outputs()
    select_model = "model1"
    dir = "/path/to/directory"
    export = True
    quiet = False
    pareto_df = pd.DataFrame({'solID': [1, 2, 3], 'cluster': [1, 2, 3]})

    # Call the function
    result = robyn_write(InputCollect, OutputCollect, select_model, dir, export, quiet, pareto_df)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert "InputCollect" in result
    assert "ModelsCollect" in result
    assert "ExportedModel" in result
    assert result["ExportedModel"]["select_model"] == select_model
    assert result["OutputCollect"]["all_sols"].equals(pareto_df)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_write_invalid_inputs():
    # Define the input parameters for the function
    InputCollect = "invalid_input"
    OutputCollect = "invalid_output"
    select_model = 123
    dir = "/path/to/nonexistent/directory"
    export = True
    quiet = False
    pareto_df = pd.DataFrame({'solID': [1, 2, 3], 'cluster': [1, 2, 3]})

    # Call the function and expect a ValueError
    with pytest.raises(ValueError):
        robyn_write(InputCollect, OutputCollect, select_model, dir, export, quiet, pareto_df)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_recreate_valid_inputs():
    # Define the input parameters for the function
    json_file = "/path/to/json/file.json"
    quiet = False
    args = ()
    kwargs = {}

    # Call the function
    result = robyn_recreate(json_file, quiet, *args, **kwargs)

    # Perform assertions on the output
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert isinstance(result[1], dict)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_recreate_with_InputCollect():
    # Define the input parameters for the function
    json_file = "/path/to/json/file.json"
    quiet = False
    args = ()
    kwargs = {'InputCollect': {'input_param': 'value'}}

    # Call the function
    result = robyn_recreate(json_file, quiet, *args, **kwargs)

    # Perform assertions on the output
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert isinstance(result[1], dict)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_recreate_invalid_json_file():
    # Define the input parameters for the function
    json_file = "/path/to/nonexistent/json/file.json"
    quiet = False
    args = ()
    kwargs = {}

    # Call the function and expect a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        robyn_recreate(json_file, quiet, *args, **kwargs)

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_chain_valid_inputs():
    # Define the input parameters for the function
    json_file = "/path/to/json/file.json"

    # Call the function
    result = robyn_chain(json_file)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert "json_files" in result
    assert "chain" in result
    assert len(result["chain"]) == 2
    assert len(result["json_files"]) == len(result) - 2

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_chain_invalid_inputs():
    # Define the input parameters for the function
    json_file = "/path/to/nonexistent/json/file.json"

    # Call the function and expect a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        robyn_chain(json_file)

# Run the tests
pytest.main(["-v", "--tb=line", "~/Documents/GitHub/Robyn/python/python/tests/test_json.py"])