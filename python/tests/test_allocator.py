import pytest
import numpy as np
import pandas as pd
from python import allocator

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_allocator():
    # Define the input parameters for the function
    robyn_object = None
    select_build = 0
    InputCollect = None
    OutputCollect = None
    select_model = None
    json_file = None
    scenario = "max_response"
    total_budget = None
    target_value = None
    date_range = None
    channel_constr_low = None
    channel_constr_up = None
    channel_constr_multiplier = 3
    optim_algo = "SLSQP_AUGLAG"
    maxeval = 100000
    constr_mode = "eq"
    plots = True
    plot_folder = None
    plot_folder_sub = None
    export = True
    quiet = False
    ui = False

    # Call the function
    result = allocator.robyn_allocator(robyn_object, select_build, InputCollect, OutputCollect, select_model, json_file,
                             scenario, total_budget, target_value, date_range, channel_constr_low, channel_constr_up,
                             channel_constr_multiplier, optim_algo, maxeval, constr_mode, plots, plot_folder,
                             plot_folder_sub, export, quiet, ui)

    # Perform assertions on the output
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == len(mediaSpendSorted)
    assert result.columns.tolist() == mediaSpendSorted

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_fx_objective():
    # Define the input parameters for the function
    x = np.array([1, 2, 3, 4, 5])
    coeff = 2
    alpha = 0.5
    inflexion = 0.1
    x_hist_carryover = None
    get_sum = False
    SIMPLIFY = True

    # Call the function
    result = allocator.fx_objective(x, coeff, alpha, inflexion, x_hist_carryover, get_sum, SIMPLIFY)

    # Perform assertions on the output
    expected_result = np.array([0.73575888, 0.27067057, 0.09957414, 0.03678794, 0.01353517])
    np.testing.assert_allclose(result, expected_result)

@pytest.mark.xfail(raises=NotImplementedError)
def test_optimize():
    # Define the input parameters for the function
    x0 = np.array([1, 2, 3, 4, 5])
    coeff = 2
    alpha = 0.5
    inflexion = 0.1
    x_hist_carryover = None
    total_budget = 10
    channel_constr_low = np.array([0, 0, 0, 0, 0])
    channel_constr_up = np.array([10, 10, 10, 10, 10])
    channel_constr_multiplier = 3
    optim_algo = "SLSQP_AUGLAG"
    maxeval = 100000
    constr_mode = "eq"

    # Call the function
    result = allocator.optimize(x0, coeff, alpha, inflexion, x_hist_carryover, total_budget, channel_constr_low,
                      channel_constr_up, channel_constr_multiplier, optim_algo, maxeval, constr_mode)

    # Perform assertions on the output
    expected_result = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(result, expected_result)

@pytest.mark.xfail(raises=NotImplementedError)
def test_fx_objective():
    # Define the input parameters for the function
    x = np.array([1, 2, 3, 4, 5])
    coeff = 2
    alpha = 0.5
    inflexion = 0.1
    x_hist_carryover = None
    get_sum = False
    SIMPLIFY = True

    # Call the function
    result = allocator.fx_objective(x, coeff, alpha, inflexion, x_hist_carryover, get_sum, SIMPLIFY)

    # Perform assertions on the output
    expected_result = np.array([0.73575888, 0.27067057, 0.09957414, 0.03678794, 0.01353517])
    np.testing.assert_allclose(result, expected_result)

@pytest.mark.xfail(raises=NotImplementedError)
def test_plot_robyn_allocator():
    # Define the input parameters for the function
    x = None
    args = []
    kwargs = {}

    # Call the function
    allocator.plot_robyn_allocator(x, *args, **kwargs)

    # Add assertions as needed

# Run the tests
pytest.main(["-v", "--tb=line", "~/Documents/GitHub/Robyn/python/python/test_allocator.py"])