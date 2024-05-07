import pytest
import pandas as pd
from src.robyn.inputs import robyn_inputs, prophet_decomp, robyn_engineering

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_inputs():
    # Test case 1: Running robyn_inputs() for the first time
    input_collect = robyn_inputs(
        dt_input=None,
        dep_var=None,
        dep_var_type=None,
        date_var="auto",
        paid_media_spends=None,
        paid_media_vars=None,
        paid_media_signs=None,
        organic_vars=None,
        organic_signs=None,
        context_vars=None,
        context_signs=None,
        factor_vars=None,
        dt_holidays=None,
        prophet_vars=None,
        prophet_signs=None,
        prophet_country=None,
        adstock=None,
        hyperparameters=None,
        window_start=None,
        window_end=None,
        calibration_input=None,
        json_file=None,
        InputCollect=None
    )
    assert isinstance(input_collect, dict)
    assert 'dt_input' in input_collect
    assert 'dt_holidays' in input_collect
    assert 'date_var' in input_collect
    assert 'dayInterval' in input_collect
    assert 'intervalType' in input_collect
    assert 'dep_var' in input_collect
    assert 'dep_var_type' in input_collect
    assert 'prophet_vars' in input_collect
    assert 'prophet_signs' in input_collect
    assert 'prophet_country' in input_collect
    assert 'context_vars' in input_collect

    # Test case 2: Running robyn_inputs() with json_file
    input_collect = robyn_inputs(
        dt_input=None,
        dep_var=None,
        dep_var_type=None,
        date_var="auto",
        paid_media_spends=None,
        paid_media_vars=None,
        paid_media_signs=None,
        organic_vars=None,
        organic_signs=None,
        context_vars=None,
        context_signs=None,
        factor_vars=None,
        dt_holidays=None,
        prophet_vars=None,
        prophet_signs=None,
        prophet_country=None,
        adstock=None,
        hyperparameters=None,
        window_start=None,
        window_end=None,
        calibration_input=None,
        json_file="path/to/json_file.json",
        InputCollect=None
    )
    assert isinstance(input_collect, dict)
    assert 'dt_input' in input_collect
    assert 'dt_holidays' in input_collect
    assert 'date_var' in input_collect
    assert 'dayInterval' in input_collect
    assert 'intervalType' in input_collect
    assert 'dep_var' in input_collect
    assert 'dep_var_type' in input_collect
    assert 'prophet_vars' in input_collect
    assert 'prophet_signs' in input_collect
    assert 'prophet_country' in input_collect
    assert 'context_vars' in input_collect

    # Add more test cases as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_inputs():
    # Test case 1: Running robyn_inputs() for the first time
    input_collect = robyn_inputs(
        dt_input=None,
        dep_var=None,
        dep_var_type=None,
        date_var="auto",
        paid_media_spends=None,
        paid_media_vars=None,
        paid_media_signs=None,
        organic_vars=None,
        organic_signs=None,
        context_vars=None,
        context_signs=None,
        factor_vars=None,
        dt_holidays=None,
        prophet_vars=None,
        prophet_signs=None,
        prophet_country=None,
        adstock=None,
        hyperparameters=None,
        window_start=None,
        window_end=None,
        calibration_input=None,
        json_file=None,
        InputCollect=None
    )
    assert isinstance(input_collect, dict)
    assert 'dt_input' in input_collect
    assert 'dt_holidays' in input_collect
    assert 'date_var' in input_collect
    assert 'dayInterval' in input_collect
    assert 'dep_var' in input_collect
    assert 'dep_var_type' in input_collect
    assert 'prophet_vars' in input_collect
    assert 'prophet_signs' in input_collect
    assert 'prophet_country' in input_collect
    assert 'context_vars' in input_collect

    # Test case 2: Running robyn_inputs() with json_file
    input_collect = robyn_inputs(
        dt_input=None,
        dep_var=None,
        dep_var_type=None,
        date_var="auto",
        paid_media_spends=None,
        paid_media_vars=None,
        paid_media_signs=None,
        organic_vars=None,
        organic_signs=None,
        context_vars=None,
        context_signs=None,
        factor_vars=None,
        dt_holidays=None,
        prophet_vars=None,
        prophet_signs=None,
        prophet_country=None,
        adstock=None,
        hyperparameters=None,
        window_start=None,
        window_end=None,
        calibration_input=None,
        json_file="path/to/json_file.json",
        InputCollect=None
    )
    assert isinstance(input_collect, dict)
    assert 'dt_input' in input_collect
    assert 'dt_holidays' in input_collect
    assert 'date_var' in input_collect
    assert 'dayInterval' in input_collect
    assert 'dep_var' in input_collect
    assert 'dep_var_type' in input_collect
    assert 'prophet_vars' in input_collect
    assert 'prophet_signs' in input_collect
    assert 'prophet_country' in input_collect
    assert 'context_vars' in input_collect

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_engineering():
    # Test case 1: Running robyn_engineering() with valid input
    x = {
        'dt_input': pd.DataFrame(),
        'unused_vars': [],
        'paid_media_vars': [],
        'paid_media_spends': [],
        'factor_vars': [],
        'rollingWindowStartWhich': 1,
        'rollingWindowEndWhich': 10,
        'date_var': 'date',
        'dep_var': 'sales',
        'mediaVarCount': 0,
        'channel': []
    }
    mod_nls_collect, plot_nls_collect, yhat_collect = robyn_engineering(x, quiet=False)
    assert isinstance(mod_nls_collect, pd.DataFrame)
    assert isinstance(plot_nls_collect, pd.DataFrame)
    assert isinstance(yhat_collect, pd.DataFrame)

    # Test case 2: Running robyn_engineering() with empty input
    x = {}
    mod_nls_collect, plot_nls_collect, yhat_collect = robyn_engineering(x, quiet=False)
    assert mod_nls_collect is None
    assert plot_nls_collect is None
    assert yhat_collect is None

@pytest.mark.xfail(raises=NotImplementedError)
def test_prophet_decomp():
    # Test case 1: Running prophet_decomp with default parameters
    dt_transform = pd.DataFrame({
        'ds': pd.date_range(start='1/1/2022', periods=10),
        'dep_var': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    dt_holidays = pd.DataFrame({
        'ds': pd.to_datetime(['2022-01-01']),
        'holiday': ['New Year']
    })
    prophet_country = 'US'
    prophet_vars = ['trend', 'holiday', 'season', 'monthly', 'weekday']
    prophet_signs = [1, 1, 1, 1, 1]
    factor_vars = ['factor1', 'factor2']
    context_vars = ['context1', 'context2']
    organic_vars = ['organic1', 'organic2']
    paid_media_spends = ['media1', 'media2']
    intervalType = 'daily'
    dayInterval = 1

    result = prophet_decomp(
        dt_transform,
        dt_holidays,
        prophet_country,
        prophet_vars,
        prophet_signs,
        factor_vars,
        context_vars,
        organic_vars,
        paid_media_spends,
        intervalType,
        dayInterval
    )

    assert isinstance(result, pd.DataFrame)
    assert 'ds' in result.columns
    assert 'dep_var' in result.columns
    assert 'trend' in result.columns
    assert 'season' in result.columns
    assert 'monthly' in result.columns
    assert 'weekday' in result.columns
    assert 'holiday' in result.columns

    # Test case 2: Running prophet_decomp with custom parameters
    custom_params = {
        'yearly_seasonality': False,
        'weekly_seasonality': True,
        'daily_seasonality': True
    }

    result = prophet_decomp(
        dt_transform,
        dt_holidays,
        prophet_country,
        prophet_vars,
        prophet_signs,
        factor_vars,
        context_vars,
        organic_vars,
        paid_media_spends,
        intervalType,
        dayInterval,
        custom_params
    )

    assert isinstance(result, pd.DataFrame)
    assert 'ds' in result.columns
    assert 'dep_var' in result.columns
    assert 'trend' in result.columns
    assert 'season' in result.columns
    assert 'monthly' in result.columns
    assert 'weekday' in result.columns
    assert 'holiday' in result.columns

    # Add more test cases as needed

# Run the tests
pytest.main(["-v", "--tb=line", "~/Documents/GitHub/Robyn/python/tests/test_inputs.py"])