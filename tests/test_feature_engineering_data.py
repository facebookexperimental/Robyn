# tests/test_feature_engineering_data.py

import pytest
import sys 
sys.path.append("/Users/yijuilee/project_robyn/modelling/Robyn/python/src")
from robyn.modeling.entities.feature_engineering_data import FeatureEngineeringInputData

def test_feature_engineering_input_data_initialization():
    dt_input = None
    date_var = "date"
    dep_var = "revenue"
    dep_var_type = "revenue"
    paid_media_spends = ["tv_S", "ooh_S"]
    paid_media_vars = ["tv_S", "ooh_S"]
    paid_media_signs = ["positive", "positive"]
    context_vars = ["competitor_sales_B"]
    context_signs = ["positive"]
    organic_vars = ["newsletter"]
    organic_signs = ["positive"]
    factor_vars = ["events"]
    dt_holidays = None
    prophet_vars = ["trend", "season"]
    prophet_signs = ["default", "default"]
    prophet_country = "DE"
    adstock = "geometric"
    hyperparameters = None
    window_start = "2016-11-23"
    window_end = "2018-08-22"
    calibration_input = None
    json_file = None

    input_data = FeatureEngineeringInputData(
        dt_input, date_var, dep_var, dep_var_type, paid_media_spends, paid_media_vars,
        paid_media_signs, context_vars, context_signs, organic_vars, organic_signs,
        factor_vars, dt_holidays, prophet_vars, prophet_signs, prophet_country,
        adstock, hyperparameters, window_start, window_end, calibration_input, json_file
    )

    assert input_data.date_var == "date"
    assert input_data.dep_var == "revenue"
    assert input_data.paid_media_spends == ["tv_S", "ooh_S"]
    assert input_data.prophet_country == "DE"