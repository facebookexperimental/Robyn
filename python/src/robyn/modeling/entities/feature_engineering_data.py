# robyn/modeling/entities/feature_engineering_data.py

import pandas as pd

class FeatureEngineeringInputData:
    """
    Class to encapsulate the input data for the feature engineering process.
    
    Attributes:
        dt_input (pd.DataFrame): The raw input data.
        date_var (str): The name of the date variable.
        dep_var (str): The name of the dependent variable.
        dep_var_type (str): The type of the dependent variable.
        paid_media_spends (list): List of paid media spend columns.
        paid_media_vars (list): List of paid media variable columns.
        paid_media_signs (list): List of signs for paid media variables.
        context_vars (list): List of context variable columns.
        context_signs (list): List of signs for context variables.
        organic_vars (list): List of organic variable columns.
        organic_signs (list): List of signs for organic variables.
        factor_vars (list): List of factor variables.
        dt_holidays (pd.DataFrame): The raw input holiday data.
        prophet_vars (list): List of prophet variables.
        prophet_signs (list): List of signs for prophet variables.
        prophet_country (str): The country for prophet holidays.
        adstock (str): The adstock type.
        hyperparameters (dict): The hyperparameters.
        window_start (str): The start date of the modeling period.
        window_end (str): The end date of the modeling period.
        calibration_input (pd.DataFrame): The calibration input data.
        json_file (str): The JSON file for importing previously exported inputs.
    """
    def __init__(self, dt_input=None, date_var=None, dep_var=None, dep_var_type=None, 
                 paid_media_spends=None, paid_media_vars=None, paid_media_signs=None, 
                 context_vars=None, context_signs=None, organic_vars=None, organic_signs=None, 
                 factor_vars=None, dt_holidays=None, prophet_vars=None, prophet_signs=None, 
                 prophet_country=None, adstock=None, hyperparameters=None, window_start=None, 
                 window_end=None, calibration_input=None, json_file=None):
        self.dt_input = dt_input
        self.date_var = date_var
        self.dep_var = dep_var
        self.dep_var_type = dep_var_type
        self.paid_media_spends = paid_media_spends
        self.paid_media_vars = paid_media_vars
        self.paid_media_signs = paid_media_signs
        self.context_vars = context_vars
        self.context_signs = context_signs
        self.organic_vars = organic_vars
        self.organic_signs = organic_signs
        self.factor_vars = factor_vars
        self.dt_holidays = dt_holidays
        self.prophet_vars = prophet_vars
        self.prophet_signs = prophet_signs
        self.prophet_country = prophet_country
        self.adstock = adstock
        self.hyperparameters = hyperparameters
        self.window_start = window_start
        self.window_end = window_end
        self.calibration_input = calibration_input
        self.json_file = json_file


class FeatureEngineeringOutputData:
    """
    Class to encapsulate the output data for the feature engineering process.
    
    Attributes:
        dt_mod (pd.DataFrame): The modified data after feature engineering.
        dt_modRollWind (pd.DataFrame): The modified data within the rolling window.
        modNLS (dict): The nonlinear model results.
    """
    def __init__(self, dt_mod=None, dt_modRollWind=None, modNLS=None):
        self.dt_mod = dt_mod
        self.dt_modRollWind = dt_modRollWind
        self.modNLS = modNLS