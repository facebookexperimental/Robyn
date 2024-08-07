#pyre-strict

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.hyperparameters import Hyperparameters

@dataclass(frozen=True)
class IntermediateData: #TODO temparory grouped as IntermediateData. Update accordingly 
    dt_mod: Optional[pd.DataFrame]
    dt_modRollWind: Optional[pd.DataFrame]
    xDecompAggPrev: Optional[pd.DataFrame]

@dataclass(frozen=True)
class ModelParameters:
    dayInterval: int
    intervalType: str
    mediaVarCount: int
    exposure_vars: List[str]
    all_media: List[str]
    all_ind_vars: List[str]
    factor_vars: Optional[List[str]]
    unused_vars: List[str]

@dataclass(frozen=True)
class TimeWindow:
    window_start: datetime
    rollingWindowStartWhich: int
    window_end: datetime
    rollingWindowEndWhich: int
    rollingWindowLength: int
    totalObservations: int
    refreshAddedStart: datetime

@dataclass(frozen=True)
class MMMDataCollection:
    """Collection of data and parameters for MMM modeling"""
    
    # Input Data and Model Configuration
    mmmdata: MMMData
    holiday_data: Optional[HolidaysData]
    adstock: AdstockType
    hyperparameters: Hyperparameters
    calibration_input: Optional[List[CalibrationInput]]
    
    # Intermediate Data
    intermediate_data: IntermediateData
    
    # Model Parameters
    model_parameters: ModelParameters
    
    # Time Window
    time_window: TimeWindow
    
    # Custom Parameters
    custom_params: Dict[str, Any]

    @classmethod
    def update(cls, obj: 'MMMDataCollection', **kwargs) -> 'MMMDataCollection':
        """Update values of the MMMDataCollection object"""
        return cls(
            mmmdata=kwargs.get('mmmdata', obj.mmmdata),
            holiday_data=kwargs.get('holiday_data', obj.holiday_data),
            adstock=kwargs.get('adstock', obj.adstock),
            hyperparameters=kwargs.get('hyperparameters', obj.hyperparameters),
            calibration_input=kwargs.get('calibration_input', obj.calibration_input),
            intermediate_data=IntermediateData(
                dt_mod=kwargs.get('dt_mod', obj.intermediate_data.dt_mod),
                dt_modRollWind=kwargs.get('dt_modRollWind', obj.intermediate_data.dt_modRollWind),
                xDecompAggPrev=kwargs.get('xDecompAggPrev', obj.intermediate_data.xDecompAggPrev)
            ),
            model_parameters=ModelParameters(
                dayInterval=kwargs.get('dayInterval', obj.model_parameters.dayInterval),
                intervalType=kwargs.get('intervalType', obj.model_parameters.intervalType),
                mediaVarCount=kwargs.get('mediaVarCount', obj.model_parameters.mediaVarCount),
                exposure_vars=kwargs.get('exposure_vars', obj.model_parameters.exposure_vars),
                all_media=kwargs.get('all_media', obj.model_parameters.all_media),
                all_ind_vars=kwargs.get('all_ind_vars', obj.model_parameters.all_ind_vars),
                factor_vars=kwargs.get('factor_vars', obj.model_parameters.factor_vars),
                unused_vars=kwargs.get('unused_vars', obj.model_parameters.unused_vars)
            ),
            time_window=TimeWindow(
                window_start=kwargs.get('window_start', obj.time_window.window_start),
                rollingWindowStartWhich=kwargs.get('rollingWindowStartWhich', obj.time_window.rollingWindowStartWhich),
                window_end=kwargs.get('window_end', obj.time_window.window_end),
                rollingWindowEndWhich=kwargs.get('rollingWindowEndWhich', obj.time_window.rollingWindowEndWhich),
                rollingWindowLength=kwargs.get('rollingWindowLength', obj.time_window.rollingWindowLength),
                totalObservations=kwargs.get('totalObservations', obj.time_window.totalObservations),
                refreshAddedStart=kwargs.get('refreshAddedStart', obj.time_window.refreshAddedStart)
            ),
            custom_params=kwargs.get('custom_params', obj.custom_params)
        )
