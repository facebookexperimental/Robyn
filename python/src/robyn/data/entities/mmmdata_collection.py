#pyre-strict

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from pyre_extensions import strict

@strict
class MMMDataCollection:
    """Collection of data and parameters for MMM modeling"""
    
    # **Input Data and Model Configuration**
    mmmdata: MMMData
    """MMM data object"""
    holiday_data: Optional[HolidaysData]
    """Holiday data object"""
    adstock: AdStock
    """Adstock configuration"""
    hyperparameters: Hyperparameters
    """Hyperparameters for modeling"""
    calibration_input: Optional[List[CalibrationInput]]
    """Calibration input data"""
    
    # **Intermediate Data**
    dt_mod: Optional[pd.DataFrame]
    """Modified data table"""
    dt_modRollWind: Optional[pd.DataFrame]
    """Modified data table with rolling window"""
    xDecompAggPrev: Optional[pd.DataFrame]
    """Decomposition aggregate previous data"""
    
    # **Model Parameters**
    dayInterval: int
    """Day interval"""
    intervalType: str
    """Interval type"""
    mediaVarCount: int
    """Number of media variables"""
    exposure_vars: List[str]
    """Exposure variable names"""
    all_media: List[str]
    """All media variable names"""
    all_ind_vars: List[str]
    """All independent variable names"""
    factor_vars: Optional[List[str]]
    """Factor variable names"""
    unused_vars: List[str]
    """Unused variable names"""
    
    # **Time Window**
    window_start: datetime
    """Window start date"""
    rollingWindowStartWhich: int
    """Rolling window start index"""
    window_end: datetime
    """Window end date"""
    rollingWindowEndWhich: int
    """Rolling window end index"""
    rollingWindowLength: int
    """Rolling window length"""
    totalObservations: int
    """Total number of observations"""
    refreshAddedStart: datetime
    """Refresh added start date"""
    
    # **Custom Parameters**
    custom_params: Dict[str, Any]
    """Custom parameters dictionary"""
    
    @classmethod
    def update(cls, obj: 'MMMDataCollection', **kwargs) -> 'MMMDataCollection':
        """Update values of the MMMDataCollection object"""
        return cls(
            mmmdata=kwargs.get('mmmdata', obj.mmmdata),
            holiday_data=kwargs.get('holiday_data', obj.holiday_data),
            adstock=kwargs.get('adstock', obj.adstock),
            hyperparameters=kwargs.get('hyperparameters', obj.hyperparameters),
            calibration_input=kwargs.get('calibration_input', obj.calibration_input),
            dt_mod=kwargs.get('dt_mod', obj.dt_mod),
            dt_modRollWind=kwargs.get('dt_modRollWind', obj.dt_modRollWind),
            xDecompAggPrev=kwargs.get('xDecompAggPrev', obj.xDecompAggPrev),
            date_var=kwargs.get('date_var', obj.date_var),
            dayInterval=kwargs.get('dayInterval', obj.dayInterval),
            intervalType=kwargs.get('intervalType', obj.intervalType),
            mediaVarCount=kwargs.get('mediaVarCount', obj.mediaVarCount),
            exposure_vars=kwargs.get('exposure_vars', obj.exposure_vars),
            all_media=kwargs.get('all_media', obj.all_media),
            all_ind_vars=kwargs.get('all_ind_vars', obj.all_ind_vars),
            factor_vars=kwargs.get('factor_vars', obj.factor_vars),
            unused_vars=kwargs.get('unused_vars', obj.unused_vars),
            window_start=kwargs.get('window_start', obj.window_start),
            rollingWindowStartWhich=kwargs.get('rollingWindowStartWhich', obj.rollingWindowStartWhich),
            window_end=kwargs.get('window_end', obj.window_end),
            rollingWindowEndWhich=kwargs.get('rollingWindowEndWhich', obj.rollingWindowEndWhich),
            rollingWindowLength=kwargs.get('rollingWindowLength', obj.rollingWindowLength),
            totalObservations=kwargs.get('totalObservations', obj.totalObservations),
            refreshAddedStart=kwargs.get('refreshAddedStart', obj.refreshAddedStart),
            custom_params=kwargs.get('custom_params', obj.custom_params)
        )
