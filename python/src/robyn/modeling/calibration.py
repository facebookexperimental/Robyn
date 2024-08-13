# pyre-strict

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from robyn.analysis.transformation import AdstockSaturationTransformation
from robyn.modeling.entities.calibration_result import CalibrationResult
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.enums import AdstockType

class Calibration:
    def __init__(
        self,
        calibration_input: CalibrationInput,
        mmmdata: MMMData,
        dayInterval: int,
        xDecompVec: pd.DataFrame,
        coefs: pd.Series,
        hypParamSam: Dict[str, float],
        wind_start: int = 1,
        wind_end: Optional[int] = None,
        adstock: Optional[AdstockType] = None
    ) -> None:
        self.calibration_input: CalibrationInput = calibration_input
        self.mmmdata: MMMData = mmmdata
        self.dayInterval: int = dayInterval
        self.xDecompVec: pd.DataFrame = xDecompVec
        self.coefs: pd.Series = coefs
        self.hypParamSam: Dict[str, float] = hypParamSam
        self.wind_start: int = wind_start
        self.wind_end: Optional[int] = wind_end
        self.adstock: Optional[AdstockType] = adstock
        self._transformer: AdstockSaturationTransformation = AdstockSaturationTransformation()

    def calibrate(self) -> CalibrationResult:
        calibration_studies: List[Dict[str, Any]] = []

        for channel, channel_data in self.calibration_input.channel_data.items():
            study_start: datetime = channel_data.lift_start_date
            study_end: datetime = channel_data.lift_end_date
            
            # Extract relevant model data for the study period
            model_data: pd.DataFrame = self._extract_model_data(study_start, study_end)
            
            # Calculate immediate and carryover effects
            immediate_effect, carryover_effect = self._calculate_channel_effects(channel, model_data)
            
            # Sum up effects
            total_effect: np.ndarray = immediate_effect + carryover_effect
            
            # Scale effect to match study duration
            scaled_effect: np.ndarray = self._scale_effect(total_effect, study_start, study_end)
            
            # Calculate metrics
            decomp_abs: float = np.sum(scaled_effect)
            total_decomp: float = np.sum(self.xDecompVec[channel])
            effect_percentage: float = (decomp_abs / total_decomp) * 100 if total_decomp != 0 else 0
            
            # Calculate MAPE
            mape: float = self._calculate_mape(channel_data.lift_abs, decomp_abs)
            
            calibration_studies.append({
                "lift_media": channel,
                "lift_start": study_start,
                "lift_end": study_end,
                "lift_abs": channel_data.lift_abs,
                "decomp_start": study_start,
                "decomp_end": study_end,
                "decomp_abs_scaled": decomp_abs,
                "decomp_abs_total_scaled": total_decomp,
                "calibrated_pct": effect_percentage,
                "mape_lift": mape
            })

        # Create CalibrationResult from the collected data
        return CalibrationResult.from_dataframe(pd.DataFrame(calibration_studies))

    def _extract_model_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        # Extract relevant model data for the given date range
        mask: pd.Series = (self.mmmdata.data[self.mmmdata.mmmdata_spec.date_var] >= start_date) & \
               (self.mmmdata.data[self.mmmdata.mmmdata_spec.date_var] <= end_date)
        return self.mmmdata.data.loc[mask]

    def _calculate_channel_effects(self, channel: str, model_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Apply adstock transformation
        adstocked_data: np.ndarray = self._apply_adstock(model_data[channel].values, channel)
        
        # Apply saturation transformation
        saturated_data: np.ndarray = self._apply_saturation(adstocked_data, channel)
        
        # Calculate immediate and carryover effects
        immediate_effect: np.ndarray = saturated_data * self.coefs[channel]
        carryover_effect: np.ndarray = (adstocked_data - model_data[channel].values) * self.coefs[channel]
        
        return immediate_effect, carryover_effect

    def _apply_adstock(self, data: np.ndarray, channel: str) -> np.ndarray:
        if self.adstock == AdstockType.GEOMETRIC:
            return self._transformer.adstock_geometric(data.tolist(), self.hypParamSam[f"{channel}_theta"])["x_decayed"]
        elif self.adstock == AdstockType.WEIBULL:
            return self._transformer.adstock_weibull(data.tolist(), self.hypParamSam[f"{channel}_shape"], self.hypParamSam[f"{channel}_scale"])["x_decayed"]
        else:
            raise ValueError("Unsupported adstock type")

    def _apply_saturation(self, data: np.ndarray, channel: str) -> np.ndarray:
        return self._transformer.saturation_hill(data, self.hypParamSam[f"{channel}_alpha"], self.hypParamSam[f"{channel}_gamma"])

    def _scale_effect(self, effect: np.ndarray, start_date: datetime, end_date: datetime) -> np.ndarray:
        # Scale effect to match study duration
        study_duration: int = (end_date - start_date).days + 1
        return effect * (study_duration / len(effect))

    def _calculate_mape(self, actual: float, predicted: float) -> float:
        return np.mean(np.abs((actual - predicted) / actual)) * 100 if actual != 0 else np.inf