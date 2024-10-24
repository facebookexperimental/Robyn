# pyre-strict
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
import pandas as pd
from typing import List, Tuple, Set, Optional, Dict, Union
from robyn.data.entities.enums import CalibrationScope, DependentVarType


class CalibrationInputValidation(Validation):
    def __init__(
        self,
        mmmdata: MMMData,
        calibration_input: CalibrationInput,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> None:
        self.mmmdata = mmmdata
        self.calibration_input = calibration_input
        self.window_start = window_start
        self.window_end = window_end
        self.valid_channels: Set[str] = set(self.mmmdata.mmmdata_spec.paid_media_spends)

    def check_obj_weights(self, objective_weights: List[float], refresh: bool) -> ValidationResult:
        """Check the objective weights for validity."""
        if objective_weights is None:
            if refresh:
                objective_weights = [0, 1, 1]  # Default weights for refresh
            else:
                return ValidationResult(status=True, error_details={}, error_message="")

        error_details: Dict[str, str] = {}
        error_messages: List[str] = []

        if len(objective_weights) not in [2, 3]:
            error_details["length"] = f"Expected 2 or 3 objective weights, got {len(objective_weights)}"
            error_messages.append("Invalid number of objective weights.")

        if any(weight < 0 or weight > 10 for weight in objective_weights):
            error_details["range"] = "Objective weights must be >= 0 & <= 10"
            error_messages.append("Objective weights out of valid range.")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _validate_channel_exists(self, channel_key: Tuple[str, ...]) -> ValidationResult:
        """Validate that all channels in the key exist in the data."""
        missing_channels = [ch for ch in channel_key if ch not in self.valid_channels]

        if missing_channels:
            msg = f"Channel(s) not found in data: {', '.join(missing_channels)}"
            return ValidationResult(status=False, error_details={channel_key: msg}, error_message=msg.lower())
        return ValidationResult(status=True, error_details={}, error_message="")

    def _check_date_range(self) -> ValidationResult:
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, data in self.calibration_input.channel_data.items():
            if data.lift_start_date < self.window_start or data.lift_end_date > self.window_end:
                error_details[channel_key] = (
                    f"Date range {data.lift_start_date} to {data.lift_end_date} "
                    f"is outside modeling window {self.window_start} to {self.window_end}"
                )
                error_messages.append(f"Date range for {'+'.join(channel_key)} is outside the modeling window.")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_spend_values(self) -> ValidationResult:
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, cal_data in self.calibration_input.channel_data.items():
            # First validate channel exists
            channel_validation = self._validate_channel_exists(channel_key)
            if not channel_validation.status:
                return channel_validation

            actual_spend = self._get_channel_spend(channel_key, cal_data.lift_start_date, cal_data.lift_end_date)

            if abs(actual_spend - cal_data.spend) > 0.1 * cal_data.spend:
                error_details[channel_key] = f"Spend mismatch: expected {cal_data.spend}, got {actual_spend}"
                error_messages.append(
                    f"Spend value for {'+'.join(channel_key)} does not match the input data (±10% tolerance)."
                )

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _get_channel_spend(
        self, channel_key: Tuple[str, ...], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> float:
        """Calculate total spend for channels in the given date range."""
        date_var = self.mmmdata.mmmdata_spec.date_var
        data = self.mmmdata.data
        date_mask = (data[date_var] >= start_date) & (data[date_var] <= end_date)
        return sum(data.loc[date_mask, channel].sum() for channel in channel_key)

    def check_calibration(self) -> ValidationResult:
        """Check all calibration inputs for consistency and correctness."""
        if self.calibration_input is None:
            return ValidationResult(status=True, error_details={}, error_message="")

        error_details = {}
        error_messages = []

        checks = [
            self._check_date_range(),
            self._check_spend_values(),
            self._check_metric_values(),
            self._check_confidence_values(),
            self._check_lift_values(),
        ]

        for result in checks:
            if not result.status:
                error_details.update(result.error_details)
                error_messages.append(result.error_message)

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def validate(self) -> List[ValidationResult]:
        """Run all validations."""
        return [self.check_calibration()]
