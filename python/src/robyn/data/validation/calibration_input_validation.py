# pyre-strict
from robyn.data.entities.calibration_input import CalibrationInput, ChannelCalibrationData
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

    def _check_metric_values(self) -> ValidationResult:
        """Check if metric values match the dependent variable."""
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        dep_var = self.mmmdata.mmmdata_spec.dep_var

        for channel_key, data in self.calibration_input.channel_data.items():
            if data.metric != DependentVarType(dep_var):
                error_details[channel_key] = f"Metric mismatch: {data.metric} vs. {dep_var}"
                error_messages.append(
                    f"Metric for {'+'.join(channel_key)} does not match the dependent variable ({dep_var})."
                )

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_confidence_values(self) -> ValidationResult:
        """Check if confidence values are within acceptable range."""
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, data in self.calibration_input.channel_data.items():
            if data.confidence < 0.8:
                error_details[channel_key] = f"Low confidence: {data.confidence}"
                error_messages.append(
                    f"Confidence for {'+'.join(channel_key)} is lower than 80%, "
                    f"which is considered low confidence."
                )

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_lift_values(self) -> ValidationResult:
        """Check if lift values are valid numbers."""
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, data in self.calibration_input.channel_data.items():
            if not isinstance(data.lift_abs, (int, float)) or pd.isna(data.lift_abs):
                error_details[channel_key] = f"Invalid lift value: {data.lift_abs}"
                error_messages.append(f"Lift value for {'+'.join(channel_key)} must be a valid number.")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )


def create_modified_calibration_input(original_input, channel_name, **kwargs):
    """Create a modified version of a calibration input with updated values."""
    # Convert the channel_name to the format stored in channel_data
    if isinstance(channel_name, str):
        channel_tuple = (channel_name,)
    elif isinstance(channel_name, tuple):
        channel_tuple = channel_name
    else:
        raise ValueError(f"Invalid channel_name type: {type(channel_name)}")

    # Handle multi-channel cases
    if len(channel_tuple) > 1:
        # For multi-channel cases, check if it exists in original_input
        if channel_tuple not in original_input.channel_data:
            # Try the '+' joined version
            channel_key = "+".join(channel_tuple)
            if channel_key not in original_input.channel_data:
                raise KeyError(f"Channel combination not found: {channel_tuple}")
        else:
            channel_key = channel_tuple
    else:
        # For single channel cases
        channel_key = channel_tuple

    # Get original data
    original_channel_data = original_input.channel_data[channel_key]

    # Create new channel data with updates
    new_channel_data = ChannelCalibrationData(
        lift_start_date=kwargs.get("lift_start_date", original_channel_data.lift_start_date),
        lift_end_date=kwargs.get("lift_end_date", original_channel_data.lift_end_date),
        lift_abs=kwargs.get("lift_abs", original_channel_data.lift_abs),
        spend=kwargs.get("spend", original_channel_data.spend),
        confidence=kwargs.get("confidence", original_channel_data.confidence),
        metric=kwargs.get("metric", original_channel_data.metric),
        calibration_scope=kwargs.get("calibration_scope", original_channel_data.calibration_scope),
    )

    # Create new dictionary with updated data
    new_channel_data_dict = original_input.channel_data.copy()
    new_channel_data_dict[channel_key] = new_channel_data

    return CalibrationInput(channel_data=new_channel_data_dict)

    def validate(self) -> List[ValidationResult]:
        """Run all validations."""
        return [self.check_calibration()]
