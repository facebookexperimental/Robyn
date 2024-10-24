# pyre-strict
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
import pandas as pd
from typing import List, Tuple, Set
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
        # Initialize valid_channels from paid_media_spends
        self.valid_channels: Set[str] = set(self.mmmdata.mmmdata_spec.paid_media_spends)

    def check_calibration(self) -> ValidationResult:
        if self.calibration_input is None:
            return ValidationResult(status=True, error_details={}, error_message="")

        error_details = {}
        error_messages = []

        date_range_result = self._check_date_range()
        if not date_range_result.status:
            error_details.update(date_range_result.error_details)
            error_messages.append(date_range_result.error_message)

        spend_result = self._check_spend_values()
        if not spend_result.status:
            error_details.update(spend_result.error_details)
            error_messages.append(spend_result.error_message)

        lift_result = self._check_lift_values()
        if not lift_result.status:
            error_details.update(lift_result.error_details)
            error_messages.append(lift_result.error_message)

        confidence_result = self._check_confidence_values()
        if not confidence_result.status:
            error_details.update(confidence_result.error_details)
            error_messages.append(confidence_result.error_message)

        metric_result = self._check_metric_values()
        if not metric_result.status:
            error_details.update(metric_result.error_details)
            error_messages.append(metric_result.error_message)

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _validate_channel_exists(self, channel_key: Tuple[str, ...]) -> ValidationResult:
        """Validate that all channels in the key exist in the data."""
        missing_channels = [ch for ch in channel_key if ch not in self.valid_channels]

        if missing_channels:
            error_msg = f"Channels not found in data: {', '.join(missing_channels)}"
            return ValidationResult(
                status=False,
                error_details={channel_key: error_msg},
                error_message=f"The following channels are not in the input data: {', '.join(missing_channels)}",
            )
        return ValidationResult(status=True)

    def _get_channel_spend(
        self, channel_key: Tuple[str, ...], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> float:
        """Calculate total spend for single or combined channels in the given date range."""
        date_var = self.mmmdata.mmmdata_spec.date_var
        data = self.mmmdata.data

        date_mask = (data[date_var] >= start_date) & (data[date_var] <= end_date)
        return sum(data.loc[date_mask, ch].sum() for ch in channel_key)

    def _check_date_range(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        for channel_key, data in self.calibration_input.channel_data.items():
            # Validate channel exists before checking dates
            channel_validation = self._validate_channel_exists(channel_key)
            if not channel_validation.status:
                return channel_validation

            if data.lift_start_date < self.window_start or data.lift_end_date > self.window_end:
                error_details[channel_key] = (
                    f"Date range {data.lift_start_date} to {data.lift_end_date} "
                    f"is outside modeling window {self.window_start} to {self.window_end}"
                )
                error_messages.append(
                    f"Calibration date range for {'+'.join(channel_key)} is outside the modeling window."
                )

            if data.lift_start_date > data.lift_end_date:
                error_details[channel_key] = (
                    f"Start date {data.lift_start_date} is after end date {data.lift_end_date}"
                )
                error_messages.append(f"Invalid date range for {'+'.join(channel_key)}: start date is after end date.")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_spend_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        for channel_key, cal_data in self.calibration_input.channel_data.items():
            # Validate channel exists before checking spend
            channel_validation = self._validate_channel_exists(channel_key)
            if not channel_validation.status:
                return channel_validation

            actual_spend = self._get_channel_spend(channel_key, cal_data.lift_start_date, cal_data.lift_end_date)

            if abs(actual_spend - cal_data.spend) > 0.1 * cal_data.spend:
                error_details[channel_key] = (
                    f"Spend mismatch: calibration input ({cal_data.spend}) " f"vs. data ({actual_spend})"
                )
                error_messages.append(
                    f"Spend value for {'+'.join(channel_key)} does not match " f"the input data (±10% tolerance)."
                )

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_lift_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        for channel_key, data in self.calibration_input.channel_data.items():
            # Validate channel exists before checking lift
            channel_validation = self._validate_channel_exists(channel_key)
            if not channel_validation.status:
                return channel_validation

            if not isinstance(data.lift_abs, (int, float)) or pd.isna(data.lift_abs):
                error_details[channel_key] = f"Invalid lift value: {data.lift_abs}"
                error_messages.append(f"Lift value for {'+'.join(channel_key)} must be a valid number.")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_confidence_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        for channel_key, data in self.calibration_input.channel_data.items():
            # Validate channel exists before checking confidence
            channel_validation = self._validate_channel_exists(channel_key)
            if not channel_validation.status:
                return channel_validation

            if data.confidence < 0.8:
                error_details[channel_key] = f"Low confidence: {data.confidence}"
                error_messages.append(
                    f"Confidence for {'+'.join(channel_key)} is lower than 80%, "
                    f"which is considered low confidence."
                )

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_metric_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        dep_var = self.mmmdata.mmmdata_spec.dep_var

        for channel_key, data in self.calibration_input.channel_data.items():
            # Validate channel exists before checking metric
            channel_validation = self._validate_channel_exists(channel_key)
            if not channel_validation.status:
                return channel_validation

            if data.metric != DependentVarType(dep_var):
                error_details[channel_key] = f"Metric mismatch: {data.metric} vs. {dep_var}"
                error_messages.append(
                    f"Metric for {'+'.join(channel_key)} does not match the dependent variable ({dep_var})."
                )

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def validate(self) -> List[ValidationResult]:
        """Perform all validations and return the results."""
        return [self.check_calibration()]
