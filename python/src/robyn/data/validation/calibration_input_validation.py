# pyre-strict
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
import pandas as pd
from typing import List, Tuple
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

    def check_calibration(self) -> ValidationResult:
        """
        This function checks the calibration input data for consistency and correctness.
        It verifies that the input data contains the required columns, that the date range
        is within the modeling window, and that the spend values match the input data.
        """

        if self.calibration_input is None:
            return ValidationResult(status=True, error_details={}, error_message="")

        error_details = {}
        error_messages = []

        # check data range
        date_range_result = self._check_date_range()
        if not date_range_result.status:
            error_details.update(date_range_result.error_details)
            error_messages.append(date_range_result.error_message)

        # Check lift values
        lift_result = self._check_lift_values()
        if not lift_result.status:
            error_details.update(lift_result.error_details)
            error_messages.append(lift_result.error_message)

        # Check spend values
        spend_result = self._check_spend_values()
        if not spend_result.status:
            error_details.update(spend_result.error_details)
            error_messages.append(spend_result.error_message)

        # Check confidence values
        confidence_result = self._check_confidence_values()
        if not confidence_result.status:
            error_details.update(confidence_result.error_details)
            error_messages.append(confidence_result.error_message)

        # Check metric values
        metric_result = self._check_metric_values()
        if not metric_result.status:
            error_details.update(metric_result.error_details)
            error_messages.append(metric_result.error_message)

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def check_obj_weights(self, objective_weights: List[float], refresh: bool) -> ValidationResult:
        """
        Check the objective weights for validity.

        :param objective_weights: List of objective weights
        :param refresh: Boolean indicating if this is a refresh run
        :return: ValidationResult with the status and any error messages
        """
        error_details = {}
        error_messages = []

        if objective_weights is None:
            if refresh:
                obj_len = 3  # Assuming 3 objectives for refresh runs
                objective_weights = [0, 1, 1]
            else:
                return ValidationResult(status=True, error_details={}, error_message="")

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
        """
        Validate that all channels in the key exist in the data.

        Args:
            channel_key: Tuple of channel names

        Returns:
            ValidationResult indicating if all channels exist
        """
        missing_channels = [ch for ch in channel_key if ch not in self.valid_channels]

        if missing_channels:
            return ValidationResult(
                status=False,
                error_details={channel_key: f"Channels not found in data: {', '.join(missing_channels)}"},
                error_message=f"The following channels for {channel_key} are not in the input data: {', '.join(missing_channels)}",
            )
        return ValidationResult(status=True)

    def _get_channel_spend(
        self, channel_key: Tuple[str, ...], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> float:
        """
        Calculate total spend for single or combined channels in the given date range.

        Args:
            channel_key: Tuple of channel names
            start_date: Start date for spend calculation
            end_date: End date for spend calculation

        Returns:
            Total spend for the channel(s)

        Raises:
            KeyError: If any channel is not found in the data
        """
        date_var = self.mmmdata.mmmdata_spec.date_var
        data = self.mmmdata.data

        # Get data for the date range
        date_mask = data[date_var].between(start_date, end_date)

        # For single channel
        if len(channel_key) == 1:
            return data.loc[date_mask, channel_key[0]].sum()

        # For multiple channels, sum their individual spends
        return sum(data.loc[date_mask, channel].sum() for channel in channel_key)

    def _check_date_range(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        for channel, data in self.calibration_input.channel_data.items():
            if data.lift_start_date < self.window_start or data.lift_end_date > self.window_end:
                error_details[channel] = (
                    f"Date range {data.lift_start_date} to {data.lift_end_date} is outside modeling window {self.window_start} to {self.window_end}"
                )
                error_messages.append(f"Calibration date range for {channel} is outside the modeling window.")

            if data.lift_start_date > data.lift_end_date:
                error_details[channel] = f"Start date {data.lift_start_date} is after end date {data.lift_end_date}"
                error_messages.append(f"Invalid date range for {channel}: start date is after end date.")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_lift_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        for channel, data in self.calibration_input.channel_data.items():
            if not isinstance(data.lift_abs, (int, float)) or pd.isna(data.lift_abs):
                error_details[channel] = f"Invalid lift value: {data.lift_abs}"
                error_messages.append(f"Lift value for {channel} must be a valid number.")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_spend_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        date_var = self.mmmdata.mmmdata_spec.date_var
        data = self.mmmdata.data

        for channel, cal_data in self.calibration_input.channel_data.items():
            channel_spend = data[data[date_var].between(cal_data.lift_start_date, cal_data.lift_end_date)][
                channel
            ].sum()

            if abs(channel_spend - cal_data.spend) > 0.1 * cal_data.spend:
                error_details[channel] = (
                    f"Spend mismatch: calibration input ({cal_data.spend}) vs. data ({channel_spend})"
                )
                error_messages.append(f"Spend value for {channel} does not match the input data (±10% tolerance).")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_confidence_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        for channel, data in self.calibration_input.channel_data.items():
            if data.confidence < 0.8:
                error_details[channel] = f"Low confidence: {data.confidence}"
                error_messages.append(
                    f"Confidence for {channel} is lower than 80%, which is considered low confidence."
                )

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def _check_metric_values(self) -> ValidationResult:
        error_details = {}
        error_messages = []

        dep_var = self.mmmdata.mmmdata_spec.dep_var

        for channel, data in self.calibration_input.channel_data.items():
            if data.metric != DependentVarType(dep_var):
                error_details[channel] = f"Metric mismatch: {data.metric} vs. {dep_var}"
                error_messages.append(f"Metric for {channel} does not match the dependent variable ({dep_var}).")

        return ValidationResult(
            status=len(error_details) == 0, error_details=error_details, error_message="\n".join(error_messages)
        )

    def validate(self) -> ValidationResult:
        """
        Perform all validations and return the results.

        :return: A dictionary containing the results of all validations.
        """
        return [self.check_calibration()]
