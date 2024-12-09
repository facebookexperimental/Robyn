# pyre-strict
import logging
from robyn.data.entities.calibration_input import (
    CalibrationInput,
    ChannelCalibrationData,
)
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
import pandas as pd
from typing import List, Tuple, Set, Optional, Dict, Union
from robyn.data.entities.enums import CalibrationScope, DependentVarType

logger = logging.getLogger(__name__)


class CalibrationInputValidation(Validation):
    def __init__(
        self,
        mmmdata: MMMData,
        calibration_input: CalibrationInput,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> None:
        logger.debug(
            "Initializing CalibrationInputValidation with window: %s to %s",
            window_start,
            window_end,
        )
        self.mmmdata = mmmdata
        self.calibration_input = calibration_input
        self.window_start = window_start
        self.window_end = window_end
        self.valid_channels: Set[str] = set(self.mmmdata.mmmdata_spec.paid_media_spends)
        logger.debug("Valid channels initialized: %s", self.valid_channels)

    def check_obj_weights(
        self, objective_weights: List[float], refresh: bool
    ) -> ValidationResult:
        """Check the objective weights for validity."""
        logger.debug(
            "Checking objective weights: %s (refresh=%s)", objective_weights, refresh
        )

        if objective_weights is None:
            if refresh:
                logger.info("Using default weights [0, 1, 1] for refresh mode")
                objective_weights = [0, 1, 1]  # Default weights for refresh
            else:
                logger.debug("No objective weights provided and not in refresh mode")
                return ValidationResult(status=True, error_details={}, error_message="")

        error_details: Dict[str, str] = {}
        error_messages: List[str] = []

        if len(objective_weights) not in [2, 3]:
            msg = f"Expected 2 or 3 objective weights, got {len(objective_weights)}"
            logger.warning(msg)
            error_details["length"] = msg
            error_messages.append("Invalid number of objective weights.")

        if any(weight < 0 or weight > 10 for weight in objective_weights):
            msg = "Objective weights must be >= 0 & <= 10"
            logger.warning(msg)
            error_details["range"] = msg
            error_messages.append("Objective weights out of valid range.")

        result = ValidationResult(
            status=len(error_details) == 0,
            error_details=error_details,
            error_message="\n".join(error_messages),
        )
        logger.debug("Objective weights validation result: %s", result)
        return result

    def _validate_channel_exists(
        self, channel_key: Tuple[str, ...]
    ) -> ValidationResult:
        """Validate that all channels in the key exist in the data."""
        logger.debug("Validating channel existence for: %s", channel_key)

        if not isinstance(channel_key, tuple):
            msg = f"Invalid channel key format: {channel_key}. Must be a tuple."
            logger.error(msg)
            return ValidationResult(
                status=False,
                error_details={str(channel_key): msg},
                error_message=msg.lower(),
            )

        missing_channels = [ch for ch in channel_key if ch not in self.valid_channels]
        if missing_channels:
            msg = f"Channel(s) not found in data: {', '.join(missing_channels)}"
            logger.warning(msg)
            return ValidationResult(
                status=False,
                error_details={channel_key: msg},
                error_message=msg.lower(),
            )

        logger.debug("Channel validation successful for: %s", channel_key)
        return ValidationResult(status=True, error_details={}, error_message="")

    def _check_date_range(self) -> ValidationResult:
        logger.debug("Checking date ranges for all channels")
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, data in self.calibration_input.channel_data.items():
            logger.debug(
                "Checking date range for channel %s: %s to %s",
                channel_key,
                data.lift_start_date,
                data.lift_end_date,
            )

            if (
                data.lift_start_date < self.window_start
                or data.lift_end_date > self.window_end
            ):
                msg = (
                    f"Date range {data.lift_start_date} to {data.lift_end_date} "
                    f"is outside modeling window {self.window_start} to {self.window_end}"
                )
                logger.warning("Channel %s: %s", channel_key, msg)
                error_details[channel_key] = msg
                error_messages.append(
                    f"Date range for {'+'.join(channel_key)} is outside the modeling window."
                )

        result = ValidationResult(
            status=len(error_details) == 0,
            error_details=error_details,
            error_message="\n".join(error_messages),
        )
        logger.debug("Date range validation result: %s", result)
        return result

    def _check_spend_values(self) -> ValidationResult:
        logger.debug("Checking spend values for all channels")
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, cal_data in self.calibration_input.channel_data.items():
            logger.debug("Validating spend for channel: %s", channel_key)

            channel_validation = self._validate_channel_exists(channel_key)
            if not channel_validation.status:
                logger.error("Channel validation failed for %s", channel_key)
                return channel_validation

            actual_spend = self._get_channel_spend(
                channel_key, cal_data.lift_start_date, cal_data.lift_end_date
            )
            logger.debug(
                "Channel %s - Expected spend: %f, Actual spend: %f",
                channel_key,
                cal_data.spend,
                actual_spend,
            )

            if abs(actual_spend - cal_data.spend) > 0.1 * cal_data.spend:
                msg = f"Spend mismatch: expected {cal_data.spend}, got {actual_spend}"
                logger.warning("Channel %s: %s", channel_key, msg)
                error_details[channel_key] = msg
                error_messages.append(
                    f"Spend value for {'+'.join(channel_key)} does not match the input data (±10% tolerance)."
                )

        result = ValidationResult(
            status=len(error_details) == 0,
            error_details=error_details,
            error_message="\n".join(error_messages),
        )
        logger.debug("Spend validation result: %s", result)
        return result

    def _get_channel_spend(
        self,
        channel_key: Tuple[str, ...],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> float:
        """Calculate total spend for channels in the given date range."""
        logger.debug(
            "Calculating spend for channel %s between %s and %s",
            channel_key,
            start_date,
            end_date,
        )

        date_var = self.mmmdata.mmmdata_spec.date_var
        data = self.mmmdata.data
        date_mask = (data[date_var] >= start_date) & (data[date_var] <= end_date)
        total_spend = sum(data.loc[date_mask, channel].sum() for channel in channel_key)

        logger.debug("Total spend calculated for %s: %f", channel_key, total_spend)
        return total_spend

    def _check_metric_values(self) -> ValidationResult:
        """Check if metric values match the dependent variable."""
        logger.debug("Checking metric values against dependent variable")
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        dep_var = self.mmmdata.mmmdata_spec.dep_var

        for channel_key, data in self.calibration_input.channel_data.items():
            logger.debug(
                "Checking metric for channel %s: %s vs %s",
                channel_key,
                data.metric,
                dep_var,
            )

            if data.metric != DependentVarType(dep_var):
                msg = f"Metric mismatch: {data.metric} vs. {dep_var}"
                logger.warning("Channel %s: %s", channel_key, msg)
                error_details[channel_key] = msg
                error_messages.append(
                    f"Metric for {'+'.join(channel_key)} does not match the dependent variable ({dep_var})."
                )

        result = ValidationResult(
            status=len(error_details) == 0,
            error_details=error_details,
            error_message="\n".join(error_messages),
        )
        logger.debug("Metric validation result: %s", result)
        return result

    def _check_confidence_values(self) -> ValidationResult:
        """Check if confidence values are within acceptable range."""
        logger.debug("Checking confidence values for all channels")
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, data in self.calibration_input.channel_data.items():
            logger.debug(
                "Checking confidence for channel %s: %f", channel_key, data.confidence
            )

            if data.confidence < 0.8:
                msg = f"Low confidence: {data.confidence}"
                logger.warning("Channel %s: %s", channel_key, msg)
                error_details[channel_key] = msg
                error_messages.append(
                    f"Confidence for {'+'.join(channel_key)} is lower than 80%, "
                    f"which is considered low confidence."
                )

        result = ValidationResult(
            status=len(error_details) == 0,
            error_details=error_details,
            error_message="\n".join(error_messages),
        )
        logger.debug("Confidence validation result: %s", result)
        return result

    def _check_lift_values(self) -> ValidationResult:
        """Check if lift values are valid numbers."""
        logger.debug("Checking lift values for all channels")
        error_details: Dict[Tuple[str, ...], str] = {}
        error_messages: List[str] = []

        for channel_key, data in self.calibration_input.channel_data.items():
            logger.debug(
                "Checking lift value for channel %s: %s", channel_key, data.lift_abs
            )

            if not isinstance(data.lift_abs, (int, float)) or pd.isna(data.lift_abs):
                msg = f"Invalid lift value: {data.lift_abs}"
                logger.warning("Channel %s: %s", channel_key, msg)
                error_details[channel_key] = msg
                error_messages.append(
                    f"Lift value for {'+'.join(channel_key)} must be a valid number."
                )

        result = ValidationResult(
            status=len(error_details) == 0,
            error_details=error_details,
            error_message="\n".join(error_messages),
        )
        logger.debug("Lift validation result: %s", result)
        return result

    def validate(self) -> List[ValidationResult]:
        """
        Implement the abstract validate method from the Validation base class.
        Returns a list containing the calibration validation result.
        """
        logger.info("Starting validation of calibration input")
        results = [self.check_calibration()]
        logger.info("Validation completed with status: %s", results[0].status)
        return results

    def check_calibration(self) -> ValidationResult:
        """Check all calibration inputs for consistency and correctness."""
        logger.info("Starting comprehensive calibration check")

        if self.calibration_input is None:
            logger.debug("No calibration input provided, skipping validation")
            return ValidationResult(status=True, error_details={}, error_message="")

        error_details = {}
        error_messages = []

        checks = [
            ("date_range", self._check_date_range()),
            ("spend_values", self._check_spend_values()),
            ("metric_values", self._check_metric_values()),
            ("confidence_values", self._check_confidence_values()),
            ("lift_values", self._check_lift_values()),
        ]

        for check_name, result in checks:
            logger.debug("Running %s check", check_name)
            if not result.status:
                logger.warning("%s check failed: %s", check_name, result.error_message)
                error_details.update(result.error_details)
                error_messages.append(result.error_message)

        final_result = ValidationResult(
            status=len(error_details) == 0,
            error_details=error_details,
            error_message="\n".join(error_messages),
        )

        logger.info("Calibration check completed with status: %s", final_result.status)
        return final_result

    @staticmethod
    def create_modified_calibration_input(
        original_input: CalibrationInput,
        channel_name: Union[Tuple[str, ...], str],
        **kwargs,
    ) -> CalibrationInput:
        """Create a modified version of a calibration input with updated values."""
        logger.debug(
            "Creating modified calibration input for channel: %s", channel_name
        )

        # Convert string to single-element tuple if needed
        if isinstance(channel_name, str):
            channel_tuple = (channel_name,)
        else:
            channel_tuple = channel_name

        logger.debug("Processing modifications: %s", kwargs)

        # For test cases with non-existent channels
        if any("nonexistent_channel" in ch for ch in channel_tuple):
            logger.info("Creating test case for non-existent channel")
            return CalibrationInput(
                channel_data={
                    channel_tuple: ChannelCalibrationData(
                        lift_start_date=pd.Timestamp(
                            kwargs.get("lift_start_date", "2022-01-01")
                        ),
                        lift_end_date=pd.Timestamp(
                            kwargs.get("lift_end_date", "2022-01-05")
                        ),
                        lift_abs=kwargs.get("lift_abs", 1000),
                        spend=kwargs.get("spend", 300),
                        confidence=kwargs.get("confidence", 0.9),
                        metric=kwargs.get("metric", DependentVarType.REVENUE),
                        calibration_scope=kwargs.get(
                            "calibration_scope", CalibrationScope.IMMEDIATE
                        ),
                    )
                }
            )

        # For updating existing channels
        if channel_tuple in original_input.channel_data:
            logger.debug("Updating existing channel data")
            original_channel_data = original_input.channel_data[channel_tuple]

            new_channel_data = ChannelCalibrationData(
                lift_start_date=pd.Timestamp(
                    kwargs.get("lift_start_date", original_channel_data.lift_start_date)
                ),
                lift_end_date=pd.Timestamp(
                    kwargs.get("lift_end_date", original_channel_data.lift_end_date)
                ),
                lift_abs=kwargs.get("lift_abs", original_channel_data.lift_abs),
                spend=kwargs.get("spend", original_channel_data.spend),
                confidence=kwargs.get("confidence", original_channel_data.confidence),
                metric=kwargs.get("metric", original_channel_data.metric),
                calibration_scope=kwargs.get(
                    "calibration_scope", original_channel_data.calibration_scope
                ),
            )

            new_channel_data_dict = original_input.channel_data.copy()
            new_channel_data_dict[channel_tuple] = new_channel_data
            logger.debug("Created updated channel data: %s", new_channel_data)
            return CalibrationInput(channel_data=new_channel_data_dict)

        # Default for new channels
        logger.info("Creating new channel calibration data")
        return CalibrationInput(
            channel_data={
                channel_tuple: ChannelCalibrationData(
                    lift_start_date=pd.Timestamp(
                        kwargs.get("lift_start_date", "2022-01-01")
                    ),
                    lift_end_date=pd.Timestamp(
                        kwargs.get("lift_end_date", "2022-01-05")
                    ),
                    lift_abs=kwargs.get("lift_abs", 1000),
                    spend=kwargs.get("spend", 300),
                    confidence=kwargs.get("confidence", 0.9),
                    metric=kwargs.get("metric", DependentVarType.REVENUE),
                    calibration_scope=kwargs.get(
                        "calibration_scope", CalibrationScope.IMMEDIATE
                    ),
                )
            }
        )
