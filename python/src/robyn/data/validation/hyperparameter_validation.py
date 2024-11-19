import logging
from typing import List
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.validation.validation import Validation, ValidationResult
from robyn.data.entities.enums import AdstockType

logger = logging.getLogger(__name__)


class HyperparametersValidation(Validation):

    def __init__(self, hyperparameters: Hyperparameters) -> None:
        self.hyperparameters: Hyperparameters = hyperparameters
        logger.debug(
            "Initialized HyperparametersValidation with hyperparameters: %s",
            getattr(hyperparameters, "__dict__", str(hyperparameters)),
        )

    def check_hyperparameters(self) -> ValidationResult:
        """
        Check if the hyperparameters are valid.
        Returns:
            ValidationResult:
                An object containing the validation status, error details, and error message.
                The status is True if no errors were found, False otherwise.
                Error details is a dictionary containing specific error information for each invalid parameter.
                Error message is a string summarizing all errors found during validation.
        """
        logger.info("Starting hyperparameters validation")
        error_details = {}
        error_message = ""

        if self.hyperparameters.train_size is None:
            logger.warning("train_size is not set, using default values [0.5, 0.8]")
            self.hyperparameters.train_size = [0.5, 0.8]

        logger.debug("Validating train_size: %s", self.hyperparameters.train_size)
        try:
            self.check_train_size()
        except ValueError as e:
            logger.error("Train size validation failed: %s", str(e))
            error_details["train_size"] = str(e)
            error_message += f"Error in train_size: {str(e)}. "

        try:
            all_media = self.hyperparameters.hyperparameters.keys()
            logger.debug("Processing media channels: %s", all_media)
            hyper_names = self.hyper_names(all_media=all_media)
            logger.debug("Generated hyperparameter names: %s", hyper_names)

            for hyper in ["thetas", "alphas", "gammas", "shapes", "scales"]:
                logger.debug("Checking limits for hyperparameter: %s", hyper)
                self.check_hyper_limits(hyper)
        except Exception as e:
            logger.error("Hyperparameter validation failed: %s", str(e))
            error_details["hyperparameters"] = str(e)
            error_message += f"Error in hyperparameters: {str(e)}. "

        validation_result = ValidationResult(
            status=not error_details,
            error_details=error_details,
            error_message=error_message,
        )
        logger.info(
            "Hyperparameter validation completed. Status: %s", validation_result.status
        )
        if error_message:
            logger.error("Validation errors: %s", error_message)

        return validation_result

    def check_train_size(self):
        logger.debug("Checking train_size validation")
        train_size = self.hyperparameters.train_size

        if not isinstance(train_size, List) or len(train_size) != 2:
            logger.error("Invalid train_size format: %s", train_size)
            raise ValueError("train_size must be a list with exactly 2 elements")

        lower, upper = train_size
        if not all(isinstance(val, float) for val in train_size):
            logger.error("Invalid train_size value types: %s", train_size)
            raise TypeError("train_size values must be floats")

        if lower < 0.1 or lower >= upper or upper > 1:
            logger.error(
                "Train_size values out of bounds: lower=%s, upper=%s", lower, upper
            )
            raise ValueError(
                "train_size must be [lower, upper] where 0.1 <= lower < upper <= 1"
            )

        logger.debug("Train_size validation passed: %s", train_size)

    def hyper_names(self, all_media: List[str]) -> List[str]:
        """
        Get the names of all hyperparameters based on the adstock type.

        :return: A list of hyperparameter names.
        """
        logger.debug(
            "Generating hyperparameter names for adstock type: %s",
            self.hyperparameters.adstock,
        )
        adstock = self.hyperparameters.adstock

        if adstock == AdstockType.GEOMETRIC:
            names = [f"{media}_thetas" for media in all_media]
        elif adstock in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            names = [
                f"{media}_{param}"
                for media in all_media
                for param in ["shapes", "scales"]
            ]
        else:
            logger.error("Invalid adstock type encountered: %s", adstock)
            raise ValueError(f"Invalid adstock type: {adstock}")

        names.extend([f"{media}_alphas" for media in all_media])
        names.extend([f"{media}_gammas" for media in all_media])
        names.extend(["lambda", "train_size"])

        logger.debug("Generated hyperparameter names: %s", names)
        return names

    def check_hyper_limits(self, hyper: str) -> None:
        """
        Check if the hyperparameter values are within the allowed limits.

        :param hyper: The name of the hyperparameter to check.
        """
        logger.debug("Checking limits for hyperparameter: %s", hyper)
        limits = Hyperparameters.get_hyperparameter_limits()[hyper]
        logger.debug("Obtained limits for %s: %s", hyper, limits)

        def parse_limit(limit_str):
            logger.debug("Parsing limit string: %s", limit_str)
            if limit_str.startswith(">="):
                return float(limit_str[2:]), True  # inclusive
            elif limit_str.startswith(">"):
                return float(limit_str[1:]), False  # exclusive
            elif limit_str.startswith("<="):
                return float(limit_str[2:]), True  # inclusive
            elif limit_str.startswith("<"):
                return float(limit_str[1:]), False  # exclusive
            else:
                return float(limit_str), True  # assume inclusive if no symbol

        lower_limit, lower_inclusive = parse_limit(limits[0])
        upper_limit, upper_inclusive = parse_limit(limits[1])
        logger.debug(
            "Parsed limits - lower: %s (%s), upper: %s (%s)",
            lower_limit,
            lower_inclusive,
            upper_limit,
            upper_inclusive,
        )

        for (
            channel,
            channel_hyperparams,
        ) in self.hyperparameters.hyperparameters.items():
            logger.debug("Checking channel %s hyperparameters", channel)
            values = getattr(channel_hyperparams, hyper)

            if values is None:
                logger.debug("No values defined for %s in channel %s", hyper, channel)
                return

            if len(values) not in [1, 2]:
                logger.error(
                    "Invalid number of values for %s in channel %s: %s",
                    hyper,
                    channel,
                    len(values),
                )
                raise ValueError(f"Hyperparameter '{hyper}' must have 1 or 2 values")

            if (lower_inclusive and float(values[0]) < lower_limit) or (
                not lower_inclusive and float(values[0]) <= lower_limit
            ):
                logger.error(
                    "Lower bound violation for %s in channel %s: %s",
                    hyper,
                    channel,
                    values[0],
                )
                raise ValueError(
                    f"{hyper}'s hyperparameter must be {'≥' if lower_inclusive else '>'} {lower_limit}"
                )

            if len(values) == 2:
                if (upper_inclusive and float(values[1]) > upper_limit) or (
                    not upper_inclusive and float(values[1]) >= upper_limit
                ):
                    logger.error(
                        "Upper bound violation for %s in channel %s: %s",
                        hyper,
                        channel,
                        values[1],
                    )
                    raise ValueError(
                        f"{hyper}'s hyperparameter must be {'≤' if upper_inclusive else '<'} {upper_limit}"
                    )
                if float(values[0]) > float(values[1]):
                    logger.error(
                        "Invalid bounds order for %s in channel %s: %s",
                        hyper,
                        channel,
                        values,
                    )
                    raise ValueError(
                        f"{hyper}'s hyperparameter must have lower bound first and upper bound second"
                    )

            logger.debug(
                "Validation passed for %s in channel %s: %s", hyper, channel, values
            )

    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the result.

        :return: The result of the validation operation.
        """
        logger.info("Starting validation process")
        results = [self.check_hyperparameters()]
        logger.info(
            "Validation completed with status: %s",
            all(result.status for result in results),
        )
        return results
