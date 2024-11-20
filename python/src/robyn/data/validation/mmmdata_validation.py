# pyre-strict
from typing import List
import logging
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MMMDataValidation(Validation):
    def __init__(self, mmm_data: MMMData) -> None:
        self.mmm_data: MMMData = mmm_data
        logger.debug(
            "Initializing MMMDataValidation with data shape: %s",
            self.mmm_data.data.shape,
        )

    def check_missing_and_infinite(self) -> ValidationResult:
        """
        Check for missing (NA) values and infinite (Inf) values in the DataFrame.

        :return: validation result with error_details containing a dictionary with keys 'missing' and 'infinite', each containing a list of column names with missing or infinite values.
        """
        logger.debug("Starting missing and infinite value check")
        data = self.mmm_data.data

        missing_cols = data.columns[data.isnull().any()].tolist()
        infinite_cols = [
            col
            for col in data.columns
            if data[col].dtype.kind in "bifc" and np.isinf(data[col]).any()
        ]

        error_details = {}
        error_message = ""

        if missing_cols:
            error_details["missing"] = missing_cols
            error_message += f"Dataset contains missing (NA) values in columns: {', '.join(missing_cols)}. "
            logger.warning("Found missing values in columns: %s", missing_cols)

        if infinite_cols:
            error_details["infinite"] = infinite_cols
            error_message += f"Dataset contains infinite (Inf) values in columns: {', '.join(infinite_cols)}. "
            logger.warning("Found infinite values in columns: %s", infinite_cols)

        if error_message:
            error_message += (
                "These values must be removed or fixed for Robyn to properly work."
            )
            logger.error("Validation failed: %s", error_message)
        else:
            logger.info("Missing and infinite value check passed successfully")

        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message,
        )

    def check_no_variance(self) -> ValidationResult:
        """
        Check for columns with no variance in the input dataframe.

        :return: A list of column names with no variance.
        """
        logger.debug("Starting no-variance check")
        data = self.mmm_data.data

        no_variance_cols = data.columns[data.nunique() == 1].tolist()

        error_details = {}
        error_message = ""

        if no_variance_cols:
            error_details["no_variance"] = no_variance_cols
            error_message = f"There are {len(no_variance_cols)} column(s) with no-variance: {', '.join(no_variance_cols)}. Please remove the variable(s) to proceed."
            logger.warning("Found columns with no variance: %s", no_variance_cols)
        else:
            logger.info("No-variance check passed successfully")

        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message,
        )

    def check_variable_names(self) -> ValidationResult:
        """
        Check variable names for duplicates and invalid characters.

        :return: A dictionary with keys 'duplicates' and 'invalid', each containing a list of problematic column names.
        """
        logger.debug("Starting variable names validation")
        mmmdata_spec = self.mmm_data.mmmdata_spec

        # Collect all variable names to check
        vars_to_check = [
            mmmdata_spec.dep_var,
            mmmdata_spec.date_var,
            mmmdata_spec.context_vars,
            mmmdata_spec.paid_media_spends,
            mmmdata_spec.organic_vars,
        ]
        vars_to_check = [var for var in vars_to_check if var is not None]
        vars_to_check = [
            item
            for sublist in vars_to_check
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        logger.debug("Checking variable names: %s", vars_to_check)

        # Check for duplicates
        duplicates = [var for var in set(vars_to_check) if vars_to_check.count(var) > 1]

        # Check for invalid characters (spaces)
        invalid = [var for var in vars_to_check if " " in var]

        error_details = {}
        error_message = ""

        if duplicates:
            error_details["duplicates"] = duplicates
            error_message += "Duplicate variable names present. "
            logger.warning("Found duplicate variable names: %s", duplicates)

        if invalid:
            error_details["invalid"] = invalid
            error_message += "Invalid variable names present. "
            logger.warning(
                "Found invalid variable names (containing spaces): %s", invalid
            )

        if not error_details:
            logger.info("Variable names validation passed successfully")

        return ValidationResult(
            status=not error_details,
            error_details=error_details,
            error_message=error_message,
        )

    def check_date_variable(self) -> ValidationResult:
        """
        Checks if the date variable is correct.

        :return: True if the date variable is valid, False otherwise.
        """
        logger.debug("Starting date variable validation")
        mmmdata_spec = self.mmm_data.mmmdata_spec
        data = self.mmm_data.data
        date_var = mmmdata_spec.date_var

        error_details = {}
        error_message = ""

        if date_var == "auto":
            error_message = (
                "Date variable is not set. Please set the date variable to proceed."
            )
            logger.error("Date variable validation failed: date variable not set")
            return ValidationResult(
                status=False,
                error_details={"date_variable": error_message},
                error_message=error_message,
            )

        logger.debug("Checking date variable: %s", date_var)

        if date_var not in data.columns:
            error_message = f"Date variable '{date_var}' not found in the input data."
            logger.error("Date variable validation failed: %s", error_message)
        else:
            try:
                data[date_var] = pd.to_datetime(data[date_var])
                if not data[date_var].is_monotonic_increasing:
                    error_message = (
                        f"Date variable '{date_var}' is not in ascending order."
                    )
                    logger.warning(
                        "Date variable validation failed: dates not in ascending order"
                    )
            except ValueError:
                error_message = (
                    f"Date variable '{date_var}' contains invalid date values."
                )
                logger.error("Date variable validation failed: invalid date values")

        if error_message:
            error_details["date_variable"] = error_message
        else:
            logger.info("Date variable validation passed successfully")

        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message,
        )

    def check_dependent_variables(self) -> ValidationResult:
        """
        Checks if dependent variables are valid.

        :return: True if the dependent variables are valid, False otherwise.
        """
        logger.debug("Starting dependent variables validation")
        mmmdata_spec = self.mmm_data.mmmdata_spec
        data = self.mmm_data.data
        dep_var = mmmdata_spec.dep_var

        error_details = {}
        error_message = ""

        logger.debug("Checking dependent variable: %s", dep_var)

        if dep_var not in data.columns:
            error_message = (
                f"Dependent variable '{dep_var}' not found in the input data."
            )
            logger.error("Dependent variable validation failed: variable not found")
        else:
            if not pd.api.types.is_numeric_dtype(data[dep_var]):
                error_message = f"Dependent variable '{dep_var}' must be numeric."
                logger.error("Dependent variable validation failed: non-numeric type")

        if error_message:
            error_details["dependent_variable"] = error_message
        else:
            logger.info("Dependent variable validation passed successfully")

        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message,
        )

    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the results.

        :return: A dictionary containing the results of all validations.
        """
        logger.info("Starting complete MMMData validation")
        results = [
            self.check_missing_and_infinite(),
            self.check_no_variance(),
            self.check_variable_names(),
            self.check_date_variable(),
            self.check_dependent_variables(),
        ]

        failed_validations = [r for r in results if not r.status]
        if failed_validations:
            logger.error(
                "Validation completed with %d failures", len(failed_validations)
            )
        else:
            logger.info("All validations passed successfully")

        return results
