# Following methods from checks module should go here.
# def check_hyperparameters(hyperparameters=None, adstock=None, paid_media_spends=None, organic_vars=None, exposure_vars=None):
# def check_train_size(hyps):
# def check_hyper_limits(hyperparameters, hyper):

import logging
from typing import List
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.validation.validation import Validation, ValidationResult
from robyn.data.entities.enums import AdstockType

class HyperparametersValidation(Validation):

    def __init__(self, hyperparameters: Hyperparameters) -> None:
        self.hyperparameters: Hyperparameters = hyperparameters

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

        error_details = {}
        error_message = ""

        if self.hyperparameters.train_size is None:
            logging.warning("train_size is not set. Using default values [0.5, 0.8]")
            self.hyperparameters.train_size = [0.5, 0.8]
        
        try:
            self.check_train_size()
        except ValueError as e:
            error_details["train_size"] = str(e)
            error_message += f"Error in train_size: {str(e)}. "

        try:
            all_media = self.hyperparameters.hyperparameters.keys()
            hyper_names = self.hyper_names(all_media=all_media)

            for hyper in ["thetas", "alphas", "gammas", "shapes", "scales"]:
                self.check_hyper_limits(hyper)
        except Exception as e:
            error_details["hyperparameters"] = str(e)
            error_message += f"Error in hyperparameters: {str(e)}. "

        return ValidationResult(
            status=not error_details,
            error_details=error_details,
            error_message=error_message
        )

    def check_train_size(self):
        train_size = self.hyperparameters.train_size
        if not isinstance(train_size, List) or len(train_size) != 2:
            raise ValueError("train_size must be a list with exactly 2 elements")
        
        lower, upper = train_size
        if not all(isinstance(val, float) for val in train_size):
            raise TypeError("train_size values must be floats")
        
        if lower < 0.1 or lower >= upper or upper > 1:
            raise ValueError("train_size must be [lower, upper] where 0.1 <= lower < upper <= 1")
    
    def hyper_names(self, all_media: List[str]) -> List[str]:
        """
        Get the names of all hyperparameters based on the adstock type.

        :return: A list of hyperparameter names.
        """
        adstock = self.hyperparameters.adstock
        
        if adstock == AdstockType.GEOMETRIC:
            names = [f"{media}_thetas" for media in all_media]
        elif adstock in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            names = [f"{media}_{param}" for media in all_media for param in ["shapes", "scales"]]
        else:
            raise ValueError(f"Invalid adstock type: {adstock}")
        
        names.extend([f"{media}_alphas" for media in all_media])
        names.extend([f"{media}_gammas" for media in all_media])
        names.extend(["lambda", "train_size"])
        
        return names
    
    def check_hyper_limits(self, hyper: str) -> None:
        """
        Check if the hyperparameter values are within the allowed limits.

        :param hyper: The name of the hyperparameter to check.
        """
        limits = Hyperparameters.get_hyperparameter_limits()[hyper]

        def parse_limit(limit_str):
            if limit_str.startswith('>='):
                return float(limit_str[2:]), True  # inclusive
            elif limit_str.startswith('>'):
                return float(limit_str[1:]), False  # exclusive
            elif limit_str.startswith('<='):
                return float(limit_str[2:]), True  # inclusive
            elif limit_str.startswith('<'):
                return float(limit_str[1:]), False  # exclusive
            else:
                return float(limit_str), True  # assume inclusive if no symbol
        
        lower_limit, lower_inclusive = parse_limit(limits[0])
        upper_limit, upper_inclusive = parse_limit(limits[1])

        for channel, channel_hyperparams in self.hyperparameters.hyperparameters.items():
            values = getattr(channel_hyperparams, hyper)
        
            if values is None:
                return
            
            if len(values) not in [1, 2]:
                raise ValueError(f"Hyperparameter '{hyper}' must have 1 or 2 values")
        
            if (lower_inclusive and float(values[0]) < lower_limit) or \
            (not lower_inclusive and float(values[0]) <= lower_limit):
                raise ValueError(f"{hyper}'s hyperparameter must be {'≥' if lower_inclusive else '>'} {lower_limit}")
        
            if len(values) == 2:
                if (upper_inclusive and float(values[1]) > upper_limit) or \
                (not upper_inclusive and float(values[1]) >= upper_limit):
                    raise ValueError(f"{hyper}'s hyperparameter must be {'≤' if upper_inclusive else '<'} {upper_limit}")
                if float(values[0]) > float(values[1]):
                    raise ValueError(f"{hyper}'s hyperparameter must have lower bound first and upper bound second")


    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the result.

        :return: The result of the validation operation.
        """
        return [self.check_hyperparameters()]
