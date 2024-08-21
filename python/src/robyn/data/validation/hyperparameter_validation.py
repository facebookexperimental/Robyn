# Following methods from checks module should go here.
# def check_hyperparameters(hyperparameters=None, adstock=None, paid_media_spends=None, organic_vars=None, exposure_vars=None):
# def check_train_size(hyps):
# def check_hyper_limits(hyperparameters, hyper):

from typing import List
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.validation.validation import Validation, ValidationResult


class HyperparametersValidation(Validation):

    def __init__(self, hyperparameters: Hyperparameters) -> None:
        self.hyperparameters: Hyperparameters = hyperparameters

    def check_hyperparameters(self) -> ValidationResult:
        """
        Check if the hyperparameters are valid.
        
        :return: A dictionary with keys 'invalid' and 'missing', each containing a list of problematic hyperparameters.
        """
        raise NotImplementedError("Not yet implemented")

    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the result.

        :return: The result of the validation operation.
        """
        #raise NotImplementedError("Not yet implemented")
        pass
