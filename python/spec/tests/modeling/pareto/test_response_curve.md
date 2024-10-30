# CLASS
## ResponseCurveCalculator
- This class is responsible for calculating response curves based on media mix modeling (MMM) data, model outputs, and hyperparameters.
- Part of a larger module that handles data transformations and response calculations.

# CONSTRUCTORS
## ResponseCurveCalculator `(mmm_data: MMMData, model_outputs: ModelOutputs, hyperparameter: Hyperparameters)`
### USAGE
- Use this constructor to create an instance of `ResponseCurveCalculator` with the necessary data and hyperparameters to compute response curves.

### IMPL
- The constructor should be tested by initializing the `ResponseCurveCalculator` class with mock instances of `MMMData`, `ModelOutputs`, and `Hyperparameters`.
- Verify that the instance variables `mmm_data`, `model_outputs`, and `hyperparameter` are correctly assigned.
- Ensure that `self.transformation` is initialized as an instance of the `Transformation` class with the `mmm_data` passed correctly.
- Test the constructor with edge cases like passing `None` or incorrect types to ensure proper error handling.