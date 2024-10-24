# CLASS
## RidgeModelBuilder
* This class is responsible for building and training Ridge regression models.
* It is designed to handle various configurations for model training, including the use of different algorithms from the Nevergrad library for optimization.
* It integrates with various data entities like `MMMData`, `HolidaysData`, `CalibrationInput`, and uses `Hyperparameters` and `FeaturizedMMMData` for feature engineering and hyperparameter management.

# CONSTRUCTORS
## `__init__(self, mmm_data: MMMData, holiday_data: HolidaysData, calibration_input: CalibrationInput, hyperparameters: Hyperparameters, featurized_mmm_data: FeaturizedMMMData)`
* Initializes the `RidgeModelBuilder` with essential data components and configurations required for building models.

### USAGE
* Use this constructor to create an instance of `RidgeModelBuilder` when you have all the necessary data entities ready.
* This sets up the internal state of the class to use these data entities for model building operations.

### IMPL
* Ensure that all the provided parameters (`mmm_data`, `holiday_data`, `calibration_input`, `hyperparameters`, `featurized_mmm_data`) are correctly assigned to instance variables.
* Verify that a logger is set up for the class to facilitate logging of various operations and processes.
* Test the constructor by checking if all instance variables are initialized properly, and the logger is configured without errors.

# METHODS


## `test_model_outputs_instance() -> None`
### USAGE
* This test function verifies that the `build_models` method of the `RidgeModelBuilder` class returns an instance of `ModelOutputs`.
* It ensures that the output conforms to the expected data type, which is essential for subsequent processing and validations.

### IMPL
* Start by setting up the necessary environment and dependencies for the test, including instantiating the `RidgeModelBuilder` class with mock data entities.
* Define the input parameters for the `build_models` method according to the test case specifications.
* Mock the `Convergence` class's `calculate_convergence` method to return predefined convergence messages.
* Mock the `RidgeModelBuilder` class's `_select_best_model` method to return a predefined model ID.
* Invoke the `build_models` method with the specified test input parameters.
* Retrieve the output from the method call.
* Assert that the type of the output is `ModelOutputs` to ensure the method's return type is as expected.

## `test_model_outputs_trials() -> None`
### USAGE
* This test function checks that the `trials` attribute of the `ModelOutputs` object returned by the `build_models` method contains a list of `Trial` objects.
* It ensures the integrity and correctness of the model building process in terms of trial outputs.

### IMPL
* Prepare the necessary environment and instantiate the `RidgeModelBuilder` class with mock data entities.
* Define the input parameters for the `build_models` method as specified in the test case.
* Mock the `Convergence` class's `calculate_convergence` method to provide consistent convergence outputs.
* Mock the `RidgeModelBuilder` class's `_select_best_model` method to simulate the selection of the best model.
* Call the `build_models` method with the prepared input parameters.
* Access the `trials` attribute from the returned `ModelOutputs` object.
* Assert that `trials` is a list and that each element in `trials` is an instance of the `Trial` class, confirming proper handling and construction of trials.

## `test_model_outputs_select_id() -> None`
### USAGE
* This test function verifies that the `select_id` attribute in the `ModelOutputs` returned from the `build_models` method matches the expected `best_model_id`.
* It validates the model selection process, ensuring the correct best model is identified and returned.

### IMPL
* Set up the test environment by creating mock entities and instantiating the `RidgeModelBuilder` class.
* Define the input parameters for the `build_models` method based on the test case details.
* Utilize mocking for the `Convergence` class's `calculate_convergence` method to return a predefined set of messages.
* Mock the `_select_best_model` method of `RidgeModelBuilder` to return a specific `best_model_id`.
* Execute the `build_models` method using the prepared parameters.
* Retrieve the `select_id` attribute from the `ModelOutputs` object returned by the method.
* Use assertions to check that the `select_id` matches the predefined `best_model_id`, verifying the correctness of model selection.

## `test_model_outputs_instance() -> None`
### USAGE
* This test verifies if the `build_models` method of the `RidgeModelBuilder` class returns an instance of `ModelOutputs`.
### IMPL
* Initialize the `RidgeModelBuilder` object with mock data entities such as `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Mock the `calculate_convergence` method of the `Convergence` class to return a predefined output.
* Mock the `_select_best_model` method of the `RidgeModelBuilder` class to return a predefined model identifier.
* Define the input parameters for the `build_models` method, adhering to the structure provided in the test case.
* Invoke the `build_models` method with the defined parameters.
* Assert that the returned object is an instance of `ModelOutputs`.

## `test_ts_validation_flag() -> None`
### USAGE
* This test checks if the `ts_validation` attribute in the returned `ModelOutputs` instance is set to `True` when time-series validation is enabled.
### IMPL
* Initialize the `RidgeModelBuilder` object with mock data entities such as `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Mock the `calculate_convergence` method of the `Convergence` class to return a predefined output.
* Mock the `_select_best_model` method of the `RidgeModelBuilder` class to return a predefined model identifier.
* Define the input parameters for the `build_models` method, ensuring that the `ts_validation` parameter is set to `True`.
* Invoke the `build_models` method with the defined parameters.
* Retrieve the `ts_validation` attribute from the returned `ModelOutputs` instance.
* Assert that the `ts_validation` attribute is `True`.

## `test_model_outputs_instance() -> None`
### USAGE
* This test verifies if the `build_models` method returns an instance of `ModelOutputs`.
* It ensures that the output type of the method matches the expected class type.
### IMPL
* *Mock Dependencies:*
  * Mock the `Convergence.calculate_convergence` method to return a predefined convergence message list.
  * Mock the `RidgeModelBuilder._select_best_model` method to return a predefined model identifier.
* *Invoke the Method:*
  * Call the `build_models` method of `RidgeModelBuilder` with the specified test input parameters.
* *Assertions:*
  * Assert that the return value of `build_models` is an instance of `ModelOutputs`.

## `test_add_penalty_factor_flag() -> None`
### USAGE
* This test checks if the `add_penalty_factor` attribute of the `ModelOutputs` object is correctly set to `True`.
* It ensures that the penalty factor is applied when specified in the input.
### IMPL
* *Mock Dependencies:*
  * Mock the `Convergence.calculate_convergence` method to return a predefined convergence message list.
  * Mock the `RidgeModelBuilder._select_best_model` method to return a predefined model identifier.
* *Invoke the Method:*
  * Call the `build_models` method of `RidgeModelBuilder` with the specified test input parameters.
* *Assertions:*
  * Assert that the `add_penalty_factor` attribute of the resulting `ModelOutputs` object is `True`.

## `test_model_outputs_instance(trials_config: TrialsConfig, ...) -> None`
### USAGE
* This function tests whether the result of the `build_models` method is an instance of `ModelOutputs`.
* It uses a mocked version of the `calculate_convergence` and `_select_best_model` methods to ensure the unit test focuses on the behavior of the `build_models` method rather than its dependencies.

### IMPL
* Begin by importing necessary modules and setting up the test environment.
* Use a mocking library to replace the `calculate_convergence` method of the `Convergence` dependency with a mock that returns a predefined output. This avoids testing the actual convergence calculation logic.
* Similarly, mock the `_select_best_model` method to return a fixed 'best_model_id' to avoid testing model selection logic.
* Create an instance of `RidgeModelBuilder` with mock data and configurations.
* Call the `build_models` method with the provided `trials_config` and other parameters, including the mocked `nevergrad_algo`.
* Capture the output of the method and assert that it is an instance of `ModelOutputs`.
* Verify that the mocked methods were called with expected arguments to ensure correct interactions.

## `test_nevergrad_algorithm(trials_config: TrialsConfig, ...) -> None`
### USAGE
* This function tests whether the `nevergrad_algo` attribute of the `ModelOutputs` returned by the `build_models` method is correctly set to `NevergradAlgorithm.CMA`.

### IMPL
* Set up the test environment and necessary imports, focusing on mocking dependencies.
* Mock the `calculate_convergence` method of the `Convergence` class to return a fixed output, allowing the test to concentrate on the `build_models` logic.
* Mock the `_select_best_model` method to return a predetermined 'best_model_id'.
* Initialize a `RidgeModelBuilder` instance using mock data and configurations.
* Invoke the `build_models` method with the specified `trials_config` and other parameters, including the `nevergrad_algo` set to `NevergradAlgorithm.CMA`.
* Store the method's output and perform an assertion to check if the `nevergrad_algo` attribute of the `ModelOutputs` is equal to `NevergradAlgorithm.CMA`.
* Ensure the mocked methods received the correct calls to confirm the method's interactions were as expected.

## `test_model_outputs_instance() -> None`
### USAGE
* This test function is designed to verify that the `build_models` method returns an output that is an instance of the `ModelOutputs` class.
### IMPL
* Begin by importing necessary testing libraries such as `unittest` and `unittest.mock`.
* Mock dependencies that the `RidgeModelBuilder` relies on. Specifically, mock the `Convergence` and `RidgeModelBuilder` class methods that interact with external factors.
* Create a mock for the `Convergence` class's `calculate_convergence` method. Configure it to return a predefined convergence message when called with a list of `Trial` objects.
* Mock the `_select_best_model` method in the `RidgeModelBuilder` class to return a specific model ID, as expected in the test scenario.
* Instantiate the `RidgeModelBuilder` using mocked data entities like `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Call the `build_models` method on the instantiated `RidgeModelBuilder` with the input parameters specified in the test case, ensuring the `intercept` parameter is set to `False`.
* Capture the output of the `build_models` method.
* Use an assertion to check that the output is an instance of the `ModelOutputs` class.

## `test_model_outputs_intercept() -> None`
### USAGE
* This test function aims to confirm that the `intercept` attribute of the `ModelOutputs` instance returned by the `build_models` method is set to `False`.
### IMPL
* Start by importing required testing libraries and utilities, such as `unittest` and `unittest.mock`.
* Mock external dependencies used by the `RidgeModelBuilder`. Specifically, mock the `Convergence` class's `calculate_convergence` method and the `RidgeModelBuilder` class's `_select_best_model` method.
* Configure the `Convergence` mock to return a predetermined convergence message when called with a list of `Trial` objects.
* Set up the `_select_best_model` mock to return a specific model ID, as expected in the test context.
* Instantiate the `RidgeModelBuilder` using mocked data entities like `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Invoke the `build_models` method on the created `RidgeModelBuilder` instance, utilizing the input parameters defined in the test case, ensuring the `intercept` parameter is `False`.
* Capture the output of the `build_models` method.
* Perform an assertion to verify that the `intercept` attribute of the captured output is `False`.

## `test_select_best_model_returns_correct_model(output_models: List[Trial]) -> None`
### USAGE
* This function tests the `_select_best_model` method of the `RidgeModelBuilder` class.
* The function checks if the method correctly identifies and returns the solution ID of the best model from a list of trials.
* The input parameter `output_models` is a list containing model trials with associated metrics.

### IMPL
1. *Setup Input*:
   - Prepare a list of `Trial` objects as `output_models` that each contain attributes like `nrmse`, `decomp_rssd`, and `result_hyp_param`.
   - Ensure these trials have varying `nrmse` and `decomp_rssd` values to simulate real-world model outputs.

2. *Invoke Method*:
   - Call the `_select_best_model` method of the `RidgeModelBuilder` class, passing in the `output_models` list.

3. *Assertion*:
   - Check that the returned value from the `_select_best_model` method is equal to the expected `solID` of the trial with the lowest combined score.
   - The expected `solID` should match "model_1", which is the `sol_id` of the model with the minimum NRMSE and decomp RSSD as per the test input.

4. *Verification*:
   - Verify that the function correctly identifies the model with the `solID` "model_1" when it has the lowest normalized combined score among all trials.

5. *No External Mocks Needed*:
   - Since this method does not rely on external dependencies, no mocking is required.

## `test_select_best_model_returns_correct_sol_id() -> None`
### USAGE
* This test function checks if the `_select_best_model` method correctly identifies and returns the `sol_id` of the best model based on the given criteria.
* Input: A list of mocked `Trial` objects with distinct `nrmse` and `decomp_rssd` values.
* Expected Output: The `sol_id` of the model with the lowest combined normalized score.

### IMPL
* Create a list named `output_models` containing three mocked objects of type `Trial`.
* Ensure each mocked `Trial` object has attributes for `nrmse`, `decomp_rssd`, and `result_hyp_param` set according to the input JSON.
  * First mock: `nrmse=0.2`, `decomp_rssd=0.1`, `result_hyp_param={"solID": ["model_1"]}`, `sol_id="model_1"`.
  * Second mock: `nrmse=0.15`, `decomp_rssd=0.09`, `result_hyp_param={"solID": ["model_2"]}`, `sol_id="model_2"`.
  * Third mock: `nrmse=0.1`, `decomp_rssd=0.08`, `result_hyp_param={"solID": ["model_3"]}`, `sol_id="model_3"`.
* Instantiate the `RidgeModelBuilder` class with any necessary dependencies.
* Call the `_select_best_model` method on the `RidgeModelBuilder` instance, passing in the `output_models` list.
* Capture the returned value in a variable named `selected_model_id`.
* Assert that the `selected_model_id` is equal to "model_3", as it's expected to have the lowest combined normalized score.

## `test_select_best_model_return(output_models: List[Dict[str, Any]]) -> None`
### USAGE
* This test verifies that the `_select_best_model` method correctly returns the best model's `solID` when there are multiple trials with identical NRMSE and decomp RSSD values.
* Parameters:
  - `output_models`: A list of dictionaries that represent the output models from each trial, including metrics such as `nrmse`, `decomp_rssd`, and `result_hyp_param`.

### IMPL
* Initialize a list `output_models` containing two dictionaries, each representing a model trial.
* Each dictionary should have identical `nrmse` and `decomp_rssd` values but different `result_hyp_param` with distinct `solID`.
* Create an instance of the `RidgeModelBuilder` class. This instance will be used to call the method being tested.
* Call the `_select_best_model` method on the `RidgeModelBuilder` instance with `output_models` as the argument.
* Capture the returned value from the method, which should be the `solID` of the best model.
* Assert that the returned `solID` equals the expected value, which is "model_1" since both models have the same metrics and the first one should be selected by default.

## `test_select_best_model_returns_correct_model_id() -> None`
### USAGE
* This unit test function verifies that the `_select_best_model` method of the `RidgeModelBuilder` class correctly identifies and returns the `solID` of the best model based on the provided list of `Trial` objects.
### IMPL
* Create a list of mock `Trial` objects representing the output models. Each `Trial` object should contain attributes `nrmse`, `decomp_rssd`, `result_hyp_param`, and `sol_id`, mimicking the structure expected by the `_select_best_model` method.
* Populate the `output_models` list with at least two `Trial` objects, each with different `nrmse` and `decomp_rssd` values to ensure variability in the selection process.
* Instantiate the `RidgeModelBuilder` class. This can be done with mock or dummy data for the constructor parameters, as they are not directly used in the `_select_best_model` method.
* Call the `_select_best_model` method on the `RidgeModelBuilder` instance, passing in the `output_models` list as the argument.
* Capture the result returned by the method. This should be a string representing the `solID` of the selected best model.
* Assert that the captured result matches the expected `solID` of the model with the lowest combined score (sum of normalized `nrmse` and `decomp_rssd`).
* Ensure the test covers edge cases such as the possibility of tied scores by differentiating models with unique identifiers.

## `test_select_best_model_returns_null_with_empty_trials() -> None`
### USAGE
* This test function is designed to verify that the `_select_best_model` method returns `null` when provided with an empty list of trials (`output_models`).
* The function will handle an edge case scenario where no trials have been conducted, thus testing the robustness of the method when handling empty inputs.

### IMPL
* Create an instance of `RidgeModelBuilder` using necessary mock or dummy objects for `mmm_data`, `holiday_data`, `calibration_input`, `hyperparameters`, and `featurized_mmm_data`.
* Prepare an empty list of trials, `output_models`, to simulate the case where no trials are available.
* Invoke the `_select_best_model` method on the `RidgeModelBuilder` instance, passing the empty `output_models` list as an argument.
* Assert that the return value of `_select_best_model` is `None`, confirming that the method handles empty input by returning `null`.
* Ensure that there is no exception or error raised during the method execution, indicating proper handling of the edge case.

## `test_model_train_length_of_trials() -> None`
### USAGE
* This test ensures that the number of trials generated during the model training process matches the expected value.
* It is based on the input configuration and the expected outcome is to verify the length of trials.

### IMPL
* Mock the `_run_nevergrad_optimization` method of the `RidgeModelBuilder` class:
  - The method should be mocked to return a predefined output for each trial, simulating successful trial results.
  - Ensure the mock returns a list of `Trial` objects with the specified success criteria.

* Instantiate the `RidgeModelBuilder` using mock or dummy data:
  - Prepare necessary mock objects for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
  - Use these mocks to create an instance of `RidgeModelBuilder`.

* Define the input parameters based on the test case:
  - Create a `hyper_collect` dictionary with the specified `alpha` value.
  - Create a `TrialsConfig` object with 5 trials and 10 iterations as specified in the input.
  - Set other parameters such as `intercept_sign`, `intercept`, `nevergrad_algo`, etc., using the test case values.

* Call the `build_models` method of `RidgeModelBuilder`:
  - Pass all necessary parameters including the `TrialsConfig` instance and other configurations as defined in the test input.

* Retrieve the result from the `build_models` method:
  - Capture the `ModelOutputs` returned by the method, which contains the trials information.

* Perform an assertion on the length of trials:
  - Extract the `trials` attribute from the `ModelOutputs`.
  - Verify that the length of the `trials` list matches the expected value of 5, ensuring that the correct number of trials were executed during model training.

* Clean up any resources or mocks as necessary.

## `test_length_of_trials() -> None`
### USAGE
* This test function checks that the `build_models` method in the `RidgeModelBuilder` class runs the expected number of trials as specified in the input configuration.
### IMPL
* Import the necessary testing modules and classes such as `unittest` and `unittest.mock`.
* Define the test function `test_length_of_trials`.
* Set up the environment by creating an instance of `RidgeModelBuilder` with mocked data dependencies like `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Prepare the input parameters for the `build_models` method, including `trials_config` and other relevant configurations.
* Mock the `_run_nevergrad_optimization` method in `RidgeModelBuilder` to ensure it returns a predictable and controlled output. This method will be mocked to return a list of `Trial` instances, each containing necessary attributes.
* Call the `build_models` method with the prepared input parameters.
* Capture the output of the `build_models` method, specifically focusing on the list of trials returned in the `ModelOutputs`.
* Assert that the length of the trials list matches the expected value (3 in this case), which indicates that the method executed the correct number of trials.
* Use assertions to validate that the number of trials aligns with the `expectedValue` from the test specification.
* Clean up any mock objects or changes to the environment after the test execution.

## `test_model_build_length_of_trials() -> None`
### USAGE
* This test function checks if the number of trials generated when building models is equal to the expected value.
* It validates the length of the trials list in the `ModelOutputs` returned by the `build_models` method.
### IMPL
* *Step 1:* Import necessary testing libraries and classes, including `unittest`, `mock`, and the `RidgeModelBuilder` class.
* *Step 2:* Define the test function `test_model_build_length_of_trials`.
* *Step 3:* Use the `mock.patch` to mock the `_run_nevergrad_optimization` method in `RidgeModelBuilder`. This method is responsible for running optimizations and will be mocked to return a predefined trial output.
* *Step 4:* Set up the mock to return a list of mocked trial objects. This is achieved by defining the return value of the `_run_nevergrad_optimization` method to be a list containing a number of trial objects equal to the expected number of trials.
* *Step 5:* Create an instance of `RidgeModelBuilder` with mock objects for its dependencies, such as `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* *Step 6:* Configure the input parameters for the `build_models` method:
  - `trials_config`: An object defining the number of trials and iterations.
  - `dt_hyper_fixed`: A DataFrame with fixed hyperparameters, if needed.
  - Other parameters as per the test case requirements.
* *Step 7:* Call the `build_models` method on the `RidgeModelBuilder` instance with the configured input parameters.
* *Step 8:* Capture the `ModelOutputs` returned from the `build_models` method.
* *Step 9:* Assert that the length of the trials list within the `ModelOutputs` matches the expected number of trials specified in the test case.
* *Step 10:* Clean up any mocks or other test setup to ensure isolation between tests.

## `test_length_of_trials() -> None`
### USAGE
* This test function verifies that the number of trials produced by the `_model_train` method in the `RidgeModelBuilder` class matches the expected value.
* The test ensures that when the model training process is executed with a specified number of trials, the resulting trials list has the correct length.

### IMPL
* *Setup Mocks:*
  * Mock the `_run_nevergrad_optimization` method of the `RidgeModelBuilder` class to control its output.
  * Configure the mock to return a predefined `Trial` object with attributes like `trial_id` and `success` set to known values.

* *Prepare Input:*
  * Create a `hyper_collect` dictionary with necessary hyperparameter configurations. This includes parameters such as `alpha`, `iterations`, `cores`, etc., that are needed for the training process.
  * Define a `TrialsConfig` object with `trials` set to 3 and `iterations` set to 10, indicating the number of trials and iterations per trial to be performed.
  * Set additional parameters such as `intercept`, `intercept_sign`, `nevergrad_algo`, `dt_hyper_fixed`, `ts_validation`, `add_penalty_factor`, `objective_weights`, `rssd_zero_penalty`, `seed`, and `cores` as required by the `_model_train` method.

* *Invoke Method:*
  * Call the `_model_train` method of the `RidgeModelBuilder` instance using the prepared input configurations.

* *Assertion:*
  * Check the length of the list returned by the `_model_train` method.
  * Assert that the length is equal to 3, which is the expected number of trials specified in the test input.

* *Cleanup:*
  * Ensure that any resources or processes started during the test are properly cleaned up, such as stopping any mock objects or resetting configurations.

## `test_length_of_trials() -> None`
### USAGE
* This unit test verifies that the number of trials produced by the `_model_train` method is as expected.
* It checks whether the length of the trials list matches the specified number of trials in the configuration.

### IMPL
* *Mock Setup for Dependencies:*
  * Mock the `RidgeModelBuilder._run_nevergrad_optimization` method.
  * The mock is set up to simulate expected behavior of the actual method.
  * For the mock setup:
    * Input parameters include `hyper_collect`, `iterations`, `cores`, `nevergrad_algo`, `intercept`, `intercept_sign`, `ts_validation`, `add_penalty_factor`, `objective_weights`, `dt_hyper_fixed`, `rssd_zero_penalty`, `trial`, `seed`, and `total_trials`.
    * The method is expected to return a `Trial` object with `trial_id` as 1 and `success` as `True`.

* *Test Execution:*
  * Instantiate the `RidgeModelBuilder` class with appropriate mock objects for `mmm_data`, `holiday_data`, `calibration_input`, `hyperparameters`, and `featurized_mmm_data`.
  * Prepare the `hyper_collect` dictionary with a key `alpha` set to 0.5, as specified in the test case input.
  * Define `trials_config` with `trials` set to 3 and `iterations` set to 10.
  * Set other parameters such as `intercept_sign`, `intercept`, `nevergrad_algo`, `dt_hyper_fixed`, `ts_validation`, `add_penalty_factor`, `objective_weights`, `rssd_zero_penalty`, `seed`, and `cores` as described in the test case input.
  * Call the `_model_train` method with the above parameters.
  * Capture the output, which is expected to be a list of `Trial` objects.

* *Assertions:*
  * Assert that the length of the list of `Trial` objects returned by `_model_train` matches the expected value of 3, as specified in the test case assertion.

## `test_trial_object_with_expected_properties() -> None`
### USAGE
* This test verifies that the `_run_nevergrad_optimization` method produces a `Trial` object with expected properties.
* It checks the correctness of the trial object returned by the method under test conditions.
### IMPL
* Begin by setting up the necessary environment for the test.
* Mock the `ng.optimizers.registry` to return a `MockOptimizer` when `TWO_POINTS_DE` is invoked.
* Ensure that the `MockOptimizer` instance will:
  * Return a candidate with `kwargs` containing `param1` set to 0.5 when `ask` is called.
  * Accept any candidate and result when `tell` is called, without returning any specific output.
* Prepare the input parameters for the `_run_nevergrad_optimization` method:
  * Define `hyper_collect` with `hyper_bound_list_updated` containing `param1` within bounds [0.0, 1.0].
  * Set `iterations` to 1 and `cores` to 1.
  * Use `NevergradAlgorithm.TWO_POINTS_DE` for the `nevergrad_algo`.
  * Set other parameters like `intercept`, `intercept_sign`, `ts_validation`, `add_penalty_factor`, `objective_weights`, `dt_hyper_fixed`, `rssd_zero_penalty`, `trial`, `seed`, and `total_trials` as described in the test case.
* Invoke the `_run_nevergrad_optimization` method with the prepared parameters.
* Capture the returned `Trial` object.
* Assert that the `Trial` object has the expected properties:
  * Verify that the `trial` attribute matches the expected trial number from the input.
  * Ensure that the hyperparameter values, especially `param1`, match expected values within the result.
  * Confirm that metrics like `nrmse`, `decomp_rssd`, and `mape` fall within expected ranges or match specific expected outputs.
* Log any pertinent information regarding the test execution to assist in debugging if necessary.

## `test_trial_object_properties()`
### USAGE
* This test function verifies that the `Trial` object returned by the `test_run_nevergrad_optimization` function possesses the expected properties.
* The primary goal is to ensure that the optimization process yields a trial with correct attributes reflecting the optimization results.

### IMPL
* *Setup Mocks for Dependencies*:
  * Mock the `TWO_POINTS_DE` method from `ng.optimizers.registry` to return `MockOptimizer` to simulate the optimization process.
  * Mock `MockOptimizer.ask` to simulate retrieving a candidate with parameters `{'param1': 0.5}`.
  * Mock `MockOptimizer.tell` to simulate the process of providing feedback to the optimizer with the candidate and result `0.1`.
* *Initialize Input Parameters*:
  * Provide the `hyper_collect` dictionary with updated hyperparameter bounds for optimization.
  * Set the number of `iterations` to `5` and `cores` to `2` to test multi-core and multi-iteration scenarios.
  * Use the `nevergrad_algo` as `TWO_POINTS_DE` for the optimization algorithm.
  * Set additional parameters such as `intercept` to `true` and `intercept_sign` to `non_negative`.
* *Invoke the Function*:
  * Call `test_run_nevergrad_optimization` with the prepared inputs and mocks.
* *Verify Output*:
  * Assert that the returned object is of type `Trial`.
  * Assert that the `Trial` object contains the expected properties and values:
    * Verify the `result_hyp_param` contains expected solution IDs and parameters.
    * Check `decomp_spend_dist`, `x_decomp_agg`, and other attributes for expected structure and correctness.
  * Ensure the trial number matches the expected trial number given in the input.

## `test_trial_object_with_expected_properties() -> None`
### USAGE
* This unit test verifies that the `Trial` object returned by the `_run_nevergrad_optimization` method has the expected properties based on the given input configuration for a single trial execution.
* The test ensures that the `Trial` object is correctly instantiated and populated with values as defined by the mocked responses and method logic.

### IMPL
* *Setup Mocking for Dependencies*:
  - Mock the `ng.optimizers.registry` to return a mock optimizer that simulates the behavior of the `TWO_POINTS_DE` Nevergrad algorithm. Ensure the mock optimizer is configured to accept the specified instrumentation and budget.
  - Mock the `ask` method of the mock optimizer to simulate requesting a candidate parameter set. Return a dictionary with a known structure that includes parameters like `param1`.
  - Mock the `tell` method of the mock optimizer to simulate the process of updating the optimizer with the candidate parameters and the result obtained from the model evaluation.

* *Prepare Input Arguments*:
  - Define a dictionary `hyper_collect` with the necessary structure, including a `hyper_bound_list_updated` with parameter bounds.
  - Set up other input parameters like `iterations`, `cores`, `nevergrad_algo`, `intercept`, `intercept_sign`, `ts_validation`, `add_penalty_factor`, `objective_weights`, `dt_hyper_fixed`, `rssd_zero_penalty`, `trial`, `seed`, and `total_trials` to match the test case requirements.

* *Invoke Method Under Test*:
  - Call the `_run_nevergrad_optimization` method with the prepared arguments and mock dependencies.

* *Assertions*:
  - Verify that the returned `Trial` object has the expected properties by checking:
    - The `result_hyp_param` DataFrame is populated with expected IDs and metrics.
    - The aggregated `decomp_spend_dist` and `x_decomp_agg` DataFrames contain expected values.
    - The metrics such as `nrmse`, `decomp_rssd`, `mape`, and others are calculated and stored correctly.
    - Ensure the `trial` attribute of the `Trial` object matches the input trial number.
    - Confirm that the `sol_id` attribute is constructed as expected, combining trial and iteration identifiers.

## `test_run_nevergrad_optimization_trial_creation() -> None`
### USAGE
* This function tests the creation of a trial object with expected properties from the `_run_nevergrad_optimization` method.
* It verifies that the trial object returned by the method contains correct data based on the test input parameters and mocked dependencies.
### IMPL
* *Mock `ng.optimizers.registry.TWO_POINTS_DE`:*
  * Simulate the Nevergrad optimizer to avoid actual optimization computation. Return a mock optimizer object named `MockOptimizer`.
* *Mock `MockOptimizer.ask`:*
  * Simulate the behavior of the optimizer's `ask` method to return predefined parameters, for example, `{'param1': 0.5}`.
* *Mock `MockOptimizer.tell`:*
  * Simulate the behavior of the optimizer's `tell` method to simply accept the candidate and result without performing any operations.
* *Prepare test inputs:*
  * Use the `hyper_collect` dictionary with the required hyperparameters and bounds.
  * Set iterations to 5, cores to 2, and other relevant parameters like `intercept`, `intercept_sign`, `ts_validation`, `add_penalty_factor`, `objective_weights`, `rssd_zero_penalty`, `trial`, `seed`, and `total_trials`.
* *Invoke `_run_nevergrad_optimization`:*
  * Call the method with the prepared inputs and mocked dependencies.
* *Assert trial object creation:*
  * Verify that the returned trial object has the expected properties:
    * Check if the trial object is an instance of the `Trial` class.
    * Confirm that the `result_hyp_param` DataFrame within the trial contains the expected 'solID'.
    * Validate other properties like `nrmse`, `decomp_rssd`, `mape`, and optimization metrics are correctly set.
* *Clean up:*
  * Remove or reset any mocks to prevent side effects on other tests.

## `test_run_nevergrad_optimization_trial_result() -> None`
### USAGE
* This test function is designed to verify that the `_run_nevergrad_optimization` method of the `RidgeModelBuilder` class correctly produces a `Trial` object with the expected properties when provided with specific inputs.
### IMPL
* Import necessary modules such as `unittest`, `mock`, and classes from the module being tested.
* Define the test function `test_run_nevergrad_optimization_trial_result`.
* Mock the `ng.optimizers.registry.TWO_POINTS_DE` to return a `MockOptimizer` object.
* Mock the methods `ask` and `tell` of `MockOptimizer`:
  * `ask` should return a dictionary with `kwargs` containing `param1` set to `0.5`.
  * `tell` should accept a candidate with `param1` as `0.5` and a result of `0.1`.
* Instantiate the `RidgeModelBuilder` class with mock dependencies, including `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Prepare the input parameters for `_run_nevergrad_optimization`:
  * Use a dictionary `hyper_collect` with `hyper_bound_list_updated` containing hyperparameter bounds.
  * Set `iterations` to `5` and `cores` to `2`.
  * Use `NevergradAlgorithm.TWO_POINTS_DE` for `nevergrad_algo`.
  * Set `intercept` to `True` and `intercept_sign` to `"non_negative"`.
  * Set `ts_validation`, `add_penalty_factor`, and `rssd_zero_penalty` to `False`.
  * Define `objective_weights` as `None`.
  * Use a mock DataFrame `dt_hyper_fixed`.
  * Set `trial` to `1`, `seed` to `123`, and `total_trials` to `1`.
* Call the `_run_nevergrad_optimization` method with the prepared inputs.
* Assert that the result is a `Trial` object and that it contains the expected properties as specified in the test case.
* Use mock assertions to verify that `MockOptimizer.ask` and `MockOptimizer.tell` are called with expected arguments.

## `test_rsq_train_calculation() -> None`
### USAGE
* Tests the calculation of the R-squared (R²) score for the training data within the `_calculate_decomp_spend_dist` method.
* Verifies that the calculated R² score matches the expected value from the assertions.

### IMPL
* *Step 1*: Instantiate a Ridge model with predefined coefficients.
  - Ensure the model is properly initialized with the provided lambda parameter.
* *Step 2*: Prepare a DataFrame `X` containing columns for paid media spends.
  - Populate this DataFrame with realistic test data aligning with the function requirements.
* *Step 3*: Prepare a Series `y` representing the target values.
  - Ensure this target variable is consistent with the features in `X`.
* *Step 4*: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
  - Use the parameters provided in the test case input.
* *Step 5*: Call `_calculate_decomp_spend_dist` with the model, `X`, `y`, and `params`.
  - Capture the output DataFrame which includes `rsq_train`.
* *Step 6*: Extract the `rsq_train` value from the output DataFrame.
  - Ensure this extraction aligns with the DataFrame's structure.
* *Step 7*: Assert that the extracted `rsq_train` value matches the expected R² score for training data.
  - Use the expected value specified in the test case assertions.

## `test_nrmse_train_calculation() -> None`
### USAGE
* Tests the calculation of the normalized Root Mean Square Error (NRMSE) for the training data.
* Verifies that the calculated NRMSE matches the expected value from the assertions.

### IMPL
* *Step 1*: Instantiate a Ridge model with predefined coefficients.
  - Ensure the model is properly initialized with the provided lambda parameter.
* *Step 2*: Prepare a DataFrame `X` containing columns for paid media spends.
  - Populate this DataFrame with realistic test data aligning with the function requirements.
* *Step 3*: Prepare a Series `y` representing the target values.
  - Ensure this target variable is consistent with the features in `X`.
* *Step 4*: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
  - Use the parameters provided in the test case input.
* *Step 5*: Call `_calculate_decomp_spend_dist` with the model, `X`, `y`, and `params`.
  - Capture the output DataFrame which includes `nrmse_train`.
* *Step 6*: Extract the `nrmse_train` value from the output DataFrame.
  - Ensure this extraction aligns with the DataFrame's structure.
* *Step 7*: Assert that the extracted `nrmse_train` value matches the expected NRMSE for training data.
  - Use the expected value specified in the test case assertions.

## `test_coef_calculation() -> None`
### USAGE
* Tests the extraction and calculation of coefficients corresponding to paid media columns.
* Verifies that the coefficients match the expected values from the assertions.

### IMPL
* *Step 1*: Instantiate a Ridge model with predefined coefficients.
  - Ensure the model is properly initialized with the provided lambda parameter.
* *Step 2*: Prepare a DataFrame `X` containing columns for paid media spends.
  - Populate this DataFrame with realistic test data aligning with the function requirements.
* *Step 3*: Prepare a Series `y` representing the target values.
  - Ensure this target variable is consistent with the features in `X`.
* *Step 4*: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
  - Use the parameters provided in the test case input.
* *Step 5*: Call `_calculate_decomp_spend_dist` with the model, `X`, `y`, and `params`.
  - Capture the output DataFrame which includes `coef`.
* *Step 6*: Extract the `coef` values from the output DataFrame.
  - Ensure this extraction aligns with the DataFrame's structure.
* *Step 7*: Assert that the extracted `coef` values match the expected coefficients for paid media columns.
  - Use the expected values specified in the test case assertions.

## `test_xDecompAgg_calculation() -> None`
### USAGE
* Tests the calculation of the sum of decomposed contributions within the `_calculate_x_decomp_agg` method.
* Verifies that the decomposed aggregate matches the expected value from the assertions.

### IMPL
* *Step 1*: Instantiate a Ridge model with predefined coefficients.
  - Ensure the model is properly initialized with the provided lambda parameter.
* *Step 2*: Prepare a DataFrame `X` containing columns for paid media spends.
  - Populate this DataFrame with realistic test data aligning with the function requirements.
* *Step 3*: Prepare a Series `y` representing the target values.
  - Ensure this target variable is consistent with the features in `X`.
* *Step 4*: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
  - Use the parameters provided in the test case input.
* *Step 5*: Call `_calculate_x_decomp_agg` with the model, `X`, `y`, and `params`.
  - Capture the output DataFrame which includes `xDecompAgg`.
* *Step 6*: Extract the `xDecompAgg` value from the output DataFrame.
  - Ensure this extraction aligns with the DataFrame's structure.
* *Step 7*: Assert that the extracted `xDecompAgg` value matches the expected sum of decomposed contributions.
  - Use the expected value specified in the test case assertions.

## `test_xDecompMeanNon0_calculation() -> None`
### USAGE
* Tests the calculation of the mean of non-zero decomposed contributions.
* Verifies that the mean value matches the expected value from the assertions.

### IMPL
* *Step 1*: Instantiate a Ridge model with predefined coefficients.
  - Ensure the model is properly initialized with the provided lambda parameter.
* *Step 2*: Prepare a DataFrame `X` containing columns for paid media spends.
  - Populate this DataFrame with realistic test data aligning with the function requirements.
* *Step 3*: Prepare a Series `y` representing the target values.
  - Ensure this target variable is consistent with the features in `X`.
* *Step 4*: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
  - Use the parameters provided in the test case input.
* *Step 5*: Call `_calculate_x_decomp_agg` with the model, `X`, `y`, and `params`.
  - Capture the output DataFrame which includes `xDecompMeanNon0`.
* *Step 6*: Extract the `xDecompMeanNon0` value from the output DataFrame.
  - Ensure this extraction aligns with the DataFrame's structure.
* *Step 7*: Assert that the extracted `xDecompMeanNon0` value matches the expected mean of non-zero decomposed contributions.
  - Use the expected value specified in the test case assertions.

## `test_calculate_decomp_spend_dist_rsq_train_with_empty_data() -> None`
### USAGE
* This test checks that the function `_calculate_decomp_spend_dist` returns a `rsq_train` of `0` when provided with an empty DataFrame and Series.
### IMPL
* Instantiate a Ridge model with predefined coefficients.
* Create an empty DataFrame for features `X`.
* Create an empty Series for target `y`.
* Define a `params` dictionary with all parameter values set to `0`.
* Call `_calculate_decomp_spend_dist` with the Ridge model, empty DataFrame, empty Series, and the params.
* Extract the `rsq_train` value from the returned DataFrame.
* Assert that the `rsq_train` value equals `0`.

## `test_calculate_decomp_spend_dist_nrmse_train_with_empty_data() -> None`
### USAGE
* This test checks that the function `_calculate_decomp_spend_dist` returns `nrmse_train` as `NaN` when provided with an empty DataFrame and Series.
### IMPL
* Instantiate a Ridge model with predefined coefficients.
* Create an empty DataFrame for features `X`.
* Create an empty Series for target `y`.
* Define a `params` dictionary with all parameter values set to `0`.
* Call `_calculate_decomp_spend_dist` with the Ridge model, empty DataFrame, empty Series, and the params.
* Extract the `nrmse_train` value from the returned DataFrame.
* Assert that the `nrmse_train` value is `NaN`.

## `test_calculate_decomp_spend_dist_coef_with_empty_data() -> None`
### USAGE
* This test checks that the function `_calculate_decomp_spend_dist` returns an empty array for `coef` when provided with an empty DataFrame and Series.
### IMPL
* Instantiate a Ridge model with predefined coefficients.
* Create an empty DataFrame for features `X`.
* Create an empty Series for target `y`.
* Define a `params` dictionary with all parameter values set to `0`.
* Call `_calculate_decomp_spend_dist` with the Ridge model, empty DataFrame, empty Series, and the params.
* Extract the `coef` array from the returned DataFrame.
* Assert that the `coef` array is empty.

## `test_calculate_decomp_spend_dist_xDecompAgg_with_empty_data() -> None`
### USAGE
* This test checks that the function `_calculate_decomp_spend_dist` returns `xDecompAgg` as `NaN` when provided with an empty DataFrame and Series.
### IMPL
* Instantiate a Ridge model with predefined coefficients.
* Create an empty DataFrame for features `X`.
* Create an empty Series for target `y`.
* Define a `params` dictionary with all parameter values set to `0`.
* Call `_calculate_decomp_spend_dist` with the Ridge model, empty DataFrame, empty Series, and the params.
* Extract the `xDecompAgg` value from the returned DataFrame.
* Assert that the `xDecompAgg` value is `NaN`.

## `test_calculate_decomp_spend_dist_xDecompMeanNon0_with_empty_data() -> None`
### USAGE
* This test checks that the function `_calculate_decomp_spend_dist` returns `xDecompMeanNon0` as `NaN` when provided with an empty DataFrame and Series.
### IMPL
* Instantiate a Ridge model with predefined coefficients.
* Create an empty DataFrame for features `X`.
* Create an empty Series for target `y`.
* Define a `params` dictionary with all parameter values set to `0`.
* Call `_calculate_decomp_spend_dist` with the Ridge model, empty DataFrame, empty Series, and the params.
* Extract the `xDecompMeanNon0` value from the returned DataFrame.
* Assert that the `xDecompMeanNon0` value is `NaN`.

## `test_coefficient_with_zero_values() -> None`
### USAGE
* This unit test aims to verify the presence of zero coefficients in the Ridge model produced by the `RidgeModelBuilder` class. The test will help ensure that the model can handle cases where some coefficients are zero, reflecting the intended behavior when certain features have no effect on the target.
* No external mock dependencies are needed for this test as it directly checks the result of model training.
### IMPL
* Initialize an instance of `RidgeModelBuilder` with appropriate mock data for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Prepare a Ridge model with some coefficients set to zero.
* Call the `_calculate_decomp_spend_dist` method using this model, a DataFrame `X` with columns representing paid media spends, and a target Series `y`.
* Extract the 'coef' column from the DataFrame returned by `_calculate_decomp_spend_dist`.
* Assert that the 'coef' array contains zero values to confirm the model's coefficients include zeros as expected.

## `test_xdecomp_agg_sum_excluding_zero() -> None`
### USAGE
* This unit test is designed to verify the correctness of the `xDecompAgg` sum calculation in the decomposition process, ensuring it accurately ignores contributions from zero coefficients. This is critical for understanding the effective impact of non-zero coefficients on the target variable.
### IMPL
* Initialize an instance of `RidgeModelBuilder` with appropriate mock data for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Prepare a Ridge model with some coefficients set to zero.
* Call the `_calculate_x_decomp_agg` method using this model, a DataFrame `X` with columns representing paid media spends, and a target Series `y`.
* Extract the 'xDecompAgg' column from the DataFrame returned by `_calculate_x_decomp_agg`.
* Calculate the expected sum of the decomposition by manually summing the contributions from non-zero coefficients.
* Assert that the calculated 'xDecompAgg' sum matches the expected sum, confirming that zero coefficients are ignored.

## `test_mean_spend_of_paid_media_columns() -> None`
### USAGE
* This unit test aims to verify the calculation of the mean spend for paid media columns, ensuring the accuracy of this metric as it plays a crucial role in interpreting the model's output in business terms.
### IMPL
* Initialize an instance of `RidgeModelBuilder` with appropriate mock data for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Use a DataFrame `X` with columns representing paid media spends, ensuring some of these columns contain non-zero values.
* Call the `_calculate_decomp_spend_dist` method using a Ridge model, the DataFrame `X`, and a target Series `y`.
* Extract the 'mean_spend' column from the DataFrame returned by `_calculate_decomp_spend_dist`.
* Calculate the expected mean spend by manually computing the average of the paid media spend columns.
* Assert that the 'mean_spend' matches the expected value to confirm the correctness of the mean calculation.

## `test_negative_coefficients_identification() -> None`
### USAGE
* This test checks whether the `_calculate_decomp_spend_dist` method correctly identifies negative coefficients in the Ridge regression model.
* It validates the output `pos` which should be an array of boolean values indicating the presence of negative coefficients.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Create a `Ridge` model with negative coefficients manually for testing.
* Prepare a mock DataFrame `X` with columns that match the expected paid media spends in the model.
* Prepare a mock Series `y` representing target values.
* Call the `_calculate_decomp_spend_dist` method with the mock model, `X`, `y`, and a dictionary of parameters.
* Extract the `pos` column from the result, which indicates whether each coefficient is positive or not.
* Assert that the `pos` column matches the expected boolean array indicating negative coefficients.

## `test_effect_share_calculation() -> None`
### USAGE
* This test checks if the `_calculate_decomp_spend_dist` method calculates the effect share correctly, even with negative effects.
* It verifies the `effect_share` result against expected values.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Create a `Ridge` model with known coefficients, including negative ones.
* Prepare a mock DataFrame `X` with columns that match the expected paid media spends in the model.
* Prepare a mock Series `y` representing target values.
* Call the `_calculate_decomp_spend_dist` method with the mock model, `X`, `y`, and a dictionary of parameters.
* Extract the `effect_share` column from the result, which represents the calculated share of each effect.
* Calculate the expected effect share manually based on the known coefficients and input data.
* Assert that the `effect_share` column matches the expected calculated share.

## `test_total_spend_calculation() -> None`
### USAGE
* This test ensures that the `_calculate_decomp_spend_dist` method correctly computes the total spend from paid media spend columns.
* It checks the `total_spend` result to confirm accuracy.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Prepare a mock DataFrame `X` with columns that are labeled as paid media spends, filled with known values.
* Prepare a mock Series `y` representing target values.
* Create a `Ridge` model that will be used as a placeholder (coefficients do not matter for this test).
* Call the `_calculate_decomp_spend_dist` method with the mock model, `X`, `y`, and a dictionary of parameters.
* Extract the `total_spend` column from the result, which represents the sum of the paid media spend columns.
* Manually calculate the total spend by summing up the values in the paid media columns of `X`.
* Assert that the `total_spend` column matches the manually calculated total spend.

## `test_rsq_val_with_missing_params() -> None`
### USAGE
* This unit test verifies that the `rsq_val` value returned by the `_calculate_decomp_spend_dist` method is 0 when the `params` dictionary is empty.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data entities for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Create a mock Ridge model with predefined coefficients.
* Prepare a mock DataFrame `X` representing feature data with columns that include paid media spends.
* Prepare a mock Series `y` representing the target values.
* Call the `_calculate_decomp_spend_dist` method with the Ridge model, DataFrame `X`, Series `y`, and an empty dictionary for `params`.
* Extract the `rsq_val` value from the resulting DataFrame.
* Assert that `rsq_val` is equal to 0, as expected when `params` is missing required entries.

## `test_rsq_test_with_missing_params() -> None`
### USAGE
* This unit test verifies that the `rsq_test` value returned by the `_calculate_decomp_spend_dist` method is 0 when the `params` dictionary is empty.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data entities for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Create a mock Ridge model with predefined coefficients.
* Prepare a mock DataFrame `X` representing feature data with columns that include paid media spends.
* Prepare a mock Series `y` representing the target values.
* Call the `_calculate_decomp_spend_dist` method with the Ridge model, DataFrame `X`, Series `y`, and an empty dictionary for `params`.
* Extract the `rsq_test` value from the resulting DataFrame.
* Assert that `rsq_test` is equal to 0, as expected when `params` is missing required entries.

## `test_nrmse_val_with_missing_params() -> None`
### USAGE
* This unit test verifies that the `nrmse_val` value returned by the `_calculate_decomp_spend_dist` method is 0 when the `params` dictionary is empty.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data entities for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Create a mock Ridge model with predefined coefficients.
* Prepare a mock DataFrame `X` representing feature data with columns that include paid media spends.
* Prepare a mock Series `y` representing the target values.
* Call the `_calculate_decomp_spend_dist` method with the Ridge model, DataFrame `X`, Series `y`, and an empty dictionary for `params`.
* Extract the `nrmse_val` value from the resulting DataFrame.
* Assert that `nrmse_val` is equal to 0, as expected when `params` is missing required entries.

## `test_nrmse_test_with_missing_params() -> None`
### USAGE
* This unit test verifies that the `nrmse_test` value returned by the `_calculate_decomp_spend_dist` method is 0 when the `params` dictionary is empty.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data entities for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Create a mock Ridge model with predefined coefficients.
* Prepare a mock DataFrame `X` representing feature data with columns that include paid media spends.
* Prepare a mock Series `y` representing the target values.
* Call the `_calculate_decomp_spend_dist` method with the Ridge model, DataFrame `X`, Series `y`, and an empty dictionary for `params`.
* Extract the `nrmse_test` value from the resulting DataFrame.
* Assert that `nrmse_test` is equal to 0, as expected when `params` is missing required entries.

## `test_lambda_with_missing_params() -> None`
### USAGE
* This unit test verifies that the `lambda` value returned by the `_calculate_decomp_spend_dist` method is 0 when the `params` dictionary is empty.
### IMPL
* Initialize the `RidgeModelBuilder` with mock data entities for `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Create a mock Ridge model with predefined coefficients.
* Prepare a mock DataFrame `X` representing feature data with columns that include paid media spends.
* Prepare a mock Series `y` representing the target values.
* Call the `_calculate_decomp_spend_dist` method with the Ridge model, DataFrame `X`, Series `y`, and an empty dictionary for `params`.
* Extract the `lambda` value from the resulting DataFrame.
* Assert that `lambda` is equal to 0, as expected when `params` is missing required entries.

## `test_train_size_parameter() -> None`
### USAGE
* This test function checks the correctness of the `train_size` parameter in the `x_decomp_agg` DataFrame returned by the `_calculate_x_decomp_agg` method.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `train_size` set to 0.8.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `train_size` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `train_size` value equals 0.8.

## `test_rsq_train_value() -> None`
### USAGE
* This test function verifies that the R-squared value for training data is close to 0.9 in the `x_decomp_agg` DataFrame.
### IMPL
* Prepare a Ridge regression model with given coefficients and fit it with `X` and `y`.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary without specific R-squared values.
* Call the `_calculate_x_decomp_agg` method.
* Extract the `rsq_train` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `rsq_train` value is approximately 0.9 using a tolerance for comparison.

## `test_rsq_val_parameter() -> None`
### USAGE
* This test function checks if the `rsq_val` parameter in the `x_decomp_agg` DataFrame is correctly set to 0.7.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `rsq_val` set to 0.7.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `rsq_val` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `rsq_val` value equals 0.7.

## `test_rsq_test_parameter() -> None`
### USAGE
* This test function verifies the `rsq_test` parameter in the `x_decomp_agg` DataFrame is correctly set to 0.6.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `rsq_test` set to 0.6.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `rsq_test` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `rsq_test` value equals 0.6.

## `test_nrmse_train_value() -> None`
### USAGE
* This test function ensures that the NRMSE value for training data is close to 0.08 in the `x_decomp_agg` DataFrame.
### IMPL
* Prepare a Ridge regression model with given coefficients and fit it with `X` and `y`.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary without specific NRMSE values.
* Call the `_calculate_x_decomp_agg` method.
* Extract the `nrmse_train` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `nrmse_train` value is approximately 0.08 using a tolerance for comparison.

## `test_nrmse_val_parameter() -> None`
### USAGE
* This test function checks if the `nrmse_val` parameter in the `x_decomp_agg` DataFrame is correctly set to 0.1.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `nrmse_val` set to 0.1.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `nrmse_val` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `nrmse_val` value equals 0.1.

## `test_nrmse_test_parameter() -> None`
### USAGE
* This test function verifies if the `nrmse_test` parameter in the `x_decomp_agg` DataFrame is correctly set to 0.15.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `nrmse_test` set to 0.15.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `nrmse_test` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `nrmse_test` value equals 0.15.

## `test_decomp_rssd_parameter() -> None`
### USAGE
* This test function checks the `decomp.rssd` parameter in the `x_decomp_agg` DataFrame is set to 0.05.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `decomp.rssd` set to 0.05.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `decomp.rssd` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `decomp.rssd` value equals 0.05.

## `test_mape_parameter() -> None`
### USAGE
* This test function checks if the `mape` parameter in the `x_decomp_agg` DataFrame is correctly set to 0.1.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `mape` set to 0.1.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `mape` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `mape` value equals 0.1.

## `test_lambda_parameter() -> None`
### USAGE
* This test function verifies if the `lambda` parameter in the `x_decomp_agg` DataFrame is correctly set to 0.01.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `lambda` set to 0.01.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `lambda` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `lambda` value equals 0.01.

## `test_lambda_hp_parameter() -> None`
### USAGE
* This test function ensures that the `lambda_hp` parameter in the `x_decomp_agg` DataFrame is set to 0.02.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `lambda_hp` set to 0.02.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `lambda_hp` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `lambda_hp` value equals 0.02.

## `test_solID_parameter() -> None`
### USAGE
* This test function verifies if the `solID` parameter in the `x_decomp_agg` DataFrame is correctly set to "test_001".
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `solID` set to "test_001".
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `solID` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `solID` value equals "test_001".

## `test_trial_parameter() -> None`
### USAGE
* This test function ensures that the `trial` parameter in the `x_decomp_agg` DataFrame is set to 1.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `trial` set to 1.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `trial` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `trial` value equals 1.

## `test_iterNG_parameter() -> None`
### USAGE
* This test function checks if the `iterNG` parameter in the `x_decomp_agg` DataFrame is correctly set to 10.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `iterNG` set to 10.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `iterNG` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `iterNG` value equals 10.

## `test_iterPar_parameter() -> None`
### USAGE
* This test function ensures that the `iterPar` parameter in the `x_decomp_agg` DataFrame is correctly set to 2.
### IMPL
* Prepare a Ridge regression model with given coefficients.
* Create a DataFrame `X` for features and a Series `y` for the target variable.
* Define the `params` dictionary including `iterPar` set to 2.
* Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`.
* Extract the `iterPar` value from the first row of the resulting `x_decomp_agg` DataFrame.
* Assert that the extracted `iterPar` value equals 2.

## `test_x_decomp_agg_sum_zero() -> None`
### USAGE
* Test that the sum of `xDecompAgg` in the resulting DataFrame is zero, indicating that the decomposition aggregates correctly when coefficients are zero.
### IMPL
* Begin by setting up the test environment and necessary inputs.
* Instantiate a `Ridge` model with zero coefficients using `np.array([0.0, 0.0, 0.0])`.
* Create a DataFrame `X` with features `feature1`, `feature2`, and `feature3`, and a Series `y` with target values.
* Define the parameters dictionary, with values for `train_size`, `rsq_val`, `rsq_test`, `nrmse_val`, `nrmse_test`, `nrmse`, `decomp_rssd`, `mape`, `lambda_`, `lambda_hp`, `lambda_max`, `lambda_min_ratio`, `solID`, `trial`, `iter_ng`, and `iter_par`.
* Call the `_calculate_x_decomp_agg` method using the instantiated `Ridge` model, DataFrame `X`, Series `y`, and the parameters dictionary.
* Obtain the resulting DataFrame from the method call, which should contain decomposition results.
* Assert that the sum of the `xDecompAgg` column in the resulting DataFrame equals `0.0`, confirming the correct handling of zero coefficients.

## `test_x_decomp_perc_sum_zero() -> None`
### USAGE
* Test that the sum of `xDecompPerc` in the resulting DataFrame is zero, verifying that the percentage decomposition aggregates correctly when coefficients are zero.
### IMPL
* Begin by setting up the test environment and necessary inputs.
* Instantiate a `Ridge` model with zero coefficients using `np.array([0.0, 0.0, 0.0])`.
* Create a DataFrame `X` with features `feature1`, `feature2`, and `feature3`, and a Series `y` with target values.
* Define the parameters dictionary, with values for `train_size`, `rsq_val`, `rsq_test`, `nrmse_val`, `nrmse_test`, `nrmse`, `decomp_rssd`, `mape`, `lambda_`, `lambda_hp`, `lambda_max`, `lambda_min_ratio`, `solID`, `trial`, `iter_ng`, and `iter_par`.
* Call the `_calculate_x_decomp_agg` method using the instantiated `Ridge` model, DataFrame `X`, Series `y`, and the parameters dictionary.
* Obtain the resulting DataFrame from the method call, which should contain decomposition results.
* Assert that the sum of the `xDecompPerc` column in the resulting DataFrame equals `0.0`, confirming the correct handling of zero coefficients.

## `test_x_decomp_agg_pos_all_false() -> None`
### USAGE
* This test verifies that the position (`pos`) values in the `x_decomp_agg` DataFrame are all `False`, given a model with negative coefficients.
* It checks if the decomposition aggregation correctly identifies negative contributions.

### IMPL
* Prepare the Test Setup:
  * Initialize a Ridge model with negative coefficients: `[-0.1, -0.2, -0.3]`.
  * Construct a DataFrame `X` with features: `feature1`, `feature2`, `feature3` each having three values.
  * Create a Series `y` with target values: `[10, 15, 20]`.

* Prepare Parameters:
  * Define a dictionary `params` containing various parameters like `train_size`, `rsq_val`, `rsq_test`, `nrmse_val`, `nrmse_test`, `nrmse`, `decomp_rssd`, `mape`, `lambda_`, `lambda_hp`, `lambda_max`, `lambda_min_ratio`, `solID`, `trial`, `iter_ng`, and `iter_par`.

* Invoke the Method:
  * Call the `_calculate_x_decomp_agg` method of `RidgeModelBuilder` using the initialized Ridge model, DataFrame `X`, Series `y`, and the `params` dictionary.
  * Capture the returned DataFrame `x_decomp_agg`.

* Perform the Assertion:
  * Assert that the 'pos' column in the `x_decomp_agg` DataFrame has all values as `False`.
  * This assertion ensures that with negative coefficients, no positive effect is incorrectly reported.

## `test_train_size_calculation() -> None`
### USAGE
* This test checks if the `train_size` value in the resulting `x_decomp_agg` DataFrame is correctly calculated and matches the expected value.
### IMPL
* Begin by setting up the necessary input data for the function.
* Use a Ridge regression model with specified coefficients.
* Create a DataFrame `X` with features and a Series `y` with target values.
* Define a dictionary `params` with various parameter values including `train_size`.
* Call the `_calculate_x_decomp_agg` method with the defined inputs.
* Validate that the `train_size` value in the `x_decomp_agg` DataFrame matches the expected value of 0.5.
* Use an assertion to ensure that the retrieved train size is as expected.

## `test_lambda_value() -> None`
### USAGE
* This test checks if the `lambda` value in the resulting `x_decomp_agg` DataFrame matches the expected value.
### IMPL
* Set up the necessary input data including a Ridge model, feature DataFrame `X`, and target Series `y`.
* Define the parameters dictionary with the expected `lambda` value.
* Execute the `_calculate_x_decomp_agg` function with the specified inputs.
* Assert that the `lambda` value in the resulting DataFrame is equal to 0.05.
* Ensure the assertion confirms the lambda value is correctly computed.

## `test_lambda_hp_value() -> None`
### USAGE
* This test verifies the `lambda_hp` value in the `x_decomp_agg` DataFrame against the expected value.
### IMPL
* Prepare the input Ridge model, DataFrame `X`, and Series `y`.
* Include `lambda_hp` in the parameters dictionary.
* Call `_calculate_x_decomp_agg` using the prepared inputs.
* Check that the `lambda_hp` value in the returned DataFrame is 0.1.
* Perform an assertion to ensure the lambda_hp value is accurately reflected.

## `test_lambda_max_value() -> None`
### USAGE
* This test ensures the `lambda_max` value in the `x_decomp_agg` DataFrame is as expected.
### IMPL
* Initialize the input Ridge model, DataFrame `X`, and Series `y`.
* Enter the expected `lambda_max` in the parameters.
* Invoke `_calculate_x_decomp_agg` with these inputs.
* Confirm the `lambda_max` value in the DataFrame equals 0.2.
* Use an assertion to verify the lambda_max value is correctly computed.

## `test_lambda_min_ratio_value() -> None`
### USAGE
* This test confirms that the `lambda_min_ratio` value in the `x_decomp_agg` DataFrame is accurate.
### IMPL
* Set up the Ridge model, DataFrame `X`, and Series `y` for testing.
* Include `lambda_min_ratio` in the parameters dictionary.
* Execute the `_calculate_x_decomp_agg` method with the input data.
* Validate that the `lambda_min_ratio` in the resulting DataFrame is 0.0005.
* Conduct an assertion to ensure the lambda_min_ratio is correctly calculated.

## `test_x_decomp_agg_empty(model: Ridge, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> None`
### USAGE
* This function tests the `_calculate_x_decomp_agg` method for the edge case where the input DataFrame `X` is empty.
* Parameters:
  - `model`: An instance of `Ridge` with coefficients, initialized with an empty coefficient set.
  - `X`: An empty pandas DataFrame representing feature data.
  - `y`: An empty pandas Series for the target variable.
  - `params`: An empty dictionary of parameters.

### IMPL
* Start by initializing an empty DataFrame `X` and an empty Series `y` to simulate the edge case.
* Instantiate a `Ridge` model with its coefficients set to an empty numpy array, representing the model's state.
* Call the `_calculate_x_decomp_agg` method with the empty `X`, `y`, and the empty `params` dictionary.
* Capture the returned DataFrame from the method call into a variable, say `x_decomp_agg`.
* Verify that the returned DataFrame `x_decomp_agg` is indeed empty.
* Use an assertion to confirm that `x_decomp_agg.empty` is `True`, as expected for this edge case.
* Ensure that no exceptions are thrown during the execution of the test, validating that the method handles empty inputs gracefully.

## `test_prepare_data_X_columns_are_correct() -> None`
### USAGE
* This unit test checks if the columns of the feature matrix `X` returned by the `_prepare_data` method are as expected.
* The test focuses on verifying that the features are correctly selected and formatted.
### IMPL
* Mock the `FeaturizedMMMData` dependency to return a DataFrame with columns `'dep_var'`, `'feature1'`, and `'feature2'`.
* Mock the `MMMData` dependency to return a specification where `'dep_var'` is the dependent variable and `'feature1'` is used as a paid media spend.
* Call the `_prepare_data` method of the `RidgeModelBuilder` class with a dictionary containing hyperparameters for transformations.
* Capture the feature matrix `X` from the method's return value.
* Assert that the columns of `X` are `['feature1']`, indicating that only the relevant feature columns are selected based on the model specification.

## `test_prepare_data_y_name_is_correct() -> None`
### USAGE
* This unit test verifies that the target series `y` returned by the `_prepare_data` method has the expected name.
* It ensures that the dependent variable is correctly identified and extracted.
### IMPL
* Mock the `FeaturizedMMMData` dependency to provide a DataFrame with columns including `'dep_var'`.
* Mock the `MMMData` dependency to specify `'dep_var'` as the dependent variable.
* Call the `_prepare_data` method of the `RidgeModelBuilder` class with a dictionary of hyperparameters.
* Capture the target series `y` from the method's return value.
* Assert that the name of `y` is `'dep_var'`, verifying that the correct dependent variable is extracted from the data.

## `test_date_column_min_value() -> None`
### USAGE
* This test checks that the minimum value in the 'date_col' column is successfully converted to 0 after preprocessing.
* It ensures that missing or NaT values in date columns are handled correctly and replaced with the minimum date value.
### IMPL
* *Mock Setup*: Mock the `FeaturizedMMMData` to return a DataFrame with columns 'dep_var', 'date_col', and 'feature1', mimicking the structure of the data to be processed.
* *Mock Setup*: Mock `MMMData` to return a specification where 'dep_var' is the dependent variable and 'feature1' is in the paid media spends list.
* *Initialize Test Object*: Instantiate the `RidgeModelBuilder` using the mocked `FeaturizedMMMData` and `MMMData`.
* *Prepare Data*: Call the `_prepare_data` method on the `RidgeModelBuilder` object with a parameter dictionary containing `feature1_thetas`.
* *Check Date Conversion*: Verify that the minimum value in the 'date_col' column of the returned DataFrame `X` is 0, confirming correct conversion and handling of date values.
* *Assertions*: Assert that the minimum value of `X['date_col']` is 0 to ensure proper date conversion.

## `test_dependent_variable_name() -> None`
### USAGE
* This test checks that the dependent variable in the prepared data has the correct name assigned as specified in the `MMMData` specification.
* It verifies that the renaming of 'dep_var' to the specified dependent variable name is performed correctly.
### IMPL
* *Mock Setup*: Mock the `FeaturizedMMMData` to return a DataFrame with columns 'dep_var', 'date_col', and 'feature1', representing the preprocessed data structure.
* *Mock Setup*: Mock `MMMData` to provide a specification where 'dep_var' is the dependent variable and 'feature1' is included in the paid media spends list.
* *Initialize Test Object*: Create an instance of the `RidgeModelBuilder` using the mocked `FeaturizedMMMData` and `MMMData`.
* *Prepare Data*: Invoke the `_prepare_data` method on the `RidgeModelBuilder` object with a dictionary containing `feature1_thetas`.
* *Verify Dependent Variable Name*: Check that the name of the returned Series `y` matches the expected dependent variable name as specified in `MMMData`.
* *Assertions*: Assert that `y.name` is equal to 'dep_var', ensuring the dependent variable is correctly named after preparation.

## `test_prepare_data_feature_columns() -> None`
### USAGE
* This unit test is designed to verify the correct preparation of feature columns in the `_prepare_data` method when handling categorical variables.
* It ensures that the resulting DataFrame `X` has the expected columns after transformation and one-hot encoding.
### IMPL
* Mock the `FeaturizedMMMData` dependency to simulate the return of a DataFrame with columns 'dep_var', 'category_col', and 'feature1'.
* Mock the `MMMData` dependency to provide the specification, particularly focusing on 'dep_var' and 'paid_media_spends' with 'feature1'.
* Call the `_prepare_data` method on the `RidgeModelBuilder` instance, passing the mocked data and parameters.
* Capture the output `X` DataFrame from the method.
* Assert that the columns of `X` match the expected list of columns, `['feature1', 'category_col_value']`, representing correct one-hot encoding and transformation.

## `test_prepare_data_dependent_variable_column() -> None`
### USAGE
* This unit test checks the proper preparation of the dependent variable in the `_prepare_data` method.
* It ensures that the resulting Series `y` is correctly named after the transformation process.
### IMPL
* Mock the `FeaturizedMMMData` dependency to simulate the return of a DataFrame with columns including 'dep_var'.
* Mock the `MMMData` dependency to provide a specification where the dependent variable is specified as 'dep_var'.
* Call the `_prepare_data` method on the `RidgeModelBuilder` instance, using the mocked data.
* Capture the output `y` Series from the method.
* Assert that the name of the `y` Series is exactly 'dep_var', confirming the dependent variable is correctly identified and processed.

## `test_prepare_data_y_isnull_sum(params: Dict[str, float]) -> None`
### USAGE
* This method tests the `_prepare_data` function to ensure that the resultant `y` Series has no missing (null) values after data preparation.
* The method takes a dictionary `params` containing the hyperparameters required for data transformations.
### IMPL
* Mock the `FeaturizedMMMData` to return a `DataFrame` with columns `'dep_var'` and `'feature1'`.
* Mock the `MMMData` to return a specification indicating `'dep_var'` as the dependent variable and `'feature1'` as a paid media spend.
* Invoke the `_prepare_data` method of `RidgeModelBuilder` with the `params` dictionary.
* Extract the `y` component from the returned tuple.
* Assert that the sum of null values in `y` is zero, ensuring there are no missing values after preparation.

## `test_prepare_data_y_name(params: Dict[str, float]) -> None`
### USAGE
* This method tests the `_prepare_data` function to verify that the name of the `y` Series is correctly set to the dependent variable name specified in `MMMData`.
* The method takes a dictionary `params` containing the hyperparameters required for data transformations.
### IMPL
* Mock the `FeaturizedMMMData` to return a `DataFrame` with columns `'dep_var'` and `'feature1'`.
* Mock the `MMMData` to return a specification indicating `'dep_var'` as the dependent variable and `'feature1'` as a paid media spend.
* Invoke the `_prepare_data` method of `RidgeModelBuilder` with the `params` dictionary.
* Extract the `y` component from the returned tuple.
* Assert that the name of the `y` Series is equal to the specified dependent variable `'dep_var'`.

## `test_feature1_no_nulls_in_X(params: Dict[str, float]) -> None`
### USAGE
* Verify that after preparing data, the 'feature1' column in the dataset `X` has no null values.
* This ensures the data processing and transformations applied correctly handle null values.
### IMPL
* Initialize a mock for the `FeaturizedMMMData` dependency to simulate the return of a DataFrame with columns 'dep_var' and 'feature1'.
* Initialize a mock for the `MMMData` dependency to simulate the return of a dictionary containing 'dep_var' and 'paid_media_spends' with 'feature1'.
* Call the `_prepare_data` method on the `RidgeModelBuilder` instance with the provided `params`.
* Capture the return values `X` and `y` from the `_prepare_data` method.
* Calculate the number of null values in the column `X['feature1']`.
* Assert that the count of null values in `X['feature1']` is zero, confirming no nulls exist after data preparation.

## `test_y_name_is_dep_var(params: Dict[str, float]) -> None`
### USAGE
* Validate that the name of the target series `y` is correctly set to 'dep_var'.
* This confirms that the dependent variable is correctly identified and renamed from 'dep_var' if necessary.
### IMPL
* Initialize a mock for the `FeaturizedMMMData` dependency to simulate the return of a DataFrame with columns 'dep_var' and 'feature1'.
* Initialize a mock for the `MMMData` dependency to simulate the return of a dictionary containing 'dep_var' and 'paid_media_spends' with 'feature1'.
* Call the `_prepare_data` method on the `RidgeModelBuilder` instance with the provided `params`.
* Capture the return values `X` and `y` from the `_prepare_data` method.
* Retrieve the name attribute of the series `y`.
* Assert that the `y.name` is equal to 'dep_var', confirming that the dependent variable name is set correctly during data preparation.

## `test_geometric_adstock_first_element() -> None`
### USAGE
* This function tests the first element of the series after applying the geometric adstock transformation.
* It checks whether the first element of the resulting series remains unchanged, as expected.
### IMPL
* Import necessary modules, including `pandas` and the function under test.
* Create a pandas Series with values `[10, 20, 30, 40, 50]`.
* Define a theta value of `0.5` for the adstock transformation.
* Call the `_geometric_adstock` function with the Series and theta as arguments.
* Capture the output of the function, which should be another pandas Series.
* Assert that the first element of the output Series is equal to the expected value `10`.
* This assertion confirms that the transformation does not alter the initial value inappropriately.

## `test_geometric_adstock_second_element() -> None`
### USAGE
* This function tests the second element of the series after applying the geometric adstock transformation.
* It verifies the cumulative effect of adstock on the second element.
### IMPL
* Import necessary modules.
* Initialize a pandas Series with values `[10, 20, 30, 40, 50]`.
* Set the theta parameter to `0.5`.
* Invoke the `_geometric_adstock` function with the Series and theta.
* Store the output Series.
* Assert that the second element of the Series is `25.0`, which is `20 + (10 * 0.5)`.
* This checks if the adstock transformation is accumulating the effect correctly.

## `test_geometric_adstock_third_element() -> None`
### USAGE
* This function checks the third element of the series post adstock transformation.
* It ensures that the transformation is correctly applying the decay factor.
### IMPL
* Import necessary modules.
* Define a pandas Series with `[10, 20, 30, 40, 50]`.
* Assign a theta of `0.5`.
* Call `_geometric_adstock` with the Series and theta.
* Retrieve the output Series.
* Assert that the third element equals `42.5`, calculated as `30 + (25 * 0.5)`.
* This validates that the adstock transformation accumulates effects over time.

## `test_geometric_adstock_fourth_element() -> None`
### USAGE
* This function tests the fourth element of a series after applying the geometric adstock transformation.
* It ensures that the adstock effect is compounded over multiple elements.
### IMPL
* Import necessary modules.
* Create a Series with values `[10, 20, 30, 40, 50]`.
* Set theta to `0.5`.
* Call the `_geometric_adstock` function.
* Capture the resulting Series.
* Assert the fourth element is `61.25`, calculated as `40 + (42.5 * 0.5)`.
* This confirms the adstock effect is compounded correctly across elements.

## `test_geometric_adstock_fifth_element() -> None`
### USAGE
* This function tests the fifth element of the series following the adstock transformation.
* It checks the cumulative effect of the transformation over the entire series.
### IMPL
* Import necessary modules.
* Initialize a Series with `[10, 20, 30, 40, 50]`.
* Use a theta of `0.5`.
* Execute the `_geometric_adstock` function.
* Retrieve the transformed Series.
* Assert that the fifth element is `80.625`, which is `50 + (61.25 * 0.5)`.
* This ensures the transformation is correctly applied over the entire series.

## `test_geometric_adstock_with_theta_zero(x: pd.Series, theta: float) -> None`
### USAGE
* This test function is designed to verify the behavior of the geometric adstock transformation when the `theta` parameter is set to 0.
* Parameters:
  - `x`: A pandas Series representing the input data to be transformed.
  - `theta`: A float representing the adstock decay parameter, which in this case is 0.
* The function checks if the transformation returns the original series unchanged, which is the expected behavior when `theta` is 0.

### IMPL
* Begin by setting up the initial test data. Create a pandas Series `x` with the values `[1, 2, 3, 4, 5]`.
* Assign the value `0` to the `theta` parameter, as the test is to evaluate the function with `theta` set to 0.
* Call the `_geometric_adstock` method from the `RidgeModelBuilder` class with the series `x` and `theta` as arguments.
* Capture the output of the `_geometric_adstock` method in a variable, say `y`.
* Use an assertion to check that the output series `y` is equal to the input series `x`. This can be done using the `equals` method of pandas Series, i.e., `y.equals(x)`.
* The assertion should confirm that `y.equals(x)` evaluates to `True`, indicating that the transformation did not alter the series.

## `test_geometric_adstock_first_element() -> None`
### USAGE
* This test case is designed to verify that the first element of the transformed series matches the expected value after applying the geometric adstock transformation with theta=1.
### IMPL
* Import necessary modules, including `pandas` for data manipulation.
* Define the test function `test_geometric_adstock_first_element`.
* Create a pandas Series `x` with values `[1, 2, 3, 4, 5]`.
* Set the adstock decay parameter `theta` to 1.
* Call the `_geometric_adstock` method with parameters `x` and `theta`.
* Capture the output in a variable `y`.
* Assert that the first element of `y` (`y.iloc[0]`) equals the expected value `1`.

## `test_geometric_adstock_second_element() -> None`
### USAGE
* This test case is designed to verify that the second element of the transformed series matches the expected value after applying the geometric adstock transformation with theta=1.
### IMPL
* Import necessary modules, including `pandas` for data manipulation.
* Define the test function `test_geometric_adstock_second_element`.
* Create a pandas Series `x` with values `[1, 2, 3, 4, 5]`.
* Set the adstock decay parameter `theta` to 1.
* Call the `_geometric_adstock` method with parameters `x` and `theta`.
* Capture the output in a variable `y`.
* Assert that the second element of `y` (`y.iloc[1]`) equals the expected value `3`.

## `test_geometric_adstock_third_element() -> None`
### USAGE
* This test case is designed to verify that the third element of the transformed series matches the expected value after applying the geometric adstock transformation with theta=1.
### IMPL
* Import necessary modules, including `pandas` for data manipulation.
* Define the test function `test_geometric_adstock_third_element`.
* Create a pandas Series `x` with values `[1, 2, 3, 4, 5]`.
* Set the adstock decay parameter `theta` to 1.
* Call the `_geometric_adstock` method with parameters `x` and `theta`.
* Capture the output in a variable `y`.
* Assert that the third element of `y` (`y.iloc[2]`) equals the expected value `6`.

## `test_geometric_adstock_fourth_element() -> None`
### USAGE
* This test case is designed to verify that the fourth element of the transformed series matches the expected value after applying the geometric adstock transformation with theta=1.
### IMPL
* Import necessary modules, including `pandas` for data manipulation.
* Define the test function `test_geometric_adstock_fourth_element`.
* Create a pandas Series `x` with values `[1, 2, 3, 4, 5]`.
* Set the adstock decay parameter `theta` to 1.
* Call the `_geometric_adstock` method with parameters `x` and `theta`.
* Capture the output in a variable `y`.
* Assert that the fourth element of `y` (`y.iloc[3]`) equals the expected value `10`.

## `test_geometric_adstock_fifth_element() -> None`
### USAGE
* This test case is designed to verify that the fifth element of the transformed series matches the expected value after applying the geometric adstock transformation with theta=1.
### IMPL
* Import necessary modules, including `pandas` for data manipulation.
* Define the test function `test_geometric_adstock_fifth_element`.
* Create a pandas Series `x` with values `[1, 2, 3, 4, 5]`.
* Set the adstock decay parameter `theta` to 1.
* Call the `_geometric_adstock` method with parameters `x` and `theta`.
* Capture the output in a variable `y`.
* Assert that the fifth element of `y` (`y.iloc[4]`) equals the expected value `15`.

## `test_geometric_adstock_empty_series() -> None`
### USAGE
* This test function is designed to verify the behavior of the `geometric_adstock` method when provided with an empty Pandas Series as input.
* `x`: An empty Pandas Series.
* `theta`: A float value representing the adstock decay parameter, set to 0.5 in this test.

### IMPL
* Initialize an empty Pandas Series to simulate the scenario where no data is present.
* Set the `theta` value to 0.5, which will be used as the adstock decay parameter.
* Invoke the `geometric_adstock` method with the empty series and `theta` as inputs.
* Capture the output of the method.
* Assert that the output is an empty series by checking the `empty` attribute of the result.
* Ensure that the assertion confirms that the method correctly handles empty input by returning an empty series.

markdown
## `assert_geometric_adstock_empty_series_empty() -> None`
### USAGE
* This function asserts that the result of the `geometric_adstock` method, when applied to an empty input series, should return an empty series.
* This ensures that the method is robust to cases where the input data is absent.

### IMPL
* Call the `geometric_adstock_empty_series` test function to perform the transformation.
* Capture the result of the transformation.
* Assert that the `empty` attribute of the result is `True`, indicating that the output is indeed an empty series.
* Verify that the expected output matches the actual output, confirming the correct handling of empty series by the `geometric_adstock` method.

## `test_geometric_adstock_first_element() -> None`
### USAGE
* This test ensures that the first element of the series is unchanged after applying the geometric adstock transformation.
* The input `x` is a pandas Series with values [10, 20, 30], and `theta` is set to 1.5.
### IMPL
* Initialize the input series `x` as a pandas Series with values [10, 20, 30].
* Set the adstock decay parameter `theta` to 1.5.
* Apply the geometric adstock transformation method to the series `x` using the `theta` parameter.
* Assert that the first element of the transformed series `y.iloc[0]` is equal to 10.
* This verifies that the initial value remains unchanged as per the adstock logic.

## `test_geometric_adstock_second_element() -> None`
### USAGE
* This test verifies that the second element of the series is correctly transformed using the geometric adstock method with `theta` greater than 1.
* The input `x` is a pandas Series with values [10, 20, 30], and `theta` is 1.5.
### IMPL
* Initialize the input series `x` as a pandas Series with values [10, 20, 30].
* Set the adstock decay parameter `theta` to 1.5.
* Apply the geometric adstock transformation method to the series `x` using the `theta` parameter.
* Calculate the expected value for the second element: `x[1] + theta * x[0]`, which should be 20 + 1.5 * 10 = 35.0.
* Assert that the second element of the transformed series `y.iloc[1]` is equal to 35.0.
* This checks the accumulation effect of the adstock transformation.

## `test_geometric_adstock_third_element() -> None`
### USAGE
* This test confirms that the third element of the series is correctly transformed using the geometric adstock method with `theta` greater than 1.
* The input `x` is a pandas Series with values [10, 20, 30], and `theta` is 1.5.
### IMPL
* Initialize the input series `x` as a pandas Series with values [10, 20, 30].
* Set the adstock decay parameter `theta` to 1.5.
* Apply the geometric adstock transformation method to the series `x` using the `theta` parameter.
* Calculate the expected value for the third element: `x[2] + theta * (x[1] + theta * x[0])`, which should be 30 + 1.5 * (20 + 1.5 * 10) = 82.5.
* Assert that the third element of the transformed series `y.iloc[2]` is equal to 82.5.
* This verifies the cumulative adstock effect over multiple periods.

## `test_hill_transformation_output() -> None`
### USAGE
* This test checks the output of the `hill_transformation` function when provided with a specific input series, alpha, and gamma values.
* The input parameters for this test are:
  - `x`: A pandas Series with values `[0.1, 0.2, 0.3, 0.4, 0.5]`.
  - `alpha`: A float value `2.0`.
  - `gamma`: A float value `1.0`.
* The expected output is a pandas Series with values `[0.0099, 0.0388, 0.0936, 0.1638, 0.25]`.

### IMPL
* Instantiate the `RidgeModelBuilder` class with mock or dummy data for required dependencies, ensuring compatibility with the `hill_transformation` method.
* Create a pandas Series `x` with values `[0.1, 0.2, 0.3, 0.4, 0.5]`.
* Define the `alpha` variable with a value of `2.0`.
* Define the `gamma` variable with a value of `1.0`.
* Call the `hill_transformation` method of the `RidgeModelBuilder` instance, passing `x`, `alpha`, and `gamma` as arguments.
* Capture the output of the `hill_transformation` method.
* Create a pandas Series `expected_output` with values `[0.0099, 0.0388, 0.0936, 0.1638, 0.25]`.
* Use an assertion to compare each element of the output Series with the corresponding element in the `expected_output` Series, allowing for a small tolerance due to potential floating-point arithmetic differences.
* Confirm that the output matches the expected values within the specified tolerance.
* Ensure proper cleanup of any resources or mocks used during the test, if necessary.

## `test_hill_transformation_output() -> None`
### USAGE
* This test verifies the output of the `_hill_transformation` method when the `alpha` parameter is set to zero.
* The input parameters are a pandas Series `x`, and float values `alpha` and `gamma`.
* The expected output is a pandas Series of zeros, indicating the transformation is correctly applied with an `alpha` of zero.

### IMPL
* Import necessary libraries such as `pandas` for creating Series and `assert` for checking test conditions.
* Define a function `test_hill_transformation_output` to encapsulate the test.
* Create a pandas Series `x` with values `[0.1, 0.2, 0.3, 0.4, 0.5]` to simulate input data.
* Set `alpha` to `0.0` and `gamma` to `1.0` based on the test case input.
* Call the `_hill_transformation` method with `x`, `alpha`, and `gamma` as arguments.
* Capture the method's output in a variable, `output`.
* Define the expected output as a pandas Series with values `[0.0, 0.0, 0.0, 0.0, 0.0]`.
* Use an assertion to compare the `output` with `expectedValue` to verify correctness. Ensure they are equal using `pandas` testing utilities for Series.
* If the output matches the expected value, the test passes; otherwise, it fails, indicating an issue with `_hill_transformation` handling when `alpha` is zero.

## `test_hill_transformation_output() -> None`
### USAGE
* This test function evaluates the behavior of the `RidgeModelBuilder._hill_transformation` method when the gamma parameter is set to zero.
* The method is expected to transform the input series based on the Hill transformation formula.
* Parameters:
  - `x`: A pandas Series representing the input data to be transformed.
  - `alpha`: A float value representing the scaling parameter for the Hill transformation.
  - `gamma`: A float value representing the shape parameter for the Hill transformation.
### IMPL
* Initialize a pandas Series with the provided values: `[0.1, 0.2, 0.3, 0.4, 0.5]`.
* Set the alpha parameter to `2.0`.
* Set the gamma parameter to `0.0`.
* Call the `_hill_transformation` method of the `RidgeModelBuilder` class with the specified parameters.
* The transformation is expected to return a series where each element is transformed according to the Hill equation: `x_scaled*alpha / (x_scaled*alpha + gamma*alpha)`.
* With gamma set to zero, the denominator becomes `1`, simplifying the transformation to `1` for each element.
* Compare the output to the expected pandas Series: `[1.0, 1.0, 1.0, 1.0, 1.0]`.
* Assert that the transformed series is equal to the expected series, indicating the transformation was correctly applied.
* Use pandas' testing utilities to compare the actual and expected series for equality.

## `test_hill_transformation_output(x: pd.Series, alpha: float, gamma: float) -> None`
### USAGE
* This test function verifies the output of the `_hill_transformation` method when applied to a constant series.
* Parameters:
  - `x`: A pandas Series object representing the input data, specifically a constant series.
  - `alpha`: A float representing the Hill curve parameter for scaling.
  - `gamma`: A float representing the Hill curve parameter for shaping.
* The function checks if the output of the transformation matches the expected transformed series.

### IMPL
* Initialize the `RidgeModelBuilder` object with mock dependencies or any required initialization parameters.
* Prepare the input Series `x` as a constant series with values `[0.5, 0.5, 0.5, 0.5, 0.5]`.
* Set the transformation parameters `alpha` to `2.0` and `gamma` to `1.0`.
* Call the `_hill_transformation` method of `RidgeModelBuilder` using the prepared `x`, `alpha`, and `gamma`.
* Capture the output of the transformation.
* Prepare the expected output as a constant Series with values `[0.25, 0.25, 0.25, 0.25, 0.25]`.
* Assert that the transformed output matches the expected Series. Use pandas' built-in testing utilities such as `pd.testing.assert_series_equal` for comparing the output and expected Series with appropriate tolerance levels.
* Ensure that any logging or exceptions are appropriately handled if the transformation does not behave as expected.

## `test_hill_transformation_output() -> None`
### USAGE
* This test function verifies the output of the `_hill_transformation` method when provided with a series of negative values.
* The function checks that the output of the transformation is a series of zeros, as expected.

### IMPL
* Initialize a Pandas Series `x` containing negative values: `[-0.1, -0.2, -0.3, -0.4, -0.5]`.
* Set the `alpha` parameter to `2.0` and the `gamma` parameter to `1.0`.
* Call the `_hill_transformation` method with `x`, `alpha`, and `gamma` as arguments.
* Capture the output of the transformation.
* Define the expected output as a Pandas Series of zeros with the same length as `x`.
* Assert that the output from the transformation matches the expected output using an appropriate assertion method to compare Pandas Series, ensuring that both the values and the index are the same.

## `test_hill_transformation_output(pd.Series, float, float) -> None`
### USAGE
* This test verifies the output of the `_hill_transformation` method.
* It takes a Pandas Series `x`, and two float parameters `alpha` and `gamma` as inputs.
* The test checks if the output matches the expected transformed series.

### IMPL
* Start by importing necessary libraries such as `pandas` and any testing framework like `unittest` or `pytest`.
* Define the test function `test_hill_transformation_output`.
* Create a Pandas Series `x` with values `[0.1, 0.2, 0.3, 0.4, 0.5]`.
* Set the `alpha` parameter to `100.0`.
* Set the `gamma` parameter to `100.0`.
* Call the `_hill_transformation` method with `x`, `alpha`, and `gamma` as arguments.
* Store the result of the transformation in a variable called `result`.
* Define the expected output as a Pandas Series with values `[0.0, 0.0, 0.0, 0.0, 0.0]`.
* Use an assertion method to check if `result` equals the expected output.
* Ensure the assertion checks that both the values and the type (i.e., Pandas Series) are the same.
* If using `unittest`, include `self.assertEqual` or similar; for `pytest`, use `assert`.
* Optionally, add a message to the assertion to clarify what is being tested.

## `test_hill_transformation_output_with_negative_alpha_and_gamma() -> None`
### USAGE
* This test function verifies that the `_hill_transformation` method in the `RidgeModelBuilder` class returns the expected output when both `alpha` and `gamma` parameters are negative. This scenario is expected to produce an output series of `NaN` values.
### IMPL
* *Step 1:* Import necessary modules such as `unittest`, `pandas` as `pd`, and `numpy` as `np`.
* *Step 2:* Create a test class that inherits from `unittest.TestCase`.
* *Step 3:* Define the test function `test_hill_transformation_output_with_negative_alpha_and_gamma`.
* *Step 4:* Initialize a `pd.Series` with values `[0.1, 0.2, 0.3, 0.4, 0.5]` to represent the input data `x`.
* *Step 5:* Set `alpha` to `-1.0` and `gamma` to `-1.0` as per the test case input.
* *Step 6:* Create an instance of the `RidgeModelBuilder` class with appropriate mock arguments for required constructor parameters.
* *Step 7:* Call the `_hill_transformation` method on the `RidgeModelBuilder` instance with `x`, `alpha`, and `gamma` as arguments.
* *Step 8:* Define the expected output as a `pd.Series` with `NaN` values for each input element.
* *Step 9:* Use `pd.testing.assert_series_equal` to assert that the actual output matches the expected output series, allowing for `NaN` value comparison.
* *Step 10:* Add a docstring to the test function describing its purpose and the specific scenario being tested.
* *Step 11:* Run the test using `unittest.main()` to ensure the method behaves as expected under the specified conditions.

## `test_calculate_rssd_without_zero_penalty() -> None`
### USAGE
* This test case is designed to verify the functionality of the `_calculate_rssd` method when the `rssd_zero_penalty` is set to `False`.
* The method should correctly compute the Root Sum of Squared Differences (RSSD) without applying any zero penalty adjustment to the coefficients.
* The expected RSSD value is provided for comparison.
### IMPL
1. *Setup Inputs*:
   - Define the coefficients as a numpy array with values `[1.0, 2.0, 3.0]`.
   - Set `rssd_zero_penalty` to `False`, indicating no penalty should be applied.

2. *Invoke Method*:
   - Call the `_calculate_rssd` method of the `RidgeModelBuilder` instance with the defined coefficients and `rssd_zero_penalty` as inputs.

3. *Perform Assertion*:
   - Compare the calculated RSSD result to the expected value `3.7416573867739413` using an appropriate assertion method to verify accuracy.
   - Ensure the assertion accounts for potential floating-point arithmetic precision issues.

## `test_calculate_rssd_returns_expected_value() -> None`
### USAGE
* This unit test function is designed to verify whether the `_calculate_rssd` method returns the expected RSSD value when provided with a specified set of coefficients and the `rssd_zero_penalty` flag set to true.
* Parameters:
  - No parameters are passed directly to this test function. It prepares necessary inputs internally.
### IMPL
* Import necessary libraries for testing, such as `unittest` or `pytest`.
* Define the test function `test_calculate_rssd_returns_expected_value`.
* Instantiate an object of the `RidgeModelBuilder` class, using mock dependencies for its constructor parameters if necessary.
* Prepare the input coefficients as a NumPy array: `[0.0, 0.0, 3.0]`.
* Set the `rssd_zero_penalty` flag to `True`.
* Call the `_calculate_rssd` method of the `RidgeModelBuilder` instance with the prepared inputs.
* Store the returned RSSD value.
* Use an assertion function, such as `assertEqual` or `assertAlmostEqual`, to check if the returned RSSD value is equal to the expected value, `6.0`.
* Ensure the test function concludes without errors, confirming the method's correct behavior.

## `test_calculate_rssd_with_zero_coefficients() -> None`
### USAGE
* This unit test verifies the behavior of the `_calculate_rssd` method in the presence of zero coefficients and when the `rssd_zero_penalty` is set to `True`. The objective is to ensure the Root Sum of Squared Differences (RSSD) is calculated correctly, taking into account any penalties for zero coefficients, which should yield a specific expected value.

### IMPL
* *Step 1:* Initialize the `RidgeModelBuilder` class.
  - This involves setting up any necessary data inputs which the class constructor might require, possibly using mock objects or simple instances of the required data classes.

* *Step 2:* Define the coefficients array (`coefs`) as `[0.0, 2.0, 0.0]`.
  - This represents the coefficients of the model, including some zero values, which are important for testing the zero penalty behavior.

* *Step 3:* Set the `rssd_zero_penalty` parameter to `True`.
  - This indicates that the penalty for zero coefficients should be included in the RSSD calculation.

* *Step 4:* Call the `_calculate_rssd` method on the `RidgeModelBuilder` instance.
  - Pass the `coefs` and `rssd_zero_penalty` as arguments. This will compute the RSSD value considering the zero penalty.

* *Step 5:* Assert that the returned RSSD value equals `4.0`.
  - This step verifies that the method correctly applies the zero penalty and calculates the RSSD as expected. The expected value is given by the test case, ensuring the method's correctness.

## `test_calculate_rssd_with_zero_coefficients_and_zero_penalty() -> None`
### USAGE
* This function tests the `_calculate_rssd` method of `RidgeModelBuilder` to ensure it correctly computes the RSSD when all coefficients are zero and the `rssd_zero_penalty` flag is set to true.
* The test verifies that the RSSD value is computed as expected, which should be 0.0 in this scenario.

### IMPL
* Instantiate a `RidgeModelBuilder` object with mock dependencies, as the test only focuses on the `_calculate_rssd` method.
* Define test inputs:
  * Use an array of coefficients `[0.0, 0.0, 0.0]`.
  * Set `rssd_zero_penalty` to `True` to apply zero penalty in the calculation.
* Call the `_calculate_rssd` method with the test inputs.
* Capture the output, which is the calculated RSSD value.
* Assert that the result is equal to the expected RSSD value of `0.0`.
* Ensure that no errors are raised during the execution of the test.

## `test_calculate_rssd_value() -> None`
### USAGE
* This test verifies the calculation of the RSSD (Root Sum of Squared Differences) value for the given coefficients with the `rssd_zero_penalty` parameter set to `True`.
* The input parameters simulate a scenario where coefficients are non-zero and the zero penalty is enabled.
* The expected RSSD value is `3.7416573867739413`.
### IMPL
* *Initialize Input Coefficients*: Start by creating an array of coefficients `[1.0, 2.0, 3.0]` which will be passed to the `_calculate_rssd` method.
* *Set RSSD Zero Penalty Flag*: Define a boolean variable `rssd_zero_penalty` and set it to `True` to enable the zero penalty in the RSSD calculation.
* *Invoke Calculate RSSD Method*: Call the `_calculate_rssd` method from the `RidgeModelBuilder` class using the coefficients and the zero penalty flag as inputs.
* *Capture RSSD Result*: Store the output of the `_calculate_rssd` method in a variable `rssd_result` for further verification.
* *Perform Assertion*: Use an assertion to verify that the `rssd_result` is approximately equal to `3.7416573867739413`, the expected RSSD value, allowing for minor floating-point inaccuracies.

## `test_calculate_rssd_returns_expected_value(coefs: List[float], rssd_zero_penalty: bool) -> None`
### USAGE
* This test function verifies that the `_calculate_rssd` method correctly calculates the Root Sum of Squared Differences (RSSD) for a single coefficient without applying a zero penalty.
* Parameters:
  * `coefs`: A list of coefficients for which RSSD is to be calculated. In this test, it contains a single coefficient `[5.0]`.
  * `rssd_zero_penalty`: A boolean indicating whether a zero penalty should be applied in the RSSD calculation. In this test, it is set to `False`.

### IMPL
* Create an instance of the `RidgeModelBuilder` class.
* Define a list `coefs` with a single value `[5.0]` representing the coefficients to be used in the RSSD calculation.
* Set the `rssd_zero_penalty` flag to `False` to indicate that no zero penalty should be applied.
* Call the `_calculate_rssd` method on the `RidgeModelBuilder` instance with the `coefs` and `rssd_zero_penalty` as arguments.
* Capture the returned RSSD value.
* Assert that the returned RSSD value is equal to the expected value `5.0` to confirm that the calculation is correct.

## `test_calculate_rssd_single_zero_coefficient_with_zero_penalty() -> None`
### USAGE
* This unit test is designed to verify the `_calculate_rssd` method's behavior when provided with a single zero coefficient and a zero penalty flag set to `true`.
* Parameters:
  - The function does not take any parameters but internally uses mock input values as described in the test case.

### IMPL
* *Step 1*: Initialize the `RidgeModelBuilder` or the relevant class instance if needed.
  * This step is often necessary to access the method being tested unless the method is a static method.
* *Step 2*: Prepare test input data.
  * Create an `np.ndarray` representing the coefficients. In this case, an array with a single element `[0.0]`.
  * Set the `rssd_zero_penalty` flag to `true`.
* *Step 3*: Call the `_calculate_rssd` method with the prepared inputs.
  * This method should process the coefficients and the penalty flag to calculate the RSSD.
* *Step 4*: Assert the expected result.
  * The expected RSSD value is `0.0`, given the input conditions.
  * Use an assertion to compare the method's output with the expected value, ensuring they match.
* *Additional Considerations*:
  * Ensure that no external dependencies or states are unintentionally affected by the test.
  * The test should run independently and consistently yield the same results on repeated executions.

## `test_calculate_rssd_with_negative_coefficients_no_zero_penalty() -> None`
### USAGE
* This test function checks the RSSD calculation with negative coefficients and without applying a zero penalty.
* It ensures that the RSSD calculation handles negative coefficients correctly and that the zero penalty is not applied.

### IMPL
* Import the necessary modules for testing, including the `numpy` library for handling arrays.
* Create a test function named `test_calculate_rssd_with_negative_coefficients_no_zero_penalty`.
* Define the input coefficients as a numpy array with values `[-1.0, -2.0, -3.0]`.
* Set the `rssd_zero_penalty` flag to `False` to indicate that no zero penalty should be applied during the calculation.
* Call the `_calculate_rssd` method from the `RidgeModelBuilder` class, passing in the coefficients and the `rssd_zero_penalty` flag.
* Store the result of the RSSD calculation.
* Define the expected value for the RSSD result as `3.7416573867739413`.
* Use an assertion to compare the calculated RSSD result with the expected value, ensuring they match within a reasonable tolerance for floating-point comparisons.
* If the assertion passes, the test confirms that the RSSD calculation behaves correctly for negative coefficients without zero penalty.
* Include documentation strings to explain each step of the test process.

## `test_calculate_rssd_with_positive_and_negative_coefficients(coefs: List[float], rssd_zero_penalty: bool) -> None`
### USAGE
* This test checks the calculation of the Root Sum of Squared Differences (RSSD) for a mix of negative and positive coefficients without applying a zero penalty.
* Parameters:
  - `coefs`: A list of coefficients that include both negative and positive values.
  - `rssd_zero_penalty`: Boolean flag indicating whether to apply a zero penalty, set to `False` in this test case.

### IMPL
* Instantiate the `RidgeModelBuilder` class, ensuring all necessary dependencies are initialized properly. Note: The constructor parameters like `mmm_data`, `holiday_data`, etc., are assumed to be correctly instantiated objects relevant to the overall system.
* Call the `_calculate_rssd` method on the `RidgeModelBuilder` instance, passing in the `coefs` and `rssd_zero_penalty`.
* Capture the output, which is the RSSD value calculated by the method.
* Assert that the calculated RSSD matches the expected value of `3.7416573867739413`. This ensures the method correctly computes the RSSD for given coefficients without any zero penalty applied.

## `test_calculate_mape_returns_zero_mape_on_valid_data() -> None`
### USAGE
* This method tests the `_calculate_mape` function of the `RidgeModelBuilder` class to ensure that it returns a Mean Absolute Percentage Error (MAPE) of 0.0 when the model predictions perfectly match the actual lift in the calibration data.
* Assumes the input model is a Ridge instance, and calibration data and predictions are set up as per the mock data.

### IMPL
* Mock the `MMMData` dependency to return a DataFrame with `date` and `dependent_variable` columns for a specific range.
* Mock the `FeaturizedMMMData` dependency to always return 0 for `rollingWindowStartWhich` and 2 for `rollingWindowEndWhich`.
* Mock the `CalibrationInput` dependency to return a dictionary with specific `liftStartDate`, `liftEndDate`, and `liftMedia` values when accessed with a key.
* Mock the Ridge model's `predict` method to return a list of predictions that match the actual dependent variable values for the specified dates.
* Create an instance of `RidgeModelBuilder`, passing mocked dependencies into the constructor.
* Call the `_calculate_mape` method with the Ridge model instance.
* Assert that the returned MAPE value is 0.0, indicating perfect prediction accuracy.

## `test_calculate_mape_mape_value() -> None`
### USAGE
* This unit test function verifies the Mean Absolute Percentage Error (MAPE) value calculation of the `_calculate_mape` method within the `RidgeModelBuilder` class.
* It checks that the calculated MAPE matches the expected value, ensuring the accuracy of model evaluation against calibration data.
### IMPL
* Initialize a mock instance of the `MMMData` class to return a predefined dataset when its `data` method is called.
* Set up a mock instance of the `FeaturizedMMMData` class to return specific start and end indices for rolling windows.
* Create a mock instance of the `CalibrationInput` class to provide calibration data when accessed with a specific key.
* Mock the `Ridge` model's `predict` method to return predefined predictions for the provided transformed feature data.
* Instantiate the `RidgeModelBuilder` class using the mocked dependencies (`MMMData`, `FeaturizedMMMData`, `CalibrationInput`, and the `Ridge` instance).
* Call the `_calculate_mape` method on the `RidgeModelBuilder` instance, passing the mocked `Ridge` model as an argument.
* Assert that the returned MAPE value is approximately equal to the expected MAPE value of 6.6667, allowing for minor floating-point differences.

## `test_calculate_mape_no_calibration_data(model: Ridge) -> None`
### USAGE
* This unit test is designed to verify the behavior of the `_calculate_mape` method in the `RidgeModelBuilder` class when no calibration data is provided.
* It checks that the method correctly returns a MAPE value of `0.0` under these conditions, as expected.

### IMPL
* Instantiate a `Ridge` model object which will be passed as an argument to the `_calculate_mape` method.
* Mock the `RidgeModelBuilder` object to simulate the scenario where no calibration data is present.
* Ensure that the `calibration_input` attribute is set to `None` to indicate the absence of calibration data.
* Invoke the `_calculate_mape` method on the mocked `RidgeModelBuilder` instance, passing the `Ridge` model as an argument.
* Capture the output of the `_calculate_mape` method, which represents the calculated MAPE value.
* Assert that the returned MAPE value is equal to `0.0`.
* This test verifies that the method handles the lack of calibration data by returning a MAPE of `0.0`, as per the specified behavior.

## `test_evaluate_model_loss() -> None`
### USAGE
* This test ensures that the 'loss' output from the `_evaluate_model` method returns a float value.
* Asserts that the 'loss' in the returned dictionary is a float type.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'loss' key in the returned dictionary is of type float.

## `test_evaluate_model_nrmse() -> None`
### USAGE
* This test ensures that the 'nrmse' output from the `_evaluate_model` method returns a float value.
* Asserts that the 'nrmse' in the returned dictionary is a float type.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'nrmse' key in the returned dictionary is of type float.

## `test_evaluate_model_decomp_rssd() -> None`
### USAGE
* This test ensures that the 'decomp_rssd' output from the `_evaluate_model` method is equal to 0.05.
* Asserts that the 'decomp_rssd' in the returned dictionary matches the expected value of 0.05.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'decomp_rssd' key in the returned dictionary is equal to 0.05.

## `test_evaluate_model_mape() -> None`
### USAGE
* This test ensures that the 'mape' output from the `_evaluate_model` method is equal to 0.07.
* Asserts that the 'mape' in the returned dictionary matches the expected value of 0.07.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'mape' key in the returned dictionary is equal to 0.07.

## `test_evaluate_model_lift_calibration() -> None`
### USAGE
* This test ensures that the 'lift_calibration' output from the `_evaluate_model` method is equal to 0.1.
* Asserts that the 'lift_calibration' in the returned dictionary matches the expected value of 0.1.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'lift_calibration' key in the returned dictionary is equal to 0.1.

## `test_evaluate_model_rsq_train() -> None`
### USAGE
* This test ensures that the 'rsq_train' output from the `_evaluate_model` method returns a float value.
* Asserts that the 'rsq_train' in the returned dictionary is a float type.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'rsq_train' key in the returned dictionary is of type float.

## `test_evaluate_model_rsq_val() -> None`
### USAGE
* This test ensures that the 'rsq_val' output from the `_evaluate_model` method returns a float value.
* Asserts that the 'rsq_val' in the returned dictionary is a float type.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'rsq_val' key in the returned dictionary is of type float.

## `test_evaluate_model_rsq_test() -> None`
### USAGE
* This test ensures that the 'rsq_test' output from the `_evaluate_model` method returns a float value.
* Asserts that the 'rsq_test' in the returned dictionary is a float type.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'rsq_test' key in the returned dictionary is of type float.

## `test_evaluate_model_lambda_() -> None`
### USAGE
* This test ensures that the 'lambda_' output from the `_evaluate_model` method is equal to 0.5.
* Asserts that the 'lambda_' in the returned dictionary matches the expected value of 0.5.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'lambda_' key in the returned dictionary is equal to 0.5.

## `test_evaluate_model_pos() -> None`
### USAGE
* This test ensures that the 'pos' output from the `_evaluate_model` method returns an integer value.
* Asserts that the 'pos' in the returned dictionary is an integer type.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'pos' key in the returned dictionary is of type int.

## `test_evaluate_model_elapsed() -> None`
### USAGE
* This test ensures that the 'elapsed' output from the `_evaluate_model` method returns a float value.
* Asserts that the 'elapsed' in the returned dictionary is a float type.

### IMPL
* Mock the `_prepare_data` method of the `RidgeModelBuilder` class to return a DataFrame of features and a Series of the target.
* Mock the `_calculate_rssd` method to return 0.05.
* Mock the `_calculate_mape` method to return 0.07.
* Mock the `_calculate_lift_calibration` method to return 0.1.
* Set up the input parameters for the `_evaluate_model` method call, including `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, and `trial`.
* Call the `_evaluate_model` method with the mocked data and input parameters.
* Assert that the 'elapsed' key in the returned dictionary is of type float.

## `test_loss_calculation(input: Dict[str, Any]) -> None`
### USAGE
* This test verifies the computation of the `loss` value returned by the `_evaluate_model` function.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Mock the `_prepare_data` method of `RidgeModelBuilder` to return a prepared DataFrame of features and a Series of targets when invoked with the given `lambda` parameter.
* Mock the `_calculate_rssd` method to return a pre-defined RSSD value when provided with model coefficients and `rssd_zero_penalty` flag.
* Mock the `_calculate_mape` method to return a fixed MAPE value for the Ridge model instance.
* Invoke the `_evaluate_model` method with the mocked dependencies and the specified input parameters.
* Extract the `loss` value from the returned result dictionary.
* Assert that the `loss` value is of type `float`.

## `test_nrmse_calculation(input: Dict[str, Any]) -> None`
### USAGE
* This test ensures that the `nrmse` value is correctly calculated and returned by the `_evaluate_model` function.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Set up a mock for the `_prepare_data` method to provide a correct DataFrame and Series with the given parameters.
* Ensure `_calculate_rssd` and `_calculate_mape` are mocked to return fixed values.
* Call `_evaluate_model` with the input parameters including `ts_validation` and other relevant flags.
* Capture the `nrmse` value from the function's output.
* Verify that the `nrmse` value is of type `float`.

## `test_decomp_rssd_value(input: Dict[str, Any]) -> None`
### USAGE
* This test checks that the `decomp_rssd` value in the `_evaluate_model` output matches the expected value.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Mock `_prepare_data` to simulate data preparation output.
* Ensure `_calculate_rssd` returns the expected RSSD value.
* Invoke `_evaluate_model` with the test input.
* Assert that the `decomp_rssd` value in the result is equal to `0.02`.

## `test_mape_value(input: Dict[str, Any]) -> None`
### USAGE
* This test verifies that the `mape` value returned by `_evaluate_model` is correctly calculated.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Mock `_prepare_data` to return prepared data.
* Ensure `_calculate_mape` returns the expected MAPE value.
* Call `_evaluate_model` with the test input.
* Check that the `mape` value in the result equals `0.05`.

## `test_lift_calibration(input: Dict[str, Any]) -> None`
### USAGE
* This test ensures that the `lift_calibration` output from `_evaluate_model` is handled as expected.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Prepare data mocks as required.
* Mock `_calculate_mape` and other dependencies as needed.
* Invoke `_evaluate_model` with relevant parameters.
* Assert that the `lift_calibration` result is `None`.

## `test_rsq_train_value(input: Dict[str, Any]) -> None`
### USAGE
* This test checks that the `rsq_train` value returned by `_evaluate_model` is a valid float.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Mock necessary data preparation and method dependencies.
* Call `_evaluate_model` with the test input.
* Assert that `rsq_train` in the result is a float.

## `test_rsq_val_value(input: Dict[str, Any]) -> None`
### USAGE
* This test ensures the `rsq_val` value returned by `_evaluate_model` is zero when time-series validation is disabled.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Set up mocks for `_prepare_data` and other methods.
* Call `_evaluate_model` with `ts_validation` set to `False`.
* Verify that `rsq_val` is `0.0`.

## `test_rsq_test_value(input: Dict[str, Any]) -> None`
### USAGE
* This test verifies that the `rsq_test` value from `_evaluate_model` is zero when time-series validation is off.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Mock necessary methods for data preparation and calculation.
* Invoke `_evaluate_model` with `ts_validation` set to `False`.
* Assert that `rsq_test` is `0.0`.

## `test_lambda_value(input: Dict[str, Any]) -> None`
### USAGE
* This test checks that the `lambda_` value in the output of `_evaluate_model` matches the input `lambda`.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Ensure data preparation and relevant calculations are mocked.
* Call `_evaluate_model` with the specified input.
* Assert that the `lambda_` result is `2.0`.

## `test_pos_value(input: Dict[str, Any]) -> None`
### USAGE
* This test verifies that the `pos` value in the `_evaluate_model` output is an integer.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Mock data preparation and necessary calculations.
* Execute `_evaluate_model` with the test input.
* Confirm that `pos` is an integer.

## `test_elapsed_time(input: Dict[str, Any]) -> None`
### USAGE
* This test ensures that the `elapsed` time value returned by `_evaluate_model` is a float.
* The `input` parameter is a dictionary containing model parameters and configurations needed for the test.
### IMPL
* Prepare mocks for data and method dependencies.
* Call `_evaluate_model` with the specified input.
* Verify that `elapsed` is a float.

## `test_loss_calculation() -> None`
### USAGE
* Validate if the loss computation in the model evaluation process yields a float value.
### IMPL
* Mock the `_prepare_data` method in `RidgeModelBuilder` to return a DataFrame of features and a Series of target.
* Mock the `_calculate_rssd` method in `RidgeModelBuilder` to return a fixed value of 0.01.
* Mock the `_lambda_seq` method in `RidgeModelBuilder` to return a sequence of lambda values.
* Create an instance of `RidgeModelBuilder` and call the `evaluate_model` method with minimal parameters and default values.
* Assert that the returned loss value is of type float.

## `test_nrmse_calculation() -> None`
### USAGE
* Verify that the Normalized Root Mean Square Error (NRMSE) is calculated as a float value.
### IMPL
* Mock the `_prepare_data` method to output a DataFrame of features and a Series of target.
* Mock the `_calculate_rssd` method to return 0.01.
* Create an instance of `RidgeModelBuilder` and invoke the `evaluate_model` function.
* Assert that the NRMSE in the result is of type float.

## `test_decomp_rssd_value() -> None`
### USAGE
* Confirm that the decomposition RSSD value is correctly returned as 0.01.
### IMPL
* Mock the `_prepare_data` and `_calculate_rssd` methods to return expected values.
* Create an instance of `RidgeModelBuilder` and call the `evaluate_model` method.
* Assert that the `decomp_rssd` in the result matches the expected value of 0.01.

## `test_mape_calculation() -> None`
### USAGE
* Check if the Mean Absolute Percentage Error (MAPE) is calculated as 0.0.
### IMPL
* Mock necessary methods such as `_prepare_data` and `_calculate_rssd`.
* Create an instance of `RidgeModelBuilder` and call `evaluate_model`.
* Assert that the `mape` value in the result is 0.0.

## `test_lift_calibration_value() -> None`
### USAGE
* Ensure that the lift calibration is returned as `None`.
### IMPL
* Mock methods and simulate model evaluation.
* Create an instance of `RidgeModelBuilder` and evaluate the model.
* Assert the `lift_calibration` value is `None`.

## `test_rsq_train_value() -> None`
### USAGE
* Verify that the R-squared value for the training set is a float.
### IMPL
* Mock methods such as `_prepare_data` and `_calculate_rssd`.
* Call `evaluate_model` on a `RidgeModelBuilder` instance.
* Assert the `rsq_train` value in the result is a float.

## `test_rsq_val_value() -> None`
### USAGE
* Validate that the R-squared value for validation is a float.
### IMPL
* Mock methods and configure the model evaluation setup.
* Call `evaluate_model` on a `RidgeModelBuilder` instance.
* Assert the `rsq_val` value in the result is a float.

## `test_rsq_test_value() -> None`
### USAGE
* Check that the R-squared value for the test set is a float.
### IMPL
* Set up mocks and run the model evaluation.
* Invoke `evaluate_model` on a `RidgeModelBuilder` instance.
* Assert the `rsq_test` value in the result is a float.

## `test_lambda_value() -> None`
### USAGE
* Confirm that the lambda value is returned as 1.0.
### IMPL
* Mock methods and simulate model evaluation.
* Create an instance of `RidgeModelBuilder` and run `evaluate_model`.
* Assert that the `lambda_` in the result is 1.0.

## `test_pos_value() -> None`
### USAGE
* Ensure that the position value is an integer.
### IMPL
* Mock methods, especially those affecting position calculation.
* Evaluate the model using `RidgeModelBuilder`.
* Assert that the `pos` value in the result is an integer.

## `test_elapsed_time_value() -> None`
### USAGE
* Verify that the elapsed time is represented as a float.
### IMPL
* Mock the necessary components and run the model evaluation.
* Invoke `evaluate_model` and check the elapsed time.
* Assert that the `elapsed` value is a float.

## `test_hyper_list_all(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Dict[str, Any], cores: int) -> None`
### USAGE
* To verify that the `hyper_list_all` from the `_hyper_collector` function contains the expected hyperparameters.
* Parameters:
  * `hyperparameters_dict`: A dictionary containing prepared hyperparameters and hyperparameters to optimize.
  * `ts_validation`: A boolean flag indicating whether time-series validation is enabled.
  * `add_penalty_factor`: A boolean flag indicating whether penalty factors should be added.
  * `dt_hyper_fixed`: Dictionary containing fixed hyperparameters.
  * `cores`: Number of CPU cores to use for parallel processing.
### IMPL
* Call the `_hyper_collector` function with the provided input parameters.
* Extract the `hyper_list_all` from the result of `_hyper_collector`.
* Compare `hyper_list_all` to the expected dictionary of hyperparameters and assert equality.

## `test_hyper_bound_list_updated(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Dict[str, Any], cores: int) -> None`
### USAGE
* To verify that the `hyper_bound_list_updated` from `_hyper_collector` function is correctly initialized as an empty list.
* Parameters:
  * `hyperparameters_dict`: A dictionary containing prepared hyperparameters and hyperparameters to optimize.
  * `ts_validation`: A boolean flag indicating whether time-series validation is enabled.
  * `add_penalty_factor`: A boolean flag indicating whether penalty factors should be added.
  * `dt_hyper_fixed`: Dictionary containing fixed hyperparameters.
  * `cores`: Number of CPU cores to use for parallel processing.
### IMPL
* Call the `_hyper_collector` function with the provided input parameters.
* Extract the `hyper_bound_list_updated` from the result of `_hyper_collector`.
* Assert that `hyper_bound_list_updated` is an empty list.

## `test_hyper_bound_list_fixed(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Dict[str, Any], cores: int) -> None`
### USAGE
* To verify that the `hyper_bound_list_fixed` from the `_hyper_collector` function contains the expected fixed hyperparameters.
* Parameters:
  * `hyperparameters_dict`: A dictionary containing prepared hyperparameters and hyperparameters to optimize.
  * `ts_validation`: A boolean flag indicating whether time-series validation is enabled.
  * `add_penalty_factor`: A boolean flag indicating whether penalty factors should be added.
  * `dt_hyper_fixed`: Dictionary containing fixed hyperparameters.
  * `cores`: Number of CPU cores to use for parallel processing.
### IMPL
* Call the `_hyper_collector` function with the provided input parameters.
* Extract the `hyper_bound_list_fixed` from the result of `_hyper_collector`.
* Compare `hyper_bound_list_fixed` to the expected dictionary of fixed hyperparameters and assert equality.

## `test_dt_hyper_fixed_mod(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Dict[str, Any], cores: int) -> None`
### USAGE
* To verify that the `dt_hyper_fixed_mod` from the `_hyper_collector` function is correctly set when `dt_hyper_fixed` is provided.
* Parameters:
  * `hyperparameters_dict`: A dictionary containing prepared hyperparameters and hyperparameters to optimize.
  * `ts_validation`: A boolean flag indicating whether time-series validation is enabled.
  * `add_penalty_factor`: A boolean flag indicating whether penalty factors should be added.
  * `dt_hyper_fixed`: Dictionary containing fixed hyperparameters.
  * `cores`: Number of CPU cores to use for parallel processing.
### IMPL
* Call the `_hyper_collector` function with the provided input parameters.
* Extract the `dt_hyper_fixed_mod` from the result of `_hyper_collector`.
* Compare `dt_hyper_fixed_mod` to the expected fixed parameters and assert equality.

## `test_all_fixed(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Dict[str, Any], cores: int) -> None`
### USAGE
* To verify that the `all_fixed` flag in the result of `_hyper_collector` function is set to `True` when all parameters are fixed.
* Parameters:
  * `hyperparameters_dict`: A dictionary containing prepared hyperparameters and hyperparameters to optimize.
  * `ts_validation`: A boolean flag indicating whether time-series validation is enabled.
  * `add_penalty_factor`: A boolean flag indicating whether penalty factors should be added.
  * `dt_hyper_fixed`: Dictionary containing fixed hyperparameters.
  * `cores`: Number of CPU cores to use for parallel processing.
### IMPL
* Call the `_hyper_collector` function with the provided input parameters.
* Extract the `all_fixed` flag from the result of `_hyper_collector`.
* Assert that `all_fixed` is `True`.

## `test_hyper_list_all(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> None`
### USAGE
* This function tests if the `hyper_list_all` result obtained from the `_hyper_collector` method matches the expected hyperparameters dictionary.
* Parameters:
  - `hyperparameters_dict`: Should contain the dictionary with prepared hyperparameters.
  - `ts_validation`: Flag indicating if time-series validation is used.
  - `add_penalty_factor`: Boolean indicating if penalty factors should be added.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `cores`: Number of CPU cores for parallel processing.
### IMPL
* Mock the logger used within the `_hyper_collector` function to suppress or verify logs.
* Invoke `_hyper_collector` with the input parameters.
* Capture the output of `_hyper_collector`.
* Assert that the `hyper_list_all` within the output matches the expected dictionary:
  * The expected dictionary should have `channel1` with its parameters set as specified in the test case.
* Ensure the test passes if the actual and expected dictionaries are equivalent.

## `test_hyper_bound_list_updated(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> None`
### USAGE
* This function tests if the `hyper_bound_list_updated` result from the `_hyper_collector` method is as expected.
* Parameters:
  - `hyperparameters_dict`: Should contain the dictionary with prepared hyperparameters.
  - `ts_validation`: Flag indicating if time-series validation is used.
  - `add_penalty_factor`: Boolean indicating if penalty factors should be added.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `cores`: Number of CPU cores for parallel processing.
### IMPL
* Mock the logger used in the `_hyper_collector` to control or validate log messages.
* Call `_hyper_collector` with the given parameters.
* Retrieve the `hyper_bound_list_updated` from the collector output.
* Verify that this list contains the specific hyperparameters that need optimization:
  * The list should match the expected list: `["channel1_shapes", "channel1_alphas", "channel1_penalty"]`.
* Use assertions to compare the actual and expected lists, ensuring they match perfectly.

## `test_hyper_bound_list_fixed(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> None`
### USAGE
* This function verifies that the `hyper_bound_list_fixed` contains the correct fixed hyperparameters.
* Parameters:
  - `hyperparameters_dict`: Should contain the dictionary with prepared hyperparameters.
  - `ts_validation`: Flag indicating if time-series validation is used.
  - `add_penalty_factor`: Boolean indicating if penalty factors should be added.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `cores`: Number of CPU cores for parallel processing.
### IMPL
* Mock the logger to track or suppress log activity during the function call.
* Execute `_hyper_collector` with the specified inputs.
* Extract `hyper_bound_list_fixed` from the output.
* The expected fixed list should have values for attributes like `thetas`, `scales`, `gammas`, `lambda`, and `train_size`.
* Confirm that these are set as expected using assertions, comparing actual output to expected fixed values.

## `test_dt_hyper_fixed_mod(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> None`
### USAGE
* This function ensures that `dt_hyper_fixed_mod` is correctly set in the output of `_hyper_collector`.
* Parameters:
  - `hyperparameters_dict`: Should contain the dictionary with prepared hyperparameters.
  - `ts_validation`: Flag indicating if time-series validation is used.
  - `add_penalty_factor`: Boolean indicating if penalty factors should be added.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `cores`: Number of CPU cores for parallel processing.
### IMPL
* Mock logging to manage or verify logging actions during method execution.
* Call `_hyper_collector` with the provided parameters.
* Verify the `dt_hyper_fixed_mod` in the output:
  * It should be an empty DataFrame if `dt_hyper_fixed` is not set.
  * Use assertions to confirm that this DataFrame is indeed empty as expected.

## `test_all_fixed(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> None`
### USAGE
* This function checks the `all_fixed` flag in the result from `_hyper_collector`.
* Parameters:
  - `hyperparameters_dict`: Should contain the dictionary with prepared hyperparameters.
  - `ts_validation`: Flag indicating if time-series validation is used.
  - `add_penalty_factor`: Boolean indicating if penalty factors should be added.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `cores`: Number of CPU cores for parallel processing.
### IMPL
* Use mock logging to capture or suppress logs generated during the function call.
* Execute `_hyper_collector` using the input parameters.
* Check the `all_fixed` flag within the output:
  * It should be `False` as per the test case.
* Implement an assertion to confirm the flag's value matches the expected boolean value.

## `test_hyper_list_all() -> None`
### USAGE
* Tests that the hyperparameter list (`hyper_list_all`) is correctly collected and returned.
* Validates the structure and content of the hyperparameters dictionary.
### IMPL
* Initialize the input dictionary with `hyperparameters_dict` containing `prepared_hyperparameters` and `hyper_to_optimize`.
* Call the `_hyper_collector` method with `hyperparameters_dict` and other required parameters (`ts_validation`, `add_penalty_factor`, `dt_hyper_fixed`, `cores`).
* Capture the returned dictionary from `_hyper_collector`.
* Assert that the `hyper_list_all` key in the returned dictionary matches the expected hyperparameters dictionary provided in the test input.

## `test_hyper_bound_list_updated() -> None`
### USAGE
* Tests that the list of hyperparameters to be optimized (`hyper_bound_list_updated`) is correctly identified and returned.
* Validates that the hyperparameters marked for optimization are collected accurately.
### IMPL
* Initialize the input dictionary with `hyperparameters_dict` containing `prepared_hyperparameters` and `hyper_to_optimize`.
* Call the `_hyper_collector` method with `hyperparameters_dict` and other required parameters (`ts_validation`, `add_penalty_factor`, `dt_hyper_fixed`, `cores`).
* Capture the returned dictionary from `_hyper_collector`.
* Assert that the `hyper_bound_list_updated` key in the returned dictionary matches the expected list of hyperparameters to optimize, as provided in the test input.

## `test_hyper_bound_list_fixed() -> None`
### USAGE
* Tests that the fixed hyperparameters (`hyper_bound_list_fixed`) are correctly identified and returned.
* Ensures that hyperparameters not marked for optimization are correctly labeled as fixed.
### IMPL
* Initialize the input dictionary with `hyperparameters_dict` containing `prepared_hyperparameters` and `hyper_to_optimize`.
* Call the `_hyper_collector` method with `hyperparameters_dict` and other required parameters (`ts_validation`, `add_penalty_factor`, `dt_hyper_fixed`, `cores`).
* Capture the returned dictionary from `_hyper_collector`.
* Assert that the `hyper_bound_list_fixed` key in the returned dictionary matches the expected fixed hyperparameters dictionary provided in the test input.

## `test_dt_hyper_fixed_mod() -> None`
### USAGE
* Tests that the modified fixed hyperparameters DataFrame (`dt_hyper_fixed_mod`) is correctly processed and returned.
* Checks the condition when `dt_hyper_fixed` is None.
### IMPL
* Initialize the input dictionary with `hyperparameters_dict` containing `prepared_hyperparameters` and `hyper_to_optimize`.
* Call the `_hyper_collector` method with `hyperparameters_dict` and other required parameters (`ts_validation`, `add_penalty_factor`, `dt_hyper_fixed`, `cores`).
* Capture the returned dictionary from `_hyper_collector`.
* Assert that the `dt_hyper_fixed_mod` key in the returned dictionary matches the expected empty DataFrame since `dt_hyper_fixed` is None in the test input.

## `test_all_fixed() -> None`
### USAGE
* Tests that the `all_fixed` flag is correctly set when no hyperparameters are to be optimized.
* Validates the logic that determines if all hyperparameters are fixed.
### IMPL
* Initialize the input dictionary with `hyperparameters_dict` containing `prepared_hyperparameters` and `hyper_to_optimize`.
* Call the `_hyper_collector` method with `hyperparameters_dict` and other required parameters (`ts_validation`, `add_penalty_factor`, `dt_hyper_fixed`, `cores`).
* Capture the returned dictionary from `_hyper_collector`.
* Assert that the `all_fixed` key in the returned dictionary is set to `False`, indicating not all hyperparameters are fixed as per the test input.

## `test_hyper_list_all(input_data: Dict[str, Any]) -> None`
### USAGE
* This unit test verifies that the `_hyper_collector` function correctly sets up the `hyper_list_all` in the returned dictionary based on the input hyperparameters.
* Parameters:
  * `input_data`: A dictionary containing the input hyperparameters and configurations for the `_hyper_collector` function.
### IMPL
* Prepare the input `hyperparameters_dict` with the necessary structure and values.
* Call the `_hyper_collector` function with the `hyperparameters_dict` and other necessary parameters.
* Extract the `hyper_list_all` from the result of `_hyper_collector`.
* Assert that `hyper_list_all` matches the expected dictionary structure and values for the hyperparameters.

## `test_hyper_bound_list_updated(input_data: Dict[str, Any]) -> None`
### USAGE
* This unit test checks that the `_hyper_collector` function correctly identifies and returns the list of hyperparameters that need to be optimized.
* Parameters:
  * `input_data`: A dictionary containing the input hyperparameters and configurations for the `_hyper_collector` function.
### IMPL
* Prepare the input `hyperparameters_dict` with fields that specify which hyperparameters are to be optimized.
* Call the `_hyper_collector` function with appropriate parameters.
* Extract the `hyper_bound_list_updated` from the `_hyper_collector` result.
* Assert that `hyper_bound_list_updated` matches the list of hyperparameters specified for optimization.

## `test_hyper_bound_list_fixed(input_data: Dict[str, Any]) -> None`
### USAGE
* This unit test ensures that `_hyper_collector` correctly identifies and returns the hyperparameters that are fixed and not to be optimized.
* Parameters:
  * `input_data`: A dictionary containing the input hyperparameters and configurations for the `_hyper_collector` function.
### IMPL
* Set up the `hyperparameters_dict` with specific values for fixed hyperparameters.
* Invoke the `_hyper_collector` method with the provided inputs.
* Obtain the `hyper_bound_list_fixed` from the result.
* Assert that `hyper_bound_list_fixed` is an empty dictionary as expected.

## `test_dt_hyper_fixed_mod(input_data: Dict[str, Any]) -> None`
### USAGE
* This unit test validates that `_hyper_collector` correctly processes and returns the `dt_hyper_fixed` data.
* Parameters:
  * `input_data`: A dictionary containing the input hyperparameters and configurations including `dt_hyper_fixed`.
### IMPL
* Prepare the input data including `dt_hyper_fixed` with specific fixed parameters.
* Call `_hyper_collector` using this input.
* Extract `dt_hyper_fixed_mod` from the function's output.
* Assert that `dt_hyper_fixed_mod` matches the expected DataFrame content.

## `test_all_fixed(input_data: Dict[str, Any]) -> None`
### USAGE
* This unit test checks if `_hyper_collector` correctly sets the `all_fixed` flag when all hyperparameters are fixed.
* Parameters:
  * `input_data`: A dictionary containing the input hyperparameters and configurations for the `_hyper_collector` function.
### IMPL
* Configure the `hyperparameters_dict` to have all hyperparameters as fixed.
* Invoke `_hyper_collector` with the input data.
* Retrieve the `all_fixed` flag from the result.
* Assert that `all_fixed` is set to `true`, indicating all hyperparameters are fixed.

## `test_rsq_train_is_float_between_0_and_1() -> None`
### USAGE
* This test verifies that the `rsq_train` attribute of the `ModelRefitOutput` is a float between 0 and 1, indicating a valid R-squared value for the training dataset.
### IMPL
* Prepare mock input data including `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test` with specified dimensions and random float values.
* Call the `_model_refit` method with the mock inputs and a specified `lambda_` parameter.
* Retrieve the `rsq_train` value from the `ModelRefitOutput`.
* Assert that the `rsq_train` is an instance of `float`.
* Assert that `0 <= rsq_train <= 1`, confirming it falls within the valid range for an R-squared value.

## `test_rsq_val_is_float_between_0_and_1() -> None`
### USAGE
* This test checks that the `rsq_val` attribute of the `ModelRefitOutput` is a float between 0 and 1, representing a valid R-squared value for the validation dataset.
### IMPL
* Prepare mock input data including `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call the `_model_refit` method using these inputs and set `lambda_` appropriately.
* Access the `rsq_val` from the resulting `ModelRefitOutput`.
* Assert that `rsq_val` is a `float`.
* Assert that `0 <= rsq_val <= 1` to ensure it is within the valid range for R-squared values.

## `test_rsq_test_is_float_between_0_and_1() -> None`
### USAGE
* This test ensures that the `rsq_test` attribute of the `ModelRefitOutput` is a float between 0 and 1, indicating a valid R-squared value for the test dataset.
### IMPL
* Mock input data for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test` with specified dimensions.
* Invoke the `_model_refit` method with these inputs and a defined `lambda_`.
* Extract `rsq_test` from the `ModelRefitOutput`.
* Assert that `rsq_test` is a `float`.
* Assert that `0 <= rsq_test <= 1` to confirm it is a valid R-squared value.

## `test_nrmse_train_is_positive() -> None`
### USAGE
* This test checks that the `nrmse_train` attribute of the `ModelRefitOutput` is a positive float, indicating the normalized root mean square error for the training dataset.
### IMPL
* Create mock data arrays for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call `_model_refit` with these inputs and a specified `lambda_`.
* Retrieve `nrmse_train` from the `ModelRefitOutput`.
* Assert that `nrmse_train` is a `float`.
* Assert that `nrmse_train > 0` to ensure it is a positive value.

## `test_nrmse_val_is_positive() -> None`
### USAGE
* This test verifies that the `nrmse_val` attribute of the `ModelRefitOutput` is a positive float, indicating the normalized root mean square error for the validation dataset.
### IMPL
* Prepare mock data for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call the `_model_refit` method with these inputs and a chosen `lambda_`.
* Access `nrmse_val` from the `ModelRefitOutput`.
* Assert that `nrmse_val` is a `float`.
* Assert that `nrmse_val > 0` to confirm it is a positive value.

## `test_nrmse_test_is_positive() -> None`
### USAGE
* This test ensures that the `nrmse_test` attribute of the `ModelRefitOutput` is a positive float, indicating the normalized root mean square error for the test dataset.
### IMPL
* Mock input data for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Invoke `_model_refit` with these inputs and a specific `lambda_`.
* Extract `nrmse_test` from the `ModelRefitOutput`.
* Assert that `nrmse_test` is a `float`.
* Assert that `nrmse_test > 0` to ensure it is a positive value.

## `test_coefs_is_array_of_length_5() -> None`
### USAGE
* This test checks that the `coefs` attribute of the `ModelRefitOutput` is a 1D array of length 5, representing the model coefficients.
### IMPL
* Create mock input data for the test, including `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call `_model_refit` using these inputs and a defined `lambda_`.
* Retrieve `coefs` from the `ModelRefitOutput`.
* Assert that `coefs` is an instance of `np.ndarray`.
* Assert that `len(coefs) == 5` to confirm it has the correct length.

## `test_y_train_pred_is_array_of_length_50() -> None`
### USAGE
* This test verifies that the `y_train_pred` attribute of the `ModelRefitOutput` is a 1D array of length 50, representing predicted values for the training dataset.
### IMPL
* Prepare mock data arrays for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Invoke `_model_refit` with these inputs and a specified `lambda_`.
* Access `y_train_pred` from the `ModelRefitOutput`.
* Assert that `y_train_pred` is an `np.ndarray`.
* Assert that `len(y_train_pred) == 50` to confirm it has the correct length.

## `test_y_val_pred_is_array_of_length_20() -> None`
### USAGE
* This test ensures that the `y_val_pred` attribute of the `ModelRefitOutput` is a 1D array of length 20, representing predicted values for the validation dataset.
### IMPL
* Mock input data for the test, including `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call `_model_refit` using these inputs and a chosen `lambda_`.
* Retrieve `y_val_pred` from the `ModelRefitOutput`.
* Assert that `y_val_pred` is an `np.ndarray`.
* Assert that `len(y_val_pred) == 20` to ensure it has the correct length.

## `test_y_test_pred_is_array_of_length_10() -> None`
### USAGE
* This test verifies the `y_test_pred` attribute of the `ModelRefitOutput` is a 1D array of length 10, indicating predicted values for the test dataset.
### IMPL
* Create mock data for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Invoke `_model_refit` with these inputs and the specified `lambda_`.
* Access `y_test_pred` from the `ModelRefitOutput`.
* Assert that `y_test_pred` is an `np.ndarray`.
* Assert that `len(y_test_pred) == 10` to confirm it has the correct length.

## `test_y_pred_is_array_of_length_80() -> None`
### USAGE
* This test ensures that the `y_pred` attribute of the `ModelRefitOutput` is a 1D array of length 80, representing concatenated predicted values for train, validation, and test datasets.
### IMPL
* Prepare mock input data arrays for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call the `_model_refit` method with these inputs and a specified `lambda_`.
* Extract `y_pred` from the `ModelRefitOutput`.
* Assert that `y_pred` is an `np.ndarray`.
* Assert that `len(y_pred) == 80` to ensure it has the correct length.

## `test_mod_is_ridge_instance() -> None`
### USAGE
* This test checks that the `mod` attribute of the `ModelRefitOutput` is an instance of a Ridge model, confirming the type of model used for fitting.
### IMPL
* Create mock data for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call `_model_refit` with these inputs and the defined `lambda_`.
* Retrieve `mod` from the `ModelRefitOutput`.
* Assert that `mod` is an instance of `Ridge`, verifying the model type.

## `test_df_int_is_1() -> None`
### USAGE
* This test verifies that the `df_int` attribute of the `ModelRefitOutput` equals 1, indicating the intercept status of the model.
### IMPL
* Prepare mock input data for `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test`.
* Call the `_model_refit` method with these inputs and a specified `lambda_`.
* Access `df_int` from the `ModelRefitOutput`.
* Assert that `df_int == 1` to confirm the model includes an intercept.

## `test_rsq_train() -> None`
### USAGE
* This test checks if the R-squared value for the training set (`rsq_train`) is a float between 0 and 1.
* It ensures that the R-squared metric is calculated correctly for the training data.
### IMPL
* Prepare mock input data:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters.
* Capture the `rsq_train` output from the method.
* Assert that `rsq_train` is a float.
* Assert that `rsq_train` is between 0 and 1.

## `test_rsq_val() -> None`
### USAGE
* This test checks if the R-squared value for the validation set (`rsq_val`) is `None` when no validation data is provided.
* It ensures that the method handles missing validation data correctly.
### IMPL
* Prepare mock input data without validation sets:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters, excluding validation data.
* Capture the `rsq_val` output from the method.
* Assert that `rsq_val` is `None`.

## `test_rsq_test() -> None`
### USAGE
* This test checks if the R-squared value for the test set (`rsq_test`) is `None` when no test data is provided.
* It ensures that the method handles missing test data correctly.
### IMPL
* Prepare mock input data without test sets:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters, excluding test data.
* Capture the `rsq_test` output from the method.
* Assert that `rsq_test` is `None`.

## `test_nrmse_train() -> None`
### USAGE
* This test checks if the Normalized Root Mean Square Error for the training set (`nrmse_train`) is a positive float.
* It ensures the correct calculation of the NRMSE metric for training data.
### IMPL
* Prepare mock input data:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters.
* Capture the `nrmse_train` output from the method.
* Assert that `nrmse_train` is a float.
* Assert that `nrmse_train` is greater than 0.

## `test_nrmse_val() -> None`
### USAGE
* This test checks if the Normalized Root Mean Square Error for the validation set (`nrmse_val`) is `None` when no validation data is provided.
* It ensures that the method handles missing validation data correctly.
### IMPL
* Prepare mock input data without validation sets:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters, excluding validation data.
* Capture the `nrmse_val` output from the method.
* Assert that `nrmse_val` is `None`.

## `test_nrmse_test() -> None`
### USAGE
* This test checks if the Normalized Root Mean Square Error for the test set (`nrmse_test`) is `None` when no test data is provided.
* It ensures that the method handles missing test data correctly.
### IMPL
* Prepare mock input data without test sets:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters, excluding test data.
* Capture the `nrmse_test` output from the method.
* Assert that `nrmse_test` is `None`.

## `test_coefs() -> None`
### USAGE
* This test checks if the coefficients (`coefs`) of the model are a 1D array of length 5.
* It ensures that the model coefficients are calculated correctly and have the expected shape.
### IMPL
* Prepare mock input data:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters.
* Capture the `coefs` output from the method.
* Assert that `coefs` is a numpy array.
* Assert that the length of `coefs` is 5.

## `test_y_train_pred() -> None`
### USAGE
* This test checks if the predicted values for the training set (`y_train_pred`) are a 1D array of length 50.
* It ensures that predictions on the training data are made correctly.
### IMPL
* Prepare mock input data:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters.
* Capture the `y_train_pred` output from the method.
* Assert that `y_train_pred` is a numpy array.
* Assert that the length of `y_train_pred` is 50.

## `test_y_val_pred() -> None`
### USAGE
* This test checks if the predicted values for the validation set (`y_val_pred`) are `None` when no validation data is provided.
* It ensures that the method handles missing validation data correctly for predictions.
### IMPL
* Prepare mock input data without validation sets:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters, excluding validation data.
* Capture the `y_val_pred` output from the method.
* Assert that `y_val_pred` is `None`.

## `test_y_test_pred() -> None`
### USAGE
* This test checks if the predicted values for the test set (`y_test_pred`) are `None` when no test data is provided.
* It ensures that the method handles missing test data correctly for predictions.
### IMPL
* Prepare mock input data without test sets:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters, excluding test data.
* Capture the `y_test_pred` output from the method.
* Assert that `y_test_pred` is `None`.

## `test_y_pred() -> None`
### USAGE
* This test checks if the predicted values (`y_pred`) are a 1D array of length 50 when only training data is provided.
* It ensures that predictions are made correctly across all provided data.
### IMPL
* Prepare mock input data:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters.
* Capture the `y_pred` output from the method.
* Assert that `y_pred` is a numpy array.
* Assert that the length of `y_pred` is 50.

## `test_mod() -> None`
### USAGE
* This test checks if the model instance (`mod`) is an instance of the Ridge regression model.
* It verifies that the Ridge model is created correctly.
### IMPL
* Prepare mock input data:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters.
* Capture the `mod` output from the method.
* Assert that `mod` is an instance of `Ridge`.

## `test_df_int() -> None`
### USAGE
* This test checks if the degree of freedom for the intercept (`df_int`) is equal to 1.
* It ensures that the intercept is correctly included in the model.
### IMPL
* Prepare mock input data:
  - Create a 2D numpy array `x_train` of shape (50, 5) with random float numbers.
  - Create a 1D numpy array `y_train` of length 50 with random float numbers.
* Call the `_model_refit` method with the mock data and default parameters.
* Capture the `df_int` output from the method.
* Assert that `df_int` is equal to 1.

## `test_rsq_train_is_float_between_0_and_1() -> None`
### USAGE
* This test checks if the `rsq_train` value returned by the `_model_refit` function is a float between 0 and 1, indicating valid R-squared for training data.
### IMPL
* Arrange the test by creating mock data inputs for `x_train` and `y_train` with appropriate shapes and random float numbers.
* Call the `_model_refit` method with these inputs and other parameters such as `lambda_`, `intercept`, and `intercept_sign`.
* Extract `rsq_train` from the returned `ModelRefitOutput` object.
* Assert that `rsq_train` is an instance of float.
* Assert that `0 <= rsq_train <= 1`.

## `test_rsq_val_is_float_between_0_and_1() -> None`
### USAGE
* This test ensures that `rsq_val` value is a float between 0 and 1, validating the R-squared for validation data.
### IMPL
* Arrange mock data for `x_train`, `y_train`, `x_val`, and `y_val`.
* Execute `_model_refit` using these mock inputs.
* Extract `rsq_val` from the `ModelRefitOutput`.
* Assert that `rsq_val` is either None or a float.
* If `rsq_val` is a float, assert that `0 <= rsq_val <= 1`.

## `test_rsq_test_is_none() -> None`
### USAGE
* This test checks that `rsq_test` is None, as no test data is provided.
### IMPL
* Provide mock inputs for `x_train` and `y_train`, leaving `x_test` and `y_test` as None.
* Invoke `_model_refit`.
* Extract `rsq_test` from the `ModelRefitOutput`.
* Assert `rsq_test` is None.

## `test_nrmse_train_is_positive_float() -> None`
### USAGE
* Validates that `nrmse_train` is a positive float, indicating a valid normalized RMSE value.
### IMPL
* Use mock data for `x_train` and `y_train`.
* Call `_model_refit`.
* Access `nrmse_train` from the output.
* Assert that `nrmse_train` is a float.
* Assert that `nrmse_train` is greater than 0.

## `test_nrmse_val_is_positive_float() -> None`
### USAGE
* Verifies that `nrmse_val` is a positive float when validation data is available.
### IMPL
* Provide mock inputs including `x_val` and `y_val`.
* Call `_model_refit`.
* Extract `nrmse_val`.
* Assert that `nrmse_val` is either None or a float.
* If `nrmse_val` is a float, assert that it is greater than 0.

## `test_nrmse_test_is_none() -> None`
### USAGE
* Confirms that `nrmse_test` is None due to absence of test data.
### IMPL
* Arrange input for `x_train` and `y_train`, leaving `x_test` and `y_test` as None.
* Call `_model_refit`.
* Retrieve `nrmse_test`.
* Assert that `nrmse_test` is None.

## `test_coefs_is_array_of_length_5() -> None`
### USAGE
* Ensures that the `coefs` array length is 5, matching the feature count.
### IMPL
* Use mock data for `x_train` and `y_train` with 5 features.
* Call `_model_refit`.
* Access `coefs`.
* Assert that `coefs` is a numpy array.
* Assert that its length is 5.

## `test_y_train_pred_is_array_of_length_50() -> None`
### USAGE
* Confirms that `y_train_pred` is an array of length 50, matching the training data.
### IMPL
* Provide mock `x_train` and `y_train` data of length 50.
* Call `_model_refit`.
* Retrieve `y_train_pred`.
* Assert that `y_train_pred` is a numpy array.
* Assert that its length is 50.

## `test_y_val_pred_is_array_of_length_20() -> None`
### USAGE
* Validates that `y_val_pred` has a length of 20, consistent with validation data size.
### IMPL
* Arrange `x_val` and `y_val` with 20 samples.
* Call `_model_refit`.
* Extract `y_val_pred`.
* Assert `y_val_pred` is either None or a numpy array.
* If it is an array, assert that its length is 20.

## `test_y_test_pred_is_none() -> None`
### USAGE
* Ensures `y_test_pred` is None due to lack of test data.
### IMPL
* Arrange inputs with `x_test` and `y_test` as None.
* Execute `_model_refit`.
* Access `y_test_pred`.
* Assert that `y_test_pred` is None.

## `test_y_pred_is_array_of_length_70() -> None`
### USAGE
* Checks that `y_pred` length is 70, summing training and validation predictions.
### IMPL
* Use mock data for `x_train`, `y_train`, `x_val`, and `y_val`.
* Call `_model_refit`.
* Retrieve `y_pred`.
* Assert `y_pred` is a numpy array.
* Assert its length is 70.

## `test_mod_is_ridge_instance() -> None`
### USAGE
* Confirms that `mod` is an instance of the Ridge model.
### IMPL
* Provide necessary inputs for `_model_refit`.
* Call the method.
* Access `mod`.
* Assert `mod` is an instance of `Ridge`.

## `test_df_int_is_1() -> None`
### USAGE
* Verifies that `df_int` is 1 when intercept is fitted.
### IMPL
* Call `_model_refit` with `intercept=True`.
* Extract `df_int` from the output.
* Assert that `df_int` equals 1.

## `test_rsq_train_is_between_0_and_1() -> None`
### USAGE
* Verify that the `rsq_train` value returned by the `_model_refit` function is a float between 0 and 1, indicating a valid R-squared value for the training dataset.
### IMPL
* Prepare a 2D array `x_train` of shape (50, 5) with random float values, simulating the feature matrix for training.
* Prepare a 1D array `y_train` of length 50 with random float values, representing the target variable for training.
* Set `lambda_` to 1000 to test with an unusual lambda value.
* Call the `_model_refit` static method with `intercept` set to `False` and other parameters. Ensure `x_val`, `y_val`, `x_test`, and `y_test` are `None`.
* Capture the `rsq_train` value from the returned `ModelRefitOutput`.
* Assert that `rsq_train` is greater than or equal to 0 and less than or equal to 1.

## `test_rsq_val_is_between_0_and_1() -> None`
### USAGE
* Verify that the `rsq_val` value returned by the `_model_refit` function is a float between 0 and 1, indicating a valid R-squared value for the validation dataset.
### IMPL
* Prepare a 2D array `x_train` of shape (50, 5) and `x_val` of shape (20, 5) with random float values.
* Prepare a 1D array `y_train` of length 50 and `y_val` of length 20 with random float values.
* Set `lambda_` to 1000 for testing.
* Call the `_model_refit` static method with `intercept` set to `False` and the prepared data.
* Capture the `rsq_val` value from the returned `ModelRefitOutput`.
* Assert that `rsq_val` is greater than or equal to 0 and less than or equal to 1.

## `test_rsq_test_is_none() -> None`
### USAGE
* Ensure that the `rsq_test` value returned by the `_model_refit` function is `None`, given that no test data is provided.
### IMPL
* Prepare `x_train` and `x_val` arrays as before, but omit `x_test`.
* Prepare `y_train` and `y_val` arrays, omitting `y_test`.
* Execute `_model_refit` with the same setup.
* Capture `rsq_test` from the returned `ModelRefitOutput`.
* Assert that `rsq_test` is `None`.

## `test_nrmse_train_is_greater_than_0() -> None`
### USAGE
* Validate that the `nrmse_train` value from `_model_refit` is a positive float, representing the normalized root mean square error for the training data.
### IMPL
* Create `x_train` and `y_train` as described above.
* Set `lambda_` to 1000.
* Call `_model_refit` with these inputs.
* Extract `nrmse_train` from the result.
* Assert that `nrmse_train` is greater than 0.

## `test_nrmse_val_is_greater_than_0() -> None`
### USAGE
* Confirm that the `nrmse_val` value from `_model_refit` is a positive float, indicating the normalized root mean square error for validation data.
### IMPL
* Define `x_train`, `y_train`, `x_val`, and `y_val` as described.
* Set `lambda_` to 1000.
* Call `_model_refit` and capture `nrmse_val`.
* Assert `nrmse_val` is greater than 0.

## `test_nrmse_test_is_none() -> None`
### USAGE
* Check that `nrmse_test` is `None` when no test data is provided to `_model_refit`.
### IMPL
* Prepare `x_train`, `y_train`, `x_val`, `y_val` without test data.
* Call `_model_refit` with these inputs.
* Capture `nrmse_test`.
* Assert that `nrmse_test` is `None`.

## `test_coefs_length_is_5() -> None`
### USAGE
* Ensure the `coefs` array from `_model_refit` has a length of 5, matching the number of features in `x_train`.
### IMPL
* Use `x_train` and `y_train` as before.
* Set `lambda_` to 1000.
* Call `_model_refit`.
* Capture `coefs`.
* Assert `coefs` has a length of 5.

## `test_y_train_pred_length_is_50() -> None`
### USAGE
* Verify `y_train_pred` from `_model_refit` has a length of 50, corresponding to `y_train`.
### IMPL
* Prepare `x_train` and `y_train`.
* Set `lambda_` to 1000.
* Call `_model_refit`.
* Capture `y_train_pred`.
* Assert `y_train_pred` has a length of 50.

## `test_y_val_pred_length_is_20() -> None`
### USAGE
* Confirm `y_val_pred` from `_model_refit` has a length of 20, matching `y_val`.
### IMPL
* Define `x_train`, `y_train`, `x_val`, `y_val`.
* Set `lambda_` to 1000.
* Call `_model_refit`.
* Capture `y_val_pred`.
* Assert `y_val_pred` has a length of 20.

## `test_y_test_pred_is_none() -> None`
### USAGE
* Check that `y_test_pred` is `None` when no test data is provided to `_model_refit`.
### IMPL
* Use `x_train`, `y_train`, `x_val`, `y_val`.
* Execute `_model_refit`.
* Capture `y_test_pred`.
* Assert `y_test_pred` is `None`.

## `test_y_pred_length_is_70() -> None`
### USAGE
* Ensure `y_pred` from `_model_refit` has a length of 70, combining predictions from train, val, and test data.
### IMPL
* Prepare `x_train`, `y_train`, `x_val`, `y_val`.
* Set `lambda_` to 1000.
* Call `_model_refit`.
* Capture `y_pred`.
* Assert `y_pred` has a length of 70.

## `test_model_instance_is_ridge() -> None`
### USAGE
* Verify the `mod` attribute from `_model_refit` is an instance of `Ridge`.
### IMPL
* Define `x_train`, `y_train`, `x_val`, `y_val`.
* Set `lambda_` to 1000.
* Call `_model_refit`.
* Capture `mod`.
* Assert `mod` is an instance of `Ridge`.

## `test_df_int_is_0_when_no_intercept() -> None`
### USAGE
* Ensure that `df_int` is 0 when `intercept` is set to `False`.
### IMPL
* Prepare `x_train`, `y_train`, `x_val`, `y_val`.
* Set `lambda_` to 1000 and `intercept` to `False`.
* Call `_model_refit`.
* Capture `df_int`.
* Assert `df_int` is 0.

## `test_lambda_seq_length() -> None`
### USAGE
* This test verifies that the `_lambda_seq` method generates a sequence of the correct length.
* Parameters:
  - `x`: A random 10x5 numpy array representing feature data.
  - `y`: A random 10-element numpy array representing target data.
  - `seq_len`: The expected length of the lambda sequence, set to 100 in this test.
  - `lambda_min_ratio`: The minimum ratio for the lambda sequence, set to 0.0001 in this test.
### IMPL
* Prepare mock input data by generating a random 10x5 numpy array `x` and a random 10-element numpy array `y`.
* Set the `seq_len` parameter to 100 and `lambda_min_ratio` to 0.0001.
* Call the `_lambda_seq` function with `x`, `y`, `seq_len`, and `lambda_min_ratio` as arguments.
* Capture the output sequence from the function call.
* Assert that the length of the output sequence equals `seq_len` (100).
* Confirm that the test passes if the output sequence has the expected length.

## `test_lambda_seq_min_value() -> None`
### USAGE
* This test checks if the minimum value in the generated lambda sequence is greater than 0.
* Parameters:
  - `x`: A random 10x5 numpy array representing feature data.
  - `y`: A random 10-element numpy array representing target data.
  - `seq_len`: The expected length of the lambda sequence.
  - `lambda_min_ratio`: The minimum ratio for the lambda sequence.
### IMPL
* Prepare mock input data by generating a random 10x5 numpy array `x` and a random 10-element numpy array `y`.
* Set the `seq_len` parameter to 100 and `lambda_min_ratio` to 0.0001.
* Call the `_lambda_seq` function with `x`, `y`, `seq_len`, and `lambda_min_ratio` as arguments.
* Capture the output sequence from the function call.
* Assert that the minimum value in the output sequence is greater than 0.
* Confirm that the test passes if the minimum value in the sequence is indeed greater than 0.

## `test_lambda_seq_output_type() -> None`
### USAGE
* This test ensures that the output of the `_lambda_seq` method is a numpy ndarray.
* Parameters:
  - `x`: A random 10x5 numpy array representing feature data.
  - `y`: A random 10-element numpy array representing target data.
  - `seq_len`: The expected length of the lambda sequence.
  - `lambda_min_ratio`: The minimum ratio for the lambda sequence.
### IMPL
* Prepare mock input data by generating a random 10x5 numpy array `x` and a random 10-element numpy array `y`.
* Set the `seq_len` parameter to 100 and `lambda_min_ratio` to 0.0001.
* Call the `_lambda_seq` function with `x`, `y`, `seq_len`, and `lambda_min_ratio` as arguments.
* Capture the output sequence from the function call.
* Assert that the type of the output sequence is `numpy.ndarray`.
* Confirm that the test passes if the output sequence is of type `numpy.ndarray`.

## `test_lambda_seq_length(x: np.ndarray, y: np.ndarray, seq_len: int = 100, lambda_min_ratio: float = 0.0001) -> None`
### USAGE
* This test function is designed to verify the length of the lambda sequence generated by the `_lambda_seq` method.
* Parameters:
  * `x`: A Numpy array with shape (1, 1) representing the feature input.
  * `y`: A Numpy array with a single element representing the target input.
  * `seq_len`: The expected length of the lambda sequence, set to 100.
  * `lambda_min_ratio`: The minimum ratio for the lambda sequence, set to 0.0001.
### IMPL
* Initialize the input `x` as a Numpy array with a shape of (1, 1), containing a single feature.
* Initialize the input `y` as a Numpy array with a single target element.
* Call the `_lambda_seq` method with the inputs `x`, `y`, `seq_len`, and `lambda_min_ratio`.
* Capture the result from the `_lambda_seq` method.
* Assert that the length of the result is equal to `seq_len` (100).

## `test_lambda_seq_min_value(x: np.ndarray, y: np.ndarray, seq_len: int = 100, lambda_min_ratio: float = 0.0001) -> None`
### USAGE
* This test function is designed to verify that the minimum value of the lambda sequence generated by the `_lambda_seq` method is greater than 0.
* Parameters:
  * `x`: A Numpy array with shape (1, 1) representing the feature input.
  * `y`: A Numpy array with a single element representing the target input.
  * `seq_len`: The length of the lambda sequence, set to 100.
  * `lambda_min_ratio`: The minimum ratio for the lambda sequence, set to 0.0001.
### IMPL
* Initialize the input `x` as a Numpy array with a shape of (1, 1), containing a single feature.
* Initialize the input `y` as a Numpy array with a single target element.
* Call the `_lambda_seq` method with the inputs `x`, `y`, `seq_len`, and `lambda_min_ratio`.
* Capture the result from the `_lambda_seq` method.
* Assert that the minimum value of the result is greater than 0.

## `test_lambda_seq_result_type(x: np.ndarray, y: np.ndarray, seq_len: int = 100, lambda_min_ratio: float = 0.0001) -> None`
### USAGE
* This test function is designed to verify that the result of the lambda sequence generated by the `_lambda_seq` method is of type `np.ndarray`.
* Parameters:
  * `x`: A Numpy array with shape (1, 1) representing the feature input.
  * `y`: A Numpy array with a single element representing the target input.
  * `seq_len`: The length of the lambda sequence, set to 100.
  * `lambda_min_ratio`: The minimum ratio for the lambda sequence, set to 0.0001.
### IMPL
* Initialize the input `x` as a Numpy array with a shape of (1, 1), containing a single feature.
* Initialize the input `y` as a Numpy array with a single target element.
* Call the `_lambda_seq` method with the inputs `x`, `y`, `seq_len`, and `lambda_min_ratio`.
* Capture the result from the `_lambda_seq` method.
* Assert that the type of the result is `np.ndarray`.

## `test_lambda_seq_length(x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float) -> None`
### USAGE
* Tests if the lambda sequence generated by the `_lambda_seq` method has the expected length.
* Parameters:
  * `x`: A random 10x5 numpy array representing features.
  * `y`: A random 10-element numpy array representing the target variable.
  * `seq_len`: The length of the lambda sequence to be generated, expected to be 100.
  * `lambda_min_ratio`: The minimum ratio for the lambda sequence, expected to be 0.0.

### IMPL
1. *Setup Input Data:*
   * Create a random 10x5 numpy array `x` as input features.
   * Create a random 10-element numpy array `y` as the target variable.
   * Set `seq_len` to 100 to specify the expected length of the lambda sequence.
   * Set `lambda_min_ratio` to 0.0 as the input parameter for generating the sequence.

2. *Invoke Method:*
   * Call the `_lambda_seq` method with inputs `x`, `y`, `seq_len`, and `lambda_min_ratio`.

3. *Assert Result Length:*
   * Assert that the length of the result from `_lambda_seq` is equal to `seq_len` (100).

## `test_lambda_seq_max_value(x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float) -> None`
### USAGE
* Tests if the maximum value of the lambda sequence generated by the `_lambda_seq` method is greater than 0.
* Parameters:
  * `x`: A random 10x5 numpy array representing features.
  * `y`: A random 10-element numpy array representing the target variable.
  * `seq_len`: The length of the lambda sequence to be generated, expected to be 100.
  * `lambda_min_ratio`: The minimum ratio for the lambda sequence, expected to be 0.0.

### IMPL
1. *Setup Input Data:*
   * Create a random 10x5 numpy array `x` as input features.
   * Create a random 10-element numpy array `y` as the target variable.
   * Set `seq_len` to 100 to specify the expected length of the lambda sequence.
   * Set `lambda_min_ratio` to 0.0 as the input parameter for generating the sequence.

2. *Invoke Method:*
   * Call the `_lambda_seq` method with inputs `x`, `y`, `seq_len`, and `lambda_min_ratio`.

3. *Assert Maximum Value:*
   * Assert that the maximum value in the resulting lambda sequence is greater than 0.

## `test_lambda_seq_type(x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float) -> None`
### USAGE
* Tests if the type of the lambda sequence generated by the `_lambda_seq` method is a numpy ndarray.
* Parameters:
  * `x`: A random 10x5 numpy array representing features.
  * `y`: A random 10-element numpy array representing the target variable.
  * `seq_len`: The length of the lambda sequence to be generated, expected to be 100.
  * `lambda_min_ratio`: The minimum ratio for the lambda sequence, expected to be 0.0.

### IMPL
1. *Setup Input Data:*
   * Create a random 10x5 numpy array `x` as input features.
   * Create a random 10-element numpy array `y` as the target variable.
   * Set `seq_len` to 100 to specify the expected length of the lambda sequence.
   * Set `lambda_min_ratio` to 0.0 as the input parameter for generating the sequence.

2. *Invoke Method:*
   * Call the `_lambda_seq` method with inputs `x`, `y`, `seq_len`, and `lambda_min_ratio`.

3. *Assert Result Type:*
   * Assert that the result from `_lambda_seq` is of type `np.ndarray`.

## `test_lambda_seq_length() -> None`
### USAGE
* This test function validates that the length of the lambda sequence generated by `_lambda_seq` is equal to the expected sequence length.
* No parameters are required as the function will generate its own test data.
### IMPL
* Generate a random 10x5 numpy array `x` to simulate feature data.
* Generate a random 10-element numpy array `y` to simulate the target variable.
* Set the sequence length `seq_len` to 100.
* Set the `lambda_min_ratio` to 1.0 for maximum reduction.
* Call the `_lambda_seq` method with `x`, `y`, `seq_len`, and `lambda_min_ratio` as inputs.
* Capture the result into a variable `result`.
* Assert that the length of `result` is equal to `seq_len` i.e., 100.

## `test_lambda_seq_min_value() -> None`
### USAGE
* This test function ensures that the minimum value in the lambda sequence is greater than 0.
* No parameters are required as the function will generate its own test data.
### IMPL
* Generate a random 10x5 numpy array `x` to simulate feature data.
* Generate a random 10-element numpy array `y` to simulate the target variable.
* Set the sequence length `seq_len` to 100.
* Set the `lambda_min_ratio` to 1.0 for maximum reduction.
* Call the `_lambda_seq` method with `x`, `y`, `seq_len`, and `lambda_min_ratio` as inputs.
* Capture the result into a variable `result`.
* Assert that the minimum value of `result` is greater than 0.

## `test_lambda_seq_max_equals_min() -> None`
### USAGE
* This test function verifies that the maximum value in the lambda sequence is equal to the minimum value when `lambda_min_ratio` is set to 1.0.
* No parameters are required as the function will generate its own test data.
### IMPL
* Generate a random 10x5 numpy array `x` to simulate feature data.
* Generate a random 10-element numpy array `y` to simulate the target variable.
* Set the sequence length `seq_len` to 100.
* Set the `lambda_min_ratio` to 1.0 for maximum reduction.
* Call the `_lambda_seq` method with `x`, `y`, `seq_len`, and `lambda_min_ratio` as inputs.
* Capture the result into a variable `result`.
* Assert that the maximum value of `result` is equal to the minimum value of `result`.

## `test_lambda_seq_type() -> None`
### USAGE
* This test function checks that the result of the lambda sequence generation is of type `np.ndarray`.
* No parameters are required as the function will generate its own test data.
### IMPL
* Generate a random 10x5 numpy array `x` to simulate feature data.
* Generate a random 10-element numpy array `y` to simulate the target variable.
* Set the sequence length `seq_len` to 100.
* Set the `lambda_min_ratio` to 1.0 for maximum reduction.
* Call the `_lambda_seq` method with `x`, `y`, `seq_len`, and `lambda_min_ratio` as inputs.
* Capture the result into a variable `result`.
* Assert that the type of `result` is `np.ndarray`.

## `test_lambda_seq_output_length(x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float) -> None`
### USAGE
* This test verifies that the `_lambda_seq` function returns an array of the expected length specified by `seq_len`.
* Parameters:
  - `x`: Random 10x5 numpy array representing the feature matrix.
  - `y`: Random 10-element numpy array representing the target vector.
  - `seq_len`: The length of the lambda sequence, set to 1 for this test.
  - `lambda_min_ratio`: The minimum ratio for the lambda sequence, set to `0.0001` for this test.

### IMPL
* Initialize the necessary input parameters: `x` as a random 10x5 numpy array, `y` as a random 10-element numpy array, `seq_len` as 1, and `lambda_min_ratio` as 0.0001.
* Call the `_lambda_seq` method using these input parameters.
* Capture the result of the method call.
* Assert that the length of the result is equal to the expected value, which is 1, using an assertion statement such as `assert len(result) == 1`.
* Document any assumptions or edge cases addressed by this test, particularly focusing on the boundary condition where `seq_len` is minimized.

## `test_lambda_seq_output_type(x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float) -> None`
### USAGE
* This test ensures that the `_lambda_seq` function returns a numpy array as the output type.
* Parameters:
  - `x`: Random 10x5 numpy array representing the feature matrix.
  - `y`: Random 10-element numpy array representing the target vector.
  - `seq_len`: The length of the lambda sequence, set to 1 for this test.
  - `lambda_min_ratio`: The minimum ratio for the lambda sequence, set to `0.0001` for this test.

### IMPL
* Initialize the necessary input parameters: `x` as a random 10x5 numpy array, `y` as a random 10-element numpy array, `seq_len` as 1, and `lambda_min_ratio` as 0.0001.
* Call the `_lambda_seq` method using these input parameters.
* Capture the result of the method call.
* Assert that the type of the result is `np.ndarray` using an assertion statement such as `assert isinstance(result, np.ndarray)`.
* Provide clear documentation on the importance of verifying the output type, including any implications for downstream processing or integration.
