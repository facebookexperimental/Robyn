# CLASS
## RidgeModelBuilderTest
* The `RidgeModelBuilderTest` class is designed to implement unit tests for the `RidgeModelBuilder` class.
* This class focuses on verifying the correctness and robustness of the RidgeModelBuilder's methods, ensuring they work as expected under various conditions by utilizing mock objects and assertions.
* This test class is crucial for maintaining the reliability of the model building and evaluation functionalities provided by the `RidgeModelBuilder`.

# CONSTRUCTORS
## RidgeModelBuilderTest `(ridge_model_builder: RidgeModelBuilder)`
* The constructor for the `RidgeModelBuilderTest` class initializes the test instance with a `RidgeModelBuilder` object.
* The `ridge_model_builder` parameter is an instance of the `RidgeModelBuilder` class that will be tested.

### USAGE
* To use this constructor, instantiate the `RidgeModelBuilderTest` class with a pre-configured `RidgeModelBuilder` instance.
* This allows the test class to access and test various methods of the `RidgeModelBuilder`.

### IMPL
* The constructor should ensure that the `RidgeModelBuilder` instance is correctly initialized and accessible for all test methods.
* It should set up any necessary preconditions or mock configurations needed for the tests.
* The tests should examine the interaction between the `RidgeModelBuilder` and its dependencies, verifying the expected behavior through assertions.
* Ensure that the `RidgeModelBuilder` instance is not modified during testing unless necessary for particular test cases.

# METHODS


## `test_build_models_default_parameters() -> None`
### USAGE
* This test validates the `build_models` function with default parameters and a single trial.
### IMPL
* Mock `RidgeModelBuilder._hyper_collector` to return an empty hyperparameter collection.
  * The method should receive `Hyperparameters instance`, `ts_validation=False`, `add_penalty_factor=False`, `dt_hyper_fixed=None`, and `cores=2`.
  * It should return a dictionary with keys `hyper_list_all`, `all_fixed`, `hyper_bound_list_updated`, and `hyper_bound_list_fixed`, all with empty lists as their values.
* Mock `RidgeModelBuilder._run_nevergrad_optimization` to return a `Trial` instance.
  * The method should receive the mocked hyperparameters collection, `iterations=1`, `cores=2`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `ts_validation=False`, `add_penalty_factor=False`, `objective_weights=[0.5, 0.5]`, `dt_hyper_fixed=None`, `rssd_zero_penalty=True`, `trial=1`, `seed=124`, `total_trials=1`.
* Mock `Convergence.calculate_convergence` to return `Convergence results`.
  * The method should receive a list containing the mocked `Trial` instance.
* Call `build_models` with the input parameters: `trials_config={'trials': 1, 'iterations': 1}`, `dt_hyper_fixed=None`, `ts_validation=False`, `add_penalty_factor=False`, `seed=123`, `rssd_zero_penalty=True`, `objective_weights=None`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `cores=2`.
* Assert that `model_outputs.trials` is a list containing one `Trial` instance.
* Assert that `model_outputs.convergence` equals `Convergence results`.
* Assert that `model_outputs.hyper_updated` is an empty list.

## `test_build_models_with_ts_validation() -> None`
### USAGE
* This test checks the `build_models` function with time series validation enabled.
### IMPL
* Mock `RidgeModelBuilder._hyper_collector` to handle `ts_validation=True`.
  * The method should receive `Hyperparameters instance`, `ts_validation=True`, `add_penalty_factor=False`, `dt_hyper_fixed=None`, and `cores=2`.
  * It should return a dictionary with empty lists for `hyper_list_all`, `all_fixed`, `hyper_bound_list_updated`, and `hyper_bound_list_fixed`.
* Mock `RidgeModelBuilder._run_nevergrad_optimization` to return a `Trial` instance with `ts_validation=True`.
  * The method should receive the mocked hyperparameters collection, `iterations=1`, `cores=2`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `ts_validation=True`, `add_penalty_factor=False`, `objective_weights=[0.5, 0.5]`, `dt_hyper_fixed=None`, `rssd_zero_penalty=True`, `trial=1`, `seed=124`, `total_trials=1`.
* Mock `Convergence.calculate_convergence` to return `Convergence results`.
  * The method should receive a list containing the mocked `Trial` instance.
* Call `build_models` with the input parameters: `trials_config={'trials': 1, 'iterations': 1}`, `dt_hyper_fixed=None`, `ts_validation=True`, `add_penalty_factor=False`, `seed=123`, `rssd_zero_penalty=True`, `objective_weights=None`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `cores=2`.
* Assert that `model_outputs.ts_validation` is `True`.
* Assert that `model_outputs.trials` is a list containing one `Trial` instance.

## `test_build_models_with_penalty_factor() -> None`
### USAGE
* This test assesses `build_models` when a penalty factor is added.
### IMPL
* Mock `RidgeModelBuilder._hyper_collector` to include `add_penalty_factor=True`.
  * The method should receive `Hyperparameters instance`, `ts_validation=False`, `add_penalty_factor=True`, `dt_hyper_fixed=None`, and `cores=2`.
  * It should return a dictionary with empty lists for `hyper_list_all`, `all_fixed`, `hyper_bound_list_updated`, and `hyper_bound_list_fixed`.
* Mock `RidgeModelBuilder._run_nevergrad_optimization` to return a `Trial` instance with `add_penalty_factor=True`.
  * The method should receive the mocked hyperparameters collection, `iterations=1`, `cores=2`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `ts_validation=False`, `add_penalty_factor=True`, `objective_weights=[0.5, 0.5]`, `dt_hyper_fixed=None`, `rssd_zero_penalty=True`, `trial=1`, `seed=124`, `total_trials=1`.
* Mock `Convergence.calculate_convergence` to return `Convergence results`.
  * The method should receive a list containing the mocked `Trial` instance.
* Call `build_models` with the input parameters: `trials_config={'trials': 1, 'iterations': 1}`, `dt_hyper_fixed=None`, `ts_validation=False`, `add_penalty_factor=True`, `seed=123`, `rssd_zero_penalty=True`, `objective_weights=None`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `cores=2`.
* Assert that `model_outputs.add_penalty_factor` is `True`.
* Assert that `model_outputs.trials` is a list containing one `Trial` instance.

## `test_build_models_with_different_objective_weights() -> None`
### USAGE
* This test checks `build_models` with different objective weights.
### IMPL
* Mock `RidgeModelBuilder._hyper_collector` with default parameters.
  * The method should receive `Hyperparameters instance`, `ts_validation=False`, `add_penalty_factor=False`, `dt_hyper_fixed=None`, and `cores=2`.
  * It should return a dictionary with empty lists for `hyper_list_all`, `all_fixed`, `hyper_bound_list_updated`, and `hyper_bound_list_fixed`.
* Mock `RidgeModelBuilder._run_nevergrad_optimization` to return a `Trial` instance with custom objective weights.
  * The method should receive the mocked hyperparameters collection, `iterations=1`, `cores=2`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `ts_validation=False`, `add_penalty_factor=False`, `objective_weights=[0.3, 0.3, 0.4]`, `dt_hyper_fixed=None`, `rssd_zero_penalty=True`, `trial=1`, `seed=124`, `total_trials=1`.
* Mock `Convergence.calculate_convergence` to return `Convergence results`.
  * The method should receive a list containing the mocked `Trial` instance.
* Call `build_models` with the input parameters: `trials_config={'trials': 1, 'iterations': 1}`, `dt_hyper_fixed=None`, `ts_validation=False`, `add_penalty_factor=False`, `seed=123`, `rssd_zero_penalty=True`, `objective_weights=[0.3, 0.3, 0.4]`, `nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE`, `intercept=True`, `intercept_sign='non_negative'`, `cores=2`.
* Assert that `model_outputs.objective_weights` is `[0.3, 0.3, 0.4]`.
* Assert that `model_outputs.trials` is a list containing one `Trial` instance.

## `test_select_best_model_equal_metrics(output_models: List[Trial]) -> None`
### USAGE
* This test case checks the behavior of the `_select_best_model` function when all models have equal `nrmse` and `decomp_rssd` values.
* The function should return the `solID` of the first model by default when the models are indistinguishable by their metrics.

### IMPL
* Start by creating a list of `output_models` where each model has the same `nrmse` and `decomp_rssd` values.
* Call the `_select_best_model` method with this list as the input.
* Verify that the returned `solID` matches the expected value, which is the `solID` of the first model in the list (i.e., "model_1").

## `test_select_best_model_distinct_metrics(output_models: List[Trial]) -> None`
### USAGE
* This test case assesses the function's ability to select the model with the lowest combined `nrmse` and `decomp_rssd` score.
* It ensures that the function identifies the model with the optimal balance of metrics.

### IMPL
* Create a list of `output_models` where one model has the lowest combined score derived from `nrmse` and `decomp_rssd`.
* Invoke the `_select_best_model` method with this list.
* Confirm that the returned `solID` is "model_1", which has the lowest combined score.

## `test_select_best_model_negative_metrics(output_models: List[Trial]) -> None`
### USAGE
* This test case evaluates the function's performance when dealing with negative `nrmse` and `decomp_rssd` values.
* It verifies that the function can handle negative numbers and still correctly identify the best model.

### IMPL
* Construct a list of `output_models` with negative `nrmse` and `decomp_rssd` values.
* Execute the `_select_best_model` method using this list.
* Ensure that the `solID` of the model with the lowest combined score is returned, which should be "model_2".

## `test_select_best_model_nan_metrics(output_models: List[Trial]) -> None`
### USAGE
* This test case tests the function's robustness when models contain NaN values for either `nrmse` or `decomp_rssd`.
* The function should skip models with NaN values and select the best of the remaining models.

### IMPL
* Prepare a list of `output_models` where some models contain NaN values for their metrics.
* Call the `_select_best_model` method with this list.
* Verify that the function returns the `solID` for the model with valid, lowest combined metrics, which is "model_3".

## `test_select_best_model_single_model(output_models: List[Trial]) -> None`
### USAGE
* This test case checks the function's output when there is only one model in the input list.
* It ensures that the function can handle a singular model scenario correctly.

### IMPL
* Create a single-element list of `output_models`.
* Invoke the `_select_best_model` method with this single model in the list.
* Confirm that the function returns the `solID` of the lone model, which should be "model_1".

## `test_run_nevergrad_optimization_successful_optimization_with_valid_hyperparameters() -> None`
### USAGE
* This test verifies the successful execution of the `_run_nevergrad_optimization` function with valid hyperparameters.
* It mocks the `ng.optimizers.registry.ask` and `ng.optimizers.registry.tell` methods to simulate the optimization process.
* The input parameters include hyperparameters, iterations, cores, and other relevant configurations.

### IMPL
* Mock the `ng.optimizers.registry.ask` method to simulate the optimizer providing hyperparameters `{'param1': 0.5, 'param2': 1.0}`.
* Mock the `ng.optimizers.registry.tell` method to simulate the optimizer receiving the result with a loss of `0.1`.
* Initialize the necessary input parameters for the test, including `hyper_collect`, `iterations`, `cores`, `nevergrad_algo`, etc., as specified in the test case.
* Call the `_run_nevergrad_optimization` method with the initialized parameters.
* Assert that the `best_result['loss']` from the optimization process equals `0.1`, indicating a successful optimization.
* Assert that the `trial` value returned by the function is `1`, confirming that the correct trial number is maintained.

## `test_run_nevergrad_optimization_optimization_with_zero_penalty_and_objective_weights() -> None`
### USAGE
* This test checks the behavior of the `_run_nevergrad_optimization` function when a zero penalty factor and specific objective weights are provided.
* It mocks the relevant Nevergrad methods to simulate the process.

### IMPL
* Mock the `ng.optimizers.registry.ask` method to simulate providing hyperparameters `{'param1': 0.3, 'param2': 1.8}`.
* Mock the `ng.optimizers.registry.tell` method to simulate receiving a result with a loss of `0.05`.
* Set up the input parameters, including `hyper_collect`, `iterations`, `cores`, `nevergrad_algo`, and other configurations as per the test case.
* Call the `_run_nevergrad_optimization` method.
* Assert that the `best_result['loss']` returned is `0.05`, verifying correct optimization.
* Assert that the `trial` value is `2`, ensuring correctness of trial tracking.

## `test_run_nevergrad_optimization_with_different_algorithms() -> None`
### USAGE
* This test examines the `_run_nevergrad_optimization` function's behavior when using a different Nevergrad algorithm.
* It mocks the appropriate methods to simulate the algorithm's interaction.

### IMPL
* Mock the `ng.optimizers.registry.ask` method to simulate the optimizer providing hyperparameters `{'param1': 0.7, 'param2': 1.5}`.
* Mock the `ng.optimizers.registry.tell` method to simulate receiving a result with a loss of `0.2`.
* Define the input parameters, including `hyper_collect`, `iterations`, `cores`, `nevergrad_algo`, and other configurations as specified in this test case.
* Invoke the `_run_nevergrad_optimization` method with these parameters.
* Assert that the `best_result['loss']` equals `0.2`, confirming that the optimization process is functioning correctly with the new algorithm.
* Confirm that the `trial` value is `3`, validating the trial tracking mechanism.

## `test_calculate_decomp_spend_dist_standard_input() -> None`
### USAGE
* This test function is designed to verify the `_calculate_decomp_spend_dist` method's output when using a standard set of input data.
* It checks if the method returns expected performance metrics in `decomp_spend_dist` DataFrame.

### IMPL
* Instantiate a Ridge model and mock its `coef_` attribute with realistic values.
* Create a DataFrame `X` with specified columns and data, and a Series `y` with given values.
* Define a dictionary `params` with expected metrics values.
* Call `_calculate_decomp_spend_dist` with the Ridge model, DataFrame `X`, Series `y`, and dictionary `params`.
* Assert that the 'rsq_train' value in the result is close to 0.95 using an assertion with a tolerance for approximation.
* Assert that the 'rsq_val' value equals 0.8, as per the `params` dictionary.
* Assert that the 'rsq_test' value equals 0.75, as per the `params` dictionary.
* Assert that the 'lambda' value equals 0.01, as specified in the `params` dictionary.

## `test_calculate_decomp_spend_dist_zero_coefficients() -> None`
### USAGE
* This test function checks the behavior of `_calculate_decomp_spend_dist` when the Ridge model has zero coefficients.
* It ensures that the total contribution and effect share are zero as expected.

### IMPL
* Instantiate a Ridge model and set its `coef_` attribute to zero for all coefficients.
* Create a DataFrame `X` with specified columns and data, and a Series `y` with given values.
* Define the `params` dictionary with relevant metrics.
* Call `_calculate_decomp_spend_dist` with the Ridge model, DataFrame `X`, Series `y`, and dictionary `params`.
* Assert that the sum of 'xDecompAgg' in the result DataFrame is zero.
* Assert that the sum of 'effect_share' in the result DataFrame is zero.

## `test_calculate_decomp_spend_dist_positive_coefficients() -> None`
### USAGE
* This test function evaluates the output of `_calculate_decomp_spend_dist` when the Ridge model has all positive coefficients.
* It confirms that all entries in the 'pos' column are `True`.

### IMPL
* Instantiate a Ridge model and mock its `coef_` attribute with positive values.
* Create a DataFrame `X` with specified columns and data, and a Series `y` with given values.
* Define the `params` dictionary with relevant metrics.
* Call `_calculate_decomp_spend_dist` with the Ridge model, DataFrame `X`, Series `y`, and dictionary `params`.
* Assert that all values in the 'pos' column of the result DataFrame are `True`.

## `test_calculate_decomp_spend_dist_mixed_coefficients() -> None`
### USAGE
* This test function checks the output of `_calculate_decomp_spend_dist` when the Ridge model has mixed coefficients (both positive and negative).
* It ensures that not all entries in the 'pos' column are `True`.

### IMPL
* Instantiate a Ridge model and mock its `coef_` attribute with a mix of positive and negative values.
* Create a DataFrame `X` with specified columns and data, and a Series `y` with given values.
* Define the `params` dictionary with relevant metrics.
* Call `_calculate_decomp_spend_dist` with the Ridge model, DataFrame `X`, Series `y`, and dictionary `params`.
* Assert that not all values in the 'pos' column of the result DataFrame are `True`.

## `test_calculate_decomp_spend_dist_empty_input() -> None`
### USAGE
* This test function verifies the behavior of `_calculate_decomp_spend_dist` when provided with empty input data.
* It checks that the method returns an empty DataFrame.

### IMPL
* Instantiate a Ridge model and mock its `coef_` attribute with any values.
* Create an empty DataFrame `X` and an empty Series `y`.
* Define an empty `params` dictionary.
* Call `_calculate_decomp_spend_dist` with the Ridge model, empty DataFrame `X`, empty Series `y`, and an empty `params` dictionary.
* Assert that the returned DataFrame is empty.

## `test_prepare_data_with_valid_data(params: Dict[str, float]) -> None`
### USAGE
* This test case validates the normal operation of `_prepare_data` when provided with valid data and parameters.
* Parameters: `params` containing keys like `media1_thetas`, `media1_alphas`, and `media1_gammas` with corresponding float values.

### IMPL
* Initialize a `RidgeModelBuilder` instance with mock data to simulate `featurized_mmm_data` and `mmm_data`.
* Invoke the `_prepare_data` method on this instance using the provided `params`.
* Validate that the output `X` is a DataFrame with numeric data types.
* Validate that the output `y` is a Series with NaN values filled appropriately.

## `test_prepare_data_with_nan_infinite(params: Dict[str, float]) -> None`
### USAGE
* This test case checks the handling of NaN and Infinite values in the input data.
* Parameters: `params` containing keys like `media2_thetas` with float values.

### IMPL
* Setup the `RidgeModelBuilder` with data including NaN and Infinite values to simulate `featurized_mmm_data`.
* Call `_prepare_data` with the specified `params`.
* Assert that the resulting `X` DataFrame has no NaN or Infinite values.
* Assert that the resulting `y` Series is free from NaN or Infinite values.

## `test_prepare_data_missing_dependent_var(params: Dict[str, float]) -> None`
### USAGE
* This test case ensures correct handling when the dependent variable column is missing.
* Parameters: `params` containing keys like `media3_alphas` and `media3_gammas` with float values.

### IMPL
* Prepare mock data for `RidgeModelBuilder` where 'dep_var' is missing in `featurized_mmm_data`.
* Execute `_prepare_data` using the provided `params`.
* Verify that `X` does not include any column originally intended as the dependent variable.
* Check that `y` is a Series filled with default values.

## `test_prepare_data_with_categorical_data(params: Dict[str, float]) -> None`
### USAGE
* This test case examines the processing of categorical data requiring one-hot encoding.
* Parameters: `params` containing keys like `media4_thetas` with float values.

### IMPL
* Initialize the `RidgeModelBuilder` with categorical data in `featurized_mmm_data`.
* Invoke `_prepare_data` with the specified `params`.
* Confirm that `X` includes one-hot encoded columns for the categorical data.
* Validate that `y` has NaN values filled accordingly.

## `test_prepare_data_with_date_columns(params: Dict[str, float]) -> None`
### USAGE
* This test case verifies the conversion of date columns to numeric format.
* Parameters: `params` containing keys like `media5_alphas` and `media5_gammas` with float values.

### IMPL
* Set up the `RidgeModelBuilder` with date columns in `featurized_mmm_data`.
* Call `_prepare_data` using the given `params`.
* Ensure that `X` includes date columns converted to numeric format (number of days since earliest date).
* Confirm that `y` is a Series with filled NaN values.

## `test_geometric_adstock_basic_regular_input() -> None`
### USAGE
* Test the `_geometric_adstock` method with a regular input series and a typical theta value to validate its basic functionality.
* Parameters:
  - `x`: a pandas Series representing the input values `[1, 2, 3, 4, 5]`.
  - `theta`: a float representing the adstock decay parameter `0.5`.

### IMPL
* Initialize the input series `x` with values `[1, 2, 3, 4, 5]`.
* Set `theta` to `0.5`.
* Call the `_geometric_adstock` method with the input series `x` and the decay factor `theta`.
* Capture the output series `y` returned by the method.
* Assert that the output series `y` matches the expected series `[1, 2.5, 4.25, 6.125, 8.0625]`.

## `test_geometric_adstock_empty_series() -> None`
### USAGE
* Test the `_geometric_adstock` method with an empty series to ensure it handles edge cases gracefully.
* Parameters:
  - `x`: a pandas Series representing an empty input.
  - `theta`: a float representing the adstock decay parameter `0.5`.

### IMPL
* Initialize the input series `x` as an empty pandas Series.
* Set `theta` to `0.5`.
* Call the `_geometric_adstock` method with the empty series `x` and the decay factor `theta`.
* Capture the output series `y` returned by the method.
* Assert that the output series `y` is an empty series, matching the expected empty output.

## `test_geometric_adstock_theta_zero() -> None`
### USAGE
* Test the `_geometric_adstock` method with a regular input series and a theta value of zero to check if the function returns the input unchanged.
* Parameters:
  - `x`: a pandas Series representing the input values `[1, 2, 3, 4, 5]`.
  - `theta`: a float representing the adstock decay parameter `0.0`.

### IMPL
* Initialize the input series `x` with values `[1, 2, 3, 4, 5]`.
* Set `theta` to `0.0`.
* Call the `_geometric_adstock` method with the input series `x` and the decay factor `theta`.
* Capture the output series `y` returned by the method.
* Assert that the output series `y` matches the input series `[1, 2, 3, 4, 5]`, indicating no change due to zero theta.

## `test_geometric_adstock_theta_one() -> None`
### USAGE
* Test the `_geometric_adstock` method with a regular input series and a theta value of one to ensure the cumulative sum is correctly computed.
* Parameters:
  - `x`: a pandas Series representing the input values `[1, 2, 3, 4, 5]`.
  - `theta`: a float representing the adstock decay parameter `1.0`.

### IMPL
* Initialize the input series `x` with values `[1, 2, 3, 4, 5]`.
* Set `theta` to `1.0`.
* Call the `_geometric_adstock` method with the input series `x` and the decay factor `theta`.
* Capture the output series `y` returned by the method.
* Assert that the output series `y` matches the expected cumulative series `[1, 3, 6, 10, 15]`.

## `test_geometric_adstock_negative_values() -> None`
### USAGE
* Test the `_geometric_adstock` method with negative values in the input series to verify correct handling of negative numbers.
* Parameters:
  - `x`: a pandas Series representing the input values `[-1, -2, -3, -4, -5]`.
  - `theta`: a float representing the adstock decay parameter `0.5`.

### IMPL
* Initialize the input series `x` with values `[-1, -2, -3, -4, -5]`.
* Set `theta` to `0.5`.
* Call the `_geometric_adstock` method with the input series `x` and the decay factor `theta`.
* Capture the output series `y` returned by the method.
* Assert that the output series `y` matches the expected series `[-1, -2.5, -4.25, -6.125, -8.0625]`.

## `test_geometric_adstock_large_theta() -> None`
### USAGE
* Test the `_geometric_adstock` method with a large theta value to observe its impact on the series.
* Parameters:
  - `x`: a pandas Series representing the input values `[1, 2, 3, 4, 5]`.
  - `theta`: a float representing the adstock decay parameter `10.0`.

### IMPL
* Initialize the input series `x` with values `[1, 2, 3, 4, 5]`.
* Set `theta` to `10.0`.
* Call the `_geometric_adstock` method with the input series `x` and the decay factor `theta`.
* Capture the output series `y` returned by the method.
* Assert that the output series `y` closely approximates `[1, 12, 123, 1234, 12345]`, reflecting a significant impact of the large theta.

## `test_geometric_adstock_small_theta() -> None`
### USAGE
* Test the `_geometric_adstock` method with a very small theta value close to zero to check precision handling.
* Parameters:
  - `x`: a pandas Series representing the input values `[1, 2, 3, 4, 5]`.
  - `theta`: a float representing the adstock decay parameter `1e-09`.

### IMPL
* Initialize the input series `x` with values `[1, 2, 3, 4, 5]`.
* Set `theta` to a very small value `1e-09`.
* Call the `_geometric_adstock` method with the input series `x` and the decay factor `theta`.
* Capture the output series `y` returned by the method.
* Assert that the output series `y` closely matches `[1, 2.000000001, 3.000000002, 4.000000003, 5.000000004]`, ensuring precision with minimal theta impact.

## `test_hill_transformation_typical_values(x: pd.Series, alpha: float, gamma: float) -> None`
### USAGE
* This test verifies the `_hill_transformation` function with typical values for `alpha` and `gamma`.
* Parameters:
  - `x`: A pandas Series containing the input values `[0, 0.5, 1]`.
  - `alpha`: A float representing the slope parameter, set to `2`.
  - `gamma`: A float representing the asymptote parameter, set to `1`.
### IMPL
* Instantiate the `RidgeModelBuilder` class as it contains the method to be tested.
* Call the method `_hill_transformation` on the instance with the provided Series `x`, and parameters `alpha` and `gamma`.
* Capture the output, which should be a transformed pandas Series.
* Assert that the transformed values match the expected transformed values Series, ensuring the function correctly applies the hill transformation with typical parameters.

## `test_hill_transformation_zero_values(x: pd.Series, alpha: float, gamma: float) -> None`
### USAGE
* This test checks the `_hill_transformation` function when both `alpha` and `gamma` are zero.
* Parameters:
  - `x`: A pandas Series containing the input values `[0, 0.5, 1]`.
  - `alpha`: A float set to `0`.
  - `gamma`: A float set to `0`.
### IMPL
* Instantiate the `RidgeModelBuilder` class for testing purposes.
* Call the `_hill_transformation` method with the input Series `x`, and parameters `alpha` and `gamma`.
* Capture the output Series from the method.
* Verify that all values in the transformed Series are `0.5`, as expected when both parameters are zero, indicating a neutral transformation.

## `test_hill_transformation_negative_values(x: pd.Series, alpha: float, gamma: float) -> None`
### USAGE
* This test ensures the `_hill_transformation` function handles negative `alpha` and `gamma` values correctly.
* Parameters:
  - `x`: A pandas Series containing the input values `[0, 0.5, 1]`.
  - `alpha`: A float set to `-1`.
  - `gamma`: A float set to `-1`.
### IMPL
* Create an instance of `RidgeModelBuilder`.
* Use the `_hill_transformation` method with the given Series `x`, and the negative parameters `alpha` and `gamma`.
* Capture the output Series.
* Assert that the resulting values are negative, reflecting the influence of negative parameters on the transformation process.

## `test_hill_transformation_large_series(x: pd.Series, alpha: float, gamma: float) -> None`
### USAGE
* This test assesses the performance and accuracy of `_hill_transformation` with a large input Series.
* Parameters:
  - `x`: A pandas Series containing `10000` values evenly spaced between `0` and `1`.
  - `alpha`: A float set to `1.5`.
  - `gamma`: A float set to `0.5`.
### IMPL
* Instantiate the `RidgeModelBuilder` class.
* Invoke the `_hill_transformation` method with the large Series `x`, along with `alpha` and `gamma`.
* Capture the transformed Series output.
* Verify that the transformed values align with expected results for a large dataset, confirming the method's scalability and correctness.

## `test_hill_transformation_uniform_values(x: pd.Series, alpha: float, gamma: float) -> None`
### USAGE
* This test evaluates `_hill_transformation` when all `x` values are identical.
* Parameters:
  - `x`: A pandas Series with all values as `0.5`.
  - `alpha`: A float set to `2`.
  - `gamma`: A float set to `1`.
### IMPL
* Instantiate the `RidgeModelBuilder`.
* Call the `_hill_transformation` method with the uniform Series `x`, and parameters `alpha` and `gamma`.
* Capture the resulting Series.
* Assert that all transformed values are the same, demonstrating consistent transformation across uniform input values.

## `test_evaluate_model_with_time_series_validation_and_positive_objective_weights() -> None`
### USAGE
* This unit test case is designed to evaluate the `_evaluate_model` function when time series validation is enabled and objective weights are positive.
* It ensures that the model processes the data correctly, calculates metrics accurately, and returns expected results.

### IMPL
* Begin by mocking the `_prepare_data` method of `RidgeModelBuilder` to return a DataFrame with features and a Series with the target based on the input parameters `train_size: 0.8` and `lambda: 0.5`.
* Mock the `_calculate_rssd` method to return `0.5` when called with coefficients array and `True` for RSSD zero penalty.
* Mock the `calibrate` method of `MediaEffectCalibrator` to return an object with a `get_mean_mape` method that returns `0.05`.
* Mock the `_calculate_decomp_spend_dist` method to return a DataFrame representing the decomposition spend distribution.
* Mock the `_calculate_x_decomp_agg` method to return aggregated decomposition values.
* Prepare test input parameters: `params`, `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, `trial`.
* Call the `_evaluate_model` function with the prepared parameters.
* Assert that the `loss` in the result matches the expected value of `0.2`.
* Assert that the `params["rsq_train"]`, `params["rsq_val"]`, `params["rsq_test"]`, `params["nrmse_train"]`, `params["nrmse_val"]`, `params["nrmse_test"]`, and `params["nrmse"]` are all of type `float`.
* Assert that `params["decomp.rssd"]` equals `0.5`.
* Assert that `params["mape"]` equals `0.05`.
* Assert that `params["lambda"]` equals `0.5`.
* Assert that `params["solID"]` equals `'1_2_1'`.
* Assert that `params["trial"]` equals `1`.
* Assert that `params["iterNG"]` equals `2`.
* Assert that `params["iterPar"]` equals `1`.
* Assert that `params["train_size"]` equals `0.8`.
* Assert that `decomp_spend_dist` in the result matches the mocked DataFrame with decomposition spend distribution.
* Assert that `x_decomp_agg` in the result matches the mocked aggregated decomposition values.

## `test_evaluate_model_without_time_series_validation_and_negative_objective_weights() -> None`
### USAGE
* This unit test case is designed to evaluate the `_evaluate_model` function when time series validation is disabled and objective weights are negative.
* It ensures that the model handles the absence of validation correctly and calculates the metrics and loss accurately.

### IMPL
* Mock the `_prepare_data` method of `RidgeModelBuilder` to return a DataFrame with features and a Series with the target based on the input parameters `train_size: 1.0` and `lambda: 0.2`.
* Mock the `_calculate_rssd` method to return `0.7` when called with coefficients array and `False` for RSSD zero penalty.
* Prepare test input parameters: `params`, `ts_validation`, `add_penalty_factor`, `rssd_zero_penalty`, `objective_weights`, `start_time`, `iter_ng`, `trial`.
* Call the `_evaluate_model` function with the prepared parameters.
* Assert that the `loss` in the result matches the expected value of `-0.35`.
* Assert that the `params["rsq_train"]`, `params["nrmse_train"]`, and `params["nrmse"]` are all of type `float`.
* Assert that `params["rsq_val"]` and `params["rsq_test"]` equal `0.0`.
* Assert that `params["nrmse_val"]` and `params["nrmse_test"]` equal `0.0`.
* Assert that `params["decomp.rssd"]` equals `0.7`.
* Assert that `params["mape"]` equals `0.0`.
* Assert that `params["lambda"]` equals `0.2`.
* Assert that `params["solID"]` equals `'2_3_1'`.
* Assert that `params["trial"]` equals `2`.
* Assert that `params["iterNG"]` equals `3`.
* Assert that `params["iterPar"]` equals `1`.
* Assert that `params["train_size"]` equals `1.0`.

## `test_hyper_collector_all_parameters_fixed() -> None`
### USAGE
* This test case is designed to validate the behavior of the `_hyper_collector` function when all hyperparameters are fixed, and no optimization is required.
* Inputs include a dictionary of hyperparameters with all parameters fixed, a flag for time series validation, and other relevant parameters.

### IMPL
* Prepare the input dictionary `hyperparameters_dict` containing `prepared_hyperparameters` where each hyperparameter is set with a specific value, and `hyper_to_optimize` is an empty list.
* Call the `_hyper_collector` function with the input dictionary, `ts_validation` set to `False`, `add_penalty_factor` set to `False`, `dt_hyper_fixed` as a DataFrame with specific fixed values, and `cores` set to 4.
* Ensure that no dependencies need to be mocked as there are no dependencies specified in the test case.
* Assert that the `hyper_list_all` result matches the expected dictionary of all fixed hyperparameters.
* Check that the `hyper_bound_list_updated` is an empty list, indicating no parameters are set for optimization.
* Verify that `hyper_bound_list_fixed` contains all the fixed hyperparameters as expected, including lambda and train_size.
* Confirm that `all_fixed` is `True`, indicating that all hyperparameters were indeed fixed.

## `test_hyper_collector_some_parameters_to_optimize() -> None`
### USAGE
* This test case aims to validate the function's ability to correctly identify and separate fixed and optimizable hyperparameters when some parameters are set for optimization.
* Inputs consist of a hyperparameters dictionary with some None values indicating parameters to be optimized.

### IMPL
* Construct the input dictionary `hyperparameters_dict` with `prepared_hyperparameters` having some hyperparameters set to `None` to specify optimization, and a non-empty `hyper_to_optimize` list.
* Invoke the `_hyper_collector` function with the dictionary, `ts_validation` set to `True`, `add_penalty_factor` set to `True`, with `dt_hyper_fixed` as `None`, and `cores` set to 2.
* No mocks are necessary for this test case as no external dependencies are involved.
* Assert that the `hyper_list_all` result matches the expected dictionary where some parameters are `None`.
* Verify that `hyper_bound_list_updated` contains the list of parameters that are set for optimization.
* Validate that `hyper_bound_list_fixed` includes only those hyperparameters that are fixed and not set for optimization.
* Confirm `all_fixed` is `False`, as some hyperparameters are designated for optimization.

## `test_hyper_collector_no_fixed_hyper_and_missing_params() -> None`
### USAGE
* This test case checks the behavior of the `_hyper_collector` function when there are no fixed hyperparameters and some parameters are missing, marked for optimization.
* Inputs include a hyperparameters dictionary with missing values indicating parameters for optimization.

### IMPL
* Create the input dictionary `hyperparameters_dict` where `prepared_hyperparameters` contains missing hyperparameters denoted by `None`, and `hyper_to_optimize` lists these missing parameters.
* Execute the `_hyper_collector` function with the prepared dictionary, `ts_validation` set to `False`, `add_penalty_factor` set to `True`, `dt_hyper_fixed` as `None`, and `cores` set to 1.
* As the test does not require dependency interaction, no mocks are needed.
* Assert that the `hyper_list_all` correctly represents the hyperparameters, showing `None` for those to be optimized.
* Confirm that `hyper_bound_list_updated` includes all the parameters that need optimization.
* Check that `hyper_bound_list_fixed` only contains the hyperparameters that are specified and not set for optimization.
* Verify that `all_fixed` is `False`, given that optimization is required for some parameters.

## `test_model_refit_with_minimum_input() -> None`
### USAGE
* This test case is designed to verify the behavior of the `_model_refit` function when only the training data is provided.
* It will test the function with minimal required inputs, focusing on ensuring that the function can handle the absence of validation and test data.

### IMPL
* Begin by setting up the test data:
  * Create a 2D numpy array `x_train` with a shape of (10, 5) filled with random numbers.
  * Create a 1D numpy array `y_train` with a shape of (10,) filled with random numbers.
* Call the `_model_refit` function with `x_train`, `y_train`, and default values for other parameters.
* Capture the output in a variable, say `output`.
* Perform assertions to verify the results:
  * Assert that `output.rsq_train` is a float between 0 and 1.
  * Assert that `output.rsq_val` is None.
  * Assert that `output.rsq_test` is None.
  * Assert that `output.nrmse_train` is a float greater than or equal to 0.
  * Assert that `output.nrmse_val` is None.
  * Assert that `output.nrmse_test` is None.
  * Assert that `output.coefs` is a 1D array of shape (5,).
  * Assert that `output.y_train_pred` is a 1D array of shape (10,).
  * Assert that `output.y_val_pred` is None.
  * Assert that `output.y_test_pred` is None.
  * Assert that `output.y_pred` is a 1D array of shape (10,).
  * Assert that `output.mod` is an instance of Ridge with `alpha=1.0`.
  * Assert that `output.df_int` equals 1.

## `test_model_refit_with_full_data() -> None`
### USAGE
* This test case checks the `_model_refit` function's performance when provided with full datasets, including training, validation, and test sets.
* It ensures that the function can handle and return expected results for all types of inputs.

### IMPL
* Set up the test data:
  * Create `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test` as numpy arrays with specified shapes filled with random numbers.
* Call the `_model_refit` function with all the above datasets and capture the output.
* Perform assertions:
  * Assert that `output.rsq_train`, `output.rsq_val`, and `output.rsq_test` are floats between 0 and 1.
  * Assert that `output.nrmse_train`, `output.nrmse_val`, and `output.nrmse_test` are floats greater than or equal to 0.
  * Assert that `output.coefs` is a 1D array of shape (5,).
  * Assert that `output.y_train_pred` is a 1D array of shape (10,).
  * Assert that `output.y_val_pred` and `output.y_test_pred` are 1D arrays of shape (5,).
  * Assert that `output.y_pred` is a 1D array of shape (20,).
  * Assert that `output.mod` is an instance of Ridge with `alpha=1.0`.
  * Assert that `output.df_int` equals 1.

## `test_model_refit_without_intercept() -> None`
### USAGE
* This test verifies the `_model_refit` function when the intercept is not included in the model.
* It checks if the function correctly handles and returns expected results when intercept is set to False.

### IMPL
* Prepare the datasets similar to the full data test case.
* Call the `_model_refit` function with the intercept parameter set to False.
* Capture the function output.
* Perform assertions:
  * Assert that `output.rsq_train`, `output.rsq_val`, and `output.rsq_test` are floats between 0 and 1.
  * Assert that `output.nrmse_train`, `output.nrmse_val`, and `output.nrmse_test` are floats greater than or equal to 0.
  * Assert that `output.coefs` is a 1D array of shape (5,).
  * Assert that `output.y_train_pred` is a 1D array of shape (10,).
  * Assert that `output.y_val_pred` and `output.y_test_pred` are 1D arrays of shape (5,).
  * Assert that `output.y_pred` is a 1D array of shape (20,).
  * Assert that `output.mod` is an instance of Ridge with `alpha=1.0`.
  * Assert that `output.df_int` equals 0.

## `test_model_refit_with_zero_lambda() -> None`
### USAGE
* This test case checks the behavior of the `_model_refit` function when the regularization parameter (lambda) is set to 0, indicating no regularization.
* It ensures that the function handles this scenario appropriately.

### IMPL
* Prepare the datasets similar to the full data test case.
* Call the `_model_refit` function with `lambda_` set to 0.0.
* Capture the output.
* Perform assertions:
  * Assert that `output.rsq_train`, `output.rsq_val`, and `output.rsq_test` are floats between 0 and 1.
  * Assert that `output.nrmse_train`, `output.nrmse_val`, and `output.nrmse_test` are floats greater than or equal to 0.
  * Assert that `output.coefs` is a 1D array of shape (5,).
  * Assert that `output.y_train_pred` is a 1D array of shape (10,).
  * Assert that `output.y_val_pred` and `output.y_test_pred` are 1D arrays of shape (5,).
  * Assert that `output.y_pred` is a 1D array of shape (20,).
  * Assert that `output.mod` is an instance of Ridge with `alpha=0.0`.
  * Assert that `output.df_int` equals 1.

## `test_model_refit_with_coefficient_limits() -> None`
### USAGE
* This test checks the functionality of the `_model_refit` function when lower and upper limits for coefficients are provided.
* It validates that the function respects these limits during model fitting.

### IMPL
* Prepare the datasets similar to the full data test case.
* Define `lower_limits` and `upper_limits` as lists of 5 floats, e.g., [-1, -1, -1, -1, -1] and [1, 1, 1, 1, 1] respectively.
* Call the `_model_refit` function with these limits.
* Capture the output.
* Perform assertions:
  * Assert that `output.rsq_train`, `output.rsq_val`, and `output.rsq_test` are floats between 0 and 1.
  * Assert that `output.nrmse_train`, `output.nrmse_val`, and `output.nrmse_test` are floats greater than or equal to 0.
  * Assert that `output.coefs` is a 1D array of shape (5,) with values between the specified lower and upper limits.
  * Assert that `output.y_train_pred` is a 1D array of shape (10,).
  * Assert that `output.y_val_pred` and `output.y_test_pred` are 1D arrays of shape (5,).
  * Assert that `output.y_pred` is a 1D array of shape (20,).
  * Assert that `output.mod` is an instance of Ridge with `alpha=1.0`.
  * Assert that `output.df_int` equals 1.

## `test_lambda_seq_with_small_dataset() -> None`
### USAGE
* Test the `_lambda_seq` function with a small dataset to verify its correctness.
### IMPL
* Prepare the input data by creating an array `x` of shape `(5, 2)` with random values and an array `y` of shape `(5,)` with random values.
* Call the `_lambda_seq` function with `x`, `y`, `seq_len=10`, and `lambda_min_ratio=0.0001`.
* Assert that the length of the result is 10 to ensure the sequence is generated correctly.
* Assert that the first element of the result is greater than the last element to verify the sequence is in descending order.
* Assert that each element in the result is greater than or equal to the subsequent element to confirm all values are in non-increasing order.

## `test_lambda_seq_with_zero_feature_data() -> None`
### USAGE
* Test the `_lambda_seq` function with feature data containing all zero values.
### IMPL
* Prepare the input data by creating an array `x` of shape `(5, 2)` with all zero values and an array `y` of shape `(5,)` with random values.
* Call the `_lambda_seq` function with `x`, `y`, `seq_len=5`, and `lambda_min_ratio=0.0001`.
* Assert that all values in the result are zero to confirm the function handles zero feature data correctly.

## `test_lambda_seq_with_single_element_data() -> None`
### USAGE
* Test the `_lambda_seq` function with single element feature data.
### IMPL
* Prepare the input data by creating an array `x` of shape `(1, 1)` with a single random value and an array `y` of shape `(1,)` with a single random value.
* Call the `_lambda_seq` function with `x`, `y`, `seq_len=5`, and `lambda_min_ratio=0.0001`.
* Assert that the length of the result is 5 to ensure the sequence is generated correctly.
* Assert that the first element of the result is greater than the last element to verify the sequence is in descending order.
* Assert that all values in the result are greater than or equal to 0 to ensure non-negative values in the sequence.

## `test_lambda_seq_with_negative_values() -> None`
### USAGE
* Test the `_lambda_seq` function with feature data containing negative values.
### IMPL
* Prepare the input data by creating an array `x` of shape `(5, 2)` with negative random values and an array `y` of shape `(5,)` with random values.
* Call the `_lambda_seq` function with `x`, `y`, `seq_len=15`, and `lambda_min_ratio=0.0001`.
* Assert that the length of the result is 15 to ensure the sequence is generated correctly.
* Assert that the first element of the result is greater than the last element to verify the sequence is in descending order.
* Assert that all values in the result are greater than or equal to 0 to ensure non-negative values in the sequence.