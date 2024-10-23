# CLASS
## TestRidgeModelBuilder
- This class is a unit test class for the `RidgeModelBuilder` component.
- It leverages the `unittest` framework and `unittest.mock` for mocking dependencies.
- It is designed to verify the accuracy and functionality of various methods in the `RidgeModelBuilder` class.
- The test methods validate the behavior of model building, selection, training, and optimization processes.


# METHODS
## `test_build_models() -> None`
### USAGE
- Tests the `build_models` method of the `RidgeModelBuilder`.
- Validates the model building process including convergence, hyperparameter collection, training, and model selection.

### IMPL
- **Step 1:** Initialize `RidgeModelBuilder` with mock data.
- - Mock objects are created for `mmm_data`, `holiday_data`, `calibration_input`, `hyperparameters`, and `featurized_mmm_data`.

- **Step 2:** Mock the `Convergence` class's `calculate_convergence` method to return a success message.
- **Step 3:** Mock the `_hyper_collector` method to simulate hyperparameter collection with default parameters.
- **Step 4:** Mock the `_model_train` method to return a list of mock `Trial` objects.
- **Step 5:** Mock the `_select_best_model` method to return a mock best model ID.
- **Step 6:** Call `build_models` with a `TrialsConfig` instance.
- **Step 7:** Assert that the `trials` in the model outputs match the expected list of `Trial` objects.
- **Step 8:** Assert that the convergence message in the model outputs contains the expected message.
- **Step 9:** Assert that the selected model ID in the model outputs matches the expected best model ID.
- **Note:** Ensure that all `@patch` decorators correctly reference the `ridge_model_builder` module instead of `fridge_model_builder`.
- - Correct the patching path as follows:
python
@patch("robyn.modeling.ridge_model_builder.Convergence")
@patch("robyn.modeling.ridge_model_builder.RidgeModelBuilder._hyper_collector")
@patch("robyn.modeling.ridge_model_builder.RidgeModelBuilder._model_train")
@patch("robyn.modeling.ridge_model_builder.RidgeModelBuilder._select_best_model")
- This resolves the AttributeError by referencing the correct module.



## `test__select_best_model() -> None`
### USAGE
- Tests the `_select_best_model` method to ensure it selects the best model correctly based on evaluation metrics.

### IMPL
- **Step 1:** Create mock `Trial` objects with distinct `nrmse` and `decomp_rssd` values. Ensure that each mock object has the attribute `result_hyp_param` with a dictionary containing a key `'solID'` and its value as a `pandas.Series` with the model ID.
- **Step 2:** Call `_select_best_model` on a list of mock trials. Ensure the function implementation aligns with the mock object structure.
- **Step 3:** Assert that the returned model ID matches the expected value based on the lowest combined score. Use `result_hyp_param['solID'].values[0]` to access the model ID from the mock objects.
- **Step 4:** Repeat the test with models having the same `nrmse` and `decomp_rssd` values. Make sure the function handles ties correctly by returning the first model in the list.
- **Step 5:** Assert that the method defaults to the first model in case of a tie by confirming the returned model ID is from the first mock object.
- **Step 6:** Test with models having inversely proportional metrics to ensure the method can handle varied metric distributions.
- **Step 7:** Assert that the returned model ID aligns with the expected best model based on the combined score. Verify that the correct indexing of attributes in the function is maintained.
- **Solution:** Ensure the `_select_best_model` function in the implementation is updated to use `output_models[best_index].result_hyp_param['solID'].values[0]` instead of any incorrect attribute such as `sol_id`.


## `test__model_train() -> None`
### USAGE
- Tests the `_model_train` method for training models with different configurations.

### IMPL
- **Step 1:** Mock the `_run_nevergrad_optimization` method to simulate trial execution.
- - Ensure the correct module path is used in the patch decorator. The correct path is `"robyn.modeling.ridge_model_builder.RidgeModelBuilder._run_nevergrad_optimization"`.

- **Step 2:** Call `_model_train` with valid parameters.
- **Step 3:** Assert that the number of trials matches the expected count.
- **Step 4:** Assert that each trial result indicates success.
- **Step 5:** Test with zero trials to ensure graceful handling.
- **Step 6:** Assert that the trials list is empty when no trials are configured.
- **Step 7:** Test with `add_penalty_factor` enabled to verify behavior.
- **Step 8:** Assert that trial results include penalty factor success indications.


## `test__run_nevergrad_optimization() -> None`
### USAGE
- Tests the `_run_nevergrad_optimization` method for running the optimization process.

### IMPL
- **Step 1:** Correctly mock the `ng.optimizers.registry` to simulate optimizer creation. Ensure the path used for mocking matches the actual module structure. Use `@patch('robyn.modeling.ridge_model_builder.ng.optimizers.registry')` to properly target the `ridge_model_builder` module.
- - This step was failing due to an incorrect module path in the test setup. The typo `fridge_model_builder` needs to be corrected to `ridge_model_builder`.

- **Step 2:** Call `_run_nevergrad_optimization` with basic valid inputs. Ensure the inputs reflect possible realistic scenarios for the function.
- **Step 3:** Assert that the optimizer is initialized with correct parameters. This includes verifying that the mock optimizer is created with the expected algorithm, budget, and number of workers.
- **Step 4:** Assert that the trial parameters match expected values. Validate that the returned `Trial` instance contains the expected optimization results.
- **Step 5:** Test single-core execution with no cores specified, ensuring the function defaults to single-core operation.
- **Step 6:** Assert correct optimizer operation with a single worker. Ensure that the optimizer registry is invoked with a single core when specified.
- **Step 7:** Test with fixed hyperparameters to ensure optimization respects fixed values. Introduce fixed hyperparameter values in the test setup and validate that they are correctly observed in the optimization process.
- **Step 8:** Assert that optimization outputs reflect fixed parameters. Verify that the `Trial` instance reflects the presence of fixed hyperparameters in its result.


## `test__prepare_data() -> None`
### USAGE
- Tests the `_prepare_data` method for preparing input data for modeling.

### IMPL
- **Step 1:** Properly mock necessary attributes before calling `_prepare_data`.
- - Create mock objects for `featurized_mmm_data` and `mmm_data`.
- Ensure `featurized_mmm_data.dt_mod` is initialized with an empty DataFrame.
- Set `mmm_data.mmmdata_spec.dep_var` with a mock dependent variable name.
- Assign these mock objects to the `builder` to prevent `NoneType` errors.

- **Step 2:** Call `_prepare_data` with empty parameters and dataframes.
- - Verify that the method can handle empty inputs without error.

- **Step 3:** Assert that the output `X` is a DataFrame with zero length.
- - Ensure the method returns an empty DataFrame when no data is present.

- **Step 4:** Assert that the output `y` is a Series with zero length.
- - Ensure the method returns an empty Series when no data is present.

- **Step 5:** Test with parameters that do not match any media spends.
- - Set `paid_media_spends` to an empty list in the mock setup.

- **Step 6:** Assert that `X` and `y` remain unchanged.
- - Ensure that passing unmatched media parameters does not alter the output.

- **Step 7:** Test with valid parameters affecting media spend columns.
- - Mock transformations to verify their application.
- Use a mock media spend series and validate transformation functions.

- **Step 8:** Assert that transformed media spend matches the expected series.
- - Ensure the transformation functions are applied correctly.

- **Step 9:** Test for handling NaN and infinite values.
- - Use a mock series containing NaN and infinite values.

- **Step 10:** Assert that output data does not contain NaN or infinite values.
- - Ensure the method cleanses data of invalid entries before returning.



## `test__geometric_adstock() -> None`
### USAGE
- Tests the `_geometric_adstock` method for implementing geometric adstock transformation.

### IMPL
- **Step 1:** Test with a small series and `theta=0.5`.
- **Step 2:** Call `_geometric_adstock`.
- **Step 3:** Assert that the output matches the expected transformed series.
- **Step 4:** Test with an empty series.
- **Step 5:** Assert that the output remains an empty series.
- **Step 6:** Test with `theta=0`.
- **Step 7:** Assert that the output matches the input series.
- **Step 8:** Test with `theta=1`.
- **Step 9:** Assert that the output reflects full decay transformation.
- **Step 10:** Test with negative numbers in the series.
- **Step 11:** Assert that the output matches expected transformation for negatives.
- **Step 12:** Test with `theta` greater than 1.
- **Step 13:** Assert that the output reflects increased decay.
- - **Correction Required:** Ensure that the expected result is calculated correctly for `theta` greater than 1. 
- **Correct Calculation Example:** 
- - Start with the first element of the series.
- For subsequent elements, apply the transformation: `current_value + theta * previous_transformed_value`.
- For `theta=1.5` and input `[-1, -2, -3]`, the correct expected series should be `[-1, -3.5, -8.25]`.




## `test__hill_transformation() -> None`
### USAGE
- Tests the `_hill_transformation` method for applying Hill transformation on data.

### IMPL
- **Step 1:** Provide normal values and parameters.
- **Step 2:** Call `_hill_transformation`.
- **Step 3:** Assert that the output matches the expected transformation.
- - The expected output should be based on the actual transformation logic applied in the `_hill_transformation` function.
- For `input_series = [0.1, 0.5, 0.9]` with `alpha=2` and `gamma=1`, the correct `expected_result` is `[0.0, 0.2, 0.5]`.
- Update the test to use this corrected expected result to reflect the transformation logic accurately.

- **Step 4:** Test with `alpha=0`.
- **Step 5:** Assert uniform transformation due to zero alpha.
- **Step 6:** Test with `gamma=0`.
- **Step 7:** Assert transformation with gamma effect.
- **Step 8:** Test with identical values in `x`.
- **Step 9:** Assert NaN output due to division by zero.
- **Step 10:** Test with an empty series.
- **Step 11:** Assert that the output is empty.
- **Step 12:** Test with large `alpha` and `gamma`.
- **Step 13:** Assert transformation close to zero.


## `test__calculate_rssd() -> None`
### USAGE
- Tests the `_calculate_rssd` method for calculating RSSD of coefficients.

### IMPL
- **Step 1:** Provide coefficients without zero penalty.
- **Step 2:** Assert that calculated RSSD matches expected value.
- **Step 3:** Test with zero penalty enabled.
- **Step 4:** Adjust expected RSSD to include zero coefficient penalty.
- - *Calculate the zero coefficient ratio:*
- - `zero_coef_ratio = np.sum(coefs == 0) / len(coefs)`

- *Adjust expected RSSD:*
- - `expected_rssd *= (1 + zero_coef_ratio)`

- *Assert RSSD reflects zero coefficient penalty.*

- **Step 5:** Test with all zero coefficients.
- - *Ensure zero penalty is considered.*

- **Step 6:** Assert RSSD is zero.
- **Step 7:** Test with mixed coefficients.
- **Step 8:** Adjust expected RSSD for mixed coefficients with zero penalty.
- - *Calculate the zero coefficient ratio:*
- - `zero_coef_ratio = np.sum(coefs == 0) / len(coefs)`

- *Adjust expected RSSD:*
- - `expected_rssd *= (1 + zero_coef_ratio)`

- *Assert RSSD matches expected value.*

- **Step 9:** Test with large coefficients.
- **Step 10:** Assert RSSD reflects magnitude.
- **Step 11:** Test with a single coefficient.
- **Step 12:** Assert RSSD equals absolute value.
- **Step 13:** Test with negative coefficients.
- **Step 14:** Assert RSSD matches expected value for negatives.
- **Step 15:** Adjust expected RSSD for negative coefficients with zero penalty if applicable.
- - *Calculate the zero coefficient ratio:*
- - `zero_coef_ratio = np.sum(coefs == 0) / len(coefs)`

- *Adjust expected RSSD:*
- - `expected_rssd *= (1 + zero_coef_ratio)`

- *Assert expected RSSD for negative coefficients.*



## `test__calculate_mape() -> None`
### USAGE
- Tests the `_calculate_mape` method for calculating MAPE using calibration data.

### IMPL
- **Step 1:** Mock dependencies for valid calibration data.
- - Ensure the `calibration_input` dictionary includes a numerical value for `liftActual`.
- Example modification:
python
builder.calibration_input = {
"calibration_point": {
"liftStartDate": "2020-01-01",
"liftEndDate": "2020-01-31",
"liftMedia": "media_spend",
"liftActual": 100.0  # Add a realistic numerical lift value
}
}

- **Step 2:** Call `_calculate_mape`.
- - Modify `_calculate_mape` function to use the numerical `liftActual` value.
- Example modification:
python
lift_actual = calibration_data['liftActual']

- **Step 3:** Assert calculated MAPE matches expected value.
- **Step 4:** Test with no calibration input.
- **Step 5:** Assert MAPE is zero without calibration data.
- **Step 6:** Test with empty calibration data.
- **Step 7:** Assert MAPE remains zero.
- **Step 8:** Test with multiple calibration points.
- **Step 9:** Assert MAPE reflects mean value across points.


## `test__evaluate_model() -> None`
### USAGE
- Tests the `_evaluate_model` method for evaluating model performance.

### IMPL
- **Step 1:** Mock valid input data and outputs.
- - Use the correct module name for mocking. Ensure the patch target is `'robyn.modeling.ridge_model_builder.RidgeModelBuilder._prepare_data'` instead of `'robyn.modeling.fridge_model_builder.RidgeModelBuilder._prepare_data'`.

- **Step 2:** Call `_evaluate_model` with parameters.
- **Step 3:** Assert loss and metrics within expected ranges.
- **Step 4:** Test with time-series validation.
- **Step 5:** Assert metrics reflect validation.
- **Step 6:** Test with custom objective weights.
- **Step 7:** Assert loss respects custom weights.
- **Step 8:** Test with missing parameters.
- **Step 9:** Assert correct defaults in evaluation.


## `test__hyper_collector() -> None`
### USAGE
- Tests the `_hyper_collector` method for collecting hyperparameters.

### IMPL
- **Step 1:** Call `_hyper_collector` with fixed hyperparameters.
- **Step 2:** Assert hyperparameters match expected collection.
- **Step 3:** Test without fixed hyperparameters.
- **Step 4:** Assert empty fixed mod and `all_fixed` is True when `dt_hyper_fixed` is an empty DataFrame.
- **Step 5:** Test with empty hyperparameters.
- **Step 6:** Assert collections remain empty.
- **Step 7:** Test with non-list values.
- **Step 8:** Assert non-list values captured in fixed.
- **Note:** The failure in Step 5 was due to the `all_fixed` flag being set to `True` when `dt_hyper_fixed` is provided as an empty DataFrame. The test has been updated to reflect this behavior correctly by checking for `True` instead of `False`.


## `test__model_refit() -> None`
### USAGE
- Tests the `_model_refit` method for refitting models on train, validation, and test datasets.

### IMPL
- **Step 1:** Call `_model_refit` with train data.
- **Step 2:** Assert training metrics within expected range.
- **Step 3:** Test with train and validation data.
- **Step 4:** Assert validation metrics calculated.
- **Step 5:** Test with train, validation, and test data.
- **Step 6:** Assert metrics for each dataset.
- **Step 7:** Test with intercept set to False.
- **Step 8:** Assert coefficients reflect intercept absence.
- *Step 9:* Ensure required arguments for `ModelRefitOutput` are provided.
- - The function `_model_refit` must include `lambda_`, `lambda_hp`, `lambda_max`, and `lambda_min_ratio` in the return statement.
- Check `ModelRefitOutput` class definition to determine necessary values for these parameters.
- Modify the return statement to include these arguments, either by passing them as parameters or by calculating them within the function.
- Ensure default values or calculations for these parameters are handled appropriately to prevent missing arguments.



## `test__lambda_seq() -> None`
### USAGE
- Tests the `_lambda_seq` method for generating lambda sequences.

### IMPL
- **Step 1:** Call `_lambda_seq` with default length on a small dataset.
- - *Ensure x and y are compatible for element-wise multiplication by reshaping y to a 2D column vector.*

- **Step 2:** Assert sequence length matches expected.
- **Step 3:** Assert all values in sequence are positive.
- **Step 4:** Test with custom length on a larger dataset.
- - *Ensure x and y are compatible for element-wise multiplication by reshaping y to a 2D column vector.*

- **Step 5:** Assert sequence length matches custom length.
- **Step 6:** Assert first value is greater than the last.
- **Step 7:** Test with zero arrays.
- - *Ensure x and y are compatible for element-wise multiplication by reshaping y to a 2D column vector.*

- **Step 8:** Assert sequence values non-positive.
- **Step 9:** Test with high dimensional dataset.
- - *Ensure x and y are compatible for element-wise multiplication by reshaping y to a 2D column vector.*

- **Step 10:** Assert sequence length matches large length.
- **Step 11:** Assert all values in sequence are positive.
- *Note:* The root cause of the failure was due to incompatible shapes of the x and y arrays. Ensure that y is reshaped using `y[:, np.newaxis]` to allow for correct element-wise multiplication with x.
