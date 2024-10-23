# CLASS
## RidgeModelBuilder
* This class is responsible for building and training Ridge regression models using marketing mix modeling data.
* It manages data preparation, model training, hyperparameter optimization, and model selection.
* It utilizes several entities such as `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.

# METHODS

## `test_build_models(input: TrialsConfig, dt_hyper_fixed: Optional[pd.DataFrame], ts_validation: bool, add_penalty_factor: bool, seed: int, rssd_zero_penalty: bool, objective_weights: Optional[List[float]], nevergrad_algo: NevergradAlgorithm, intercept: bool, intercept_sign: str, cores: int) -> ModelOutputs`
### USAGE
* This method tests the `build_models` function of the RidgeModelBuilder class under various scenarios, including default configurations, time-series validation, and penalty factor enabled. It verifies the model's ability to produce outputs that meet expected criteria.

### IMPL
* **Step 1:** Initialize the RidgeModelBuilder with mock dependencies and inputs.
* **Step 2:** Mock the `Convergence` class's `calculate_convergence` method to return a successful convergence message.
* **Step 3:** Mock the `_hyper_collector` method to simulate hyperparameter collection with default parameters.
* **Step 4:** Mock the `_model_train` method to return a list of mock Trial objects.
* **Step 5:** Mock the `_select_best_model` method to return a mock best model ID.
* **Step 6:** Call the `build_models` method with the specified input parameters.
* **Step 7:** Assert that `model_outputs.trials` matches the expected list of Trial objects.
* **Step 8:** Assert that `model_outputs.convergence.conv_msg` contains the expected convergence message.
* **Step 9:** Assert that `model_outputs.select_id` matches the expected best model ID.

## `test__select_best_model(output_models: List[Trial]) -> str`
### USAGE
* This method tests the `_select_best_model` function of the RidgeModelBuilder class to ensure it correctly selects the best model based on given criteria.

### IMPL
* **Step 1:** Provide a list of mock Trial objects with distinct nrmse and decomp_rssd values.
* **Step 2:** Call the `_select_best_model` method with the provided input models.
* **Step 3:** Assert that the returned solID matches the expected value based on the lowest combined score.
* **Step 4:** Repeat the test with models having the same nrmse and decomp_rssd values.
* **Step 5:** Assert that the returned solID defaults to the first model in case of a tie.
* **Step 6:** Test with models having inversely proportional nrmse and decomp_rssd values.
* **Step 7:** Assert that the returned solID aligns with the expected best model based on combined score.

## `test__model_train(hyper_collect: Dict[str, Any], trials_config: TrialsConfig, intercept_sign: str, intercept: bool, nevergrad_algo: NevergradAlgorithm, dt_hyper_fixed: Optional[pd.DataFrame], ts_validation: bool, add_penalty_factor: bool, objective_weights: Optional[List[float]], rssd_zero_penalty: bool, seed: int, cores: int) -> List[Trial]`
### USAGE
* This method tests the `_model_train` function of the RidgeModelBuilder class, evaluating its ability to handle different configurations and produce expected trial results.

### IMPL
* **Step 1:** Mock the `_run_nevergrad_optimization` to simulate successful trial execution.
* **Step 2:** Call the `_model_train` method with valid parameters where all trials should succeed.
* **Step 3:** Assert that the length of trials matches the expected trial count.
* **Step 4:** Assert that each trial result indicates success.
* **Step 5:** Test with zero trials to ensure it handles gracefully.
* **Step 6:** Assert that the trials list is empty when no trials are configured.
* **Step 7:** Test with `add_penalty_factor` enabled to verify behavior change.
* **Step 8:** Assert that trial results include penalty factor success indications.

## `test__run_nevergrad_optimization(hyper_collect: Dict[str, Any], iterations: int, cores: int, nevergrad_algo: NevergradAlgorithm, intercept: bool, intercept_sign: str, ts_validation: bool, add_penalty_factor: bool, objective_weights: Optional[List[float]], dt_hyper_fixed: Optional[pd.DataFrame], rssd_zero_penalty: bool, trial: int, seed: int, total_trials: int) -> Trial`
### USAGE
* This method tests the `_run_nevergrad_optimization` function of the RidgeModelBuilder class to ensure it performs optimization under different configurations and returns the best trial.

### IMPL
* **Step 1:** Mock the `ng.optimizers.registry` to simulate optimizer instance creation.
* **Step 2:** Call `_run_nevergrad_optimization` with basic valid inputs.
* **Step 3:** Assert that the optimizer is called with correct parameters.
* **Step 4:** Assert that the best trial parameters match expected values.
* **Step 5:** Test with no cores specified, ensuring single-core execution.
* **Step 6:** Assert that the optimizer operates correctly with a single worker.
* **Step 7:** Test with fixed hyperparameters and ensure optimization respects fixed values.
* **Step 8:** Assert that optimization outputs reflect fixed parameters.

## `test__prepare_data(params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]`
### USAGE
* This method tests the `_prepare_data` function of the RidgeModelBuilder class to ensure it correctly processes input data based on specified parameters.

### IMPL
* **Step 1:** Call `_prepare_data` with empty parameters and empty dataframes.
* **Step 2:** Assert that the output X is a pandas DataFrame with zero length.
* **Step 3:** Assert that the output y is a pandas Series with zero length.
* **Step 4:** Test with parameters that do not match any media spends.
* **Step 5:** Assert that X and y remain unchanged from their original versions.
* **Step 6:** Test with valid parameters affecting media spend columns.
* **Step 7:** Mock transformations to verify correct application.
* **Step 8:** Assert that transformed media spend matches expected series.
* **Step 9:** Test for handling NaN and infinite values.
* **Step 10:** Assert that output data does not contain NaN or infinite values.

## `test__geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function of the RidgeModelBuilder class to verify its transformation logic under different scenarios.

### IMPL
* **Step 1:** Provide a small array and theta=0.5 for basic functionality testing.
* **Step 2:** Call `_geometric_adstock` with the provided inputs.
* **Step 3:** Assert that the output matches the expected transformed series.
* **Step 4:** Test with an empty series for edge case handling.
* **Step 5:** Assert that the output remains an empty series.
* **Step 6:** Test with theta=0 to ensure no decay effect.
* **Step 7:** Assert that the output matches the input series.
* **Step 8:** Test with theta=1 for full decay effect.
* **Step 9:** Assert that the output series reflects expected full decay transformation.
* **Step 10:** Test with negative numbers in the series.
* **Step 11:** Assert that the output series matches expected transformation for negative values.
* **Step 12:** Test with theta greater than 1.
* **Step 13:** Assert that the output series reflects transformation with increased decay.

## `test__hill_transformation(x: pd.Series, alpha: float, gamma: float) -> pd.Series`
### USAGE
* This method tests the `_hill_transformation` function of the RidgeModelBuilder class to ensure proper saturation transformation with different parameters.

### IMPL
* **Step 1:** Provide a series of normal values and parameters for basic testing.
* **Step 2:** Call `_hill_transformation` with the given inputs.
* **Step 3:** Assert that the output matches the expected transformed series.
* **Step 4:** Test with alpha as zero.
* **Step 5:** Assert that the output series reflects uniform transformation due to zero alpha.
* **Step 6:** Test with gamma as zero.
* **Step 7:** Assert that the output series reflects transformation with gamma effect.
* **Step 8:** Test with x having all identical values.
* **Step 9:** Assert that the output series contains NaN due to division by zero.
* **Step 10:** Test with an empty series.
* **Step 11:** Assert that the output remains an empty series.
* **Step 12:** Test with large alpha and gamma.
* **Step 13:** Assert that the output series values are transformed close to zero.

## `test__calculate_rssd(coefs: np.ndarray, rssd_zero_penalty: bool) -> float`
### USAGE
* This method tests the `_calculate_rssd` function of the RidgeModelBuilder class to verify RSSD calculations under different configurations.

### IMPL
* **Step 1:** Provide coefficients and RSSD without zero penalty.
* **Step 2:** Call `_calculate_rssd` with the provided inputs.
* **Step 3:** Assert that the calculated RSSD matches the expected value.
* **Step 4:** Test with zero penalty enabled.
* **Step 5:** Assert that the RSSD reflects zero coefficient penalty.
* **Step 6:** Test with all zero coefficients.
* **Step 7:** Assert that the RSSD is zero.
* **Step 8:** Test with mixed coefficients.
* **Step 9:** Assert that the RSSD matches expected value for mixed coefficients.
* **Step 10:** Test with large coefficients.
* **Step 11:** Assert that the RSSD reflects the magnitude of coefficients.
* **Step 12:** Test with a single coefficient.
* **Step 13:** Assert that the RSSD equals the absolute value of the coefficient.
* **Step 14:** Test with negative coefficients.
* **Step 15:** Assert that the RSSD matches expected value for negative coefficients.

## `test__calculate_mape(model: Ridge) -> float`
### USAGE
* This method tests the `_calculate_mape` function of the RidgeModelBuilder class to verify MAPE calculations under different conditions.

### IMPL
* **Step 1:** Mock dependencies to simulate valid calibration data.
* **Step 2:** Call `_calculate_mape` with a Ridge model instance.
* **Step 3:** Assert that the calculated MAPE matches the expected value.
* **Step 4:** Test with no calibration input.
* **Step 5:** Assert that the MAPE is zero when no calibration data is present.
* **Step 6:** Test with empty calibration data.
* **Step 7:** Assert that the MAPE remains zero for empty calibration data.
* **Step 8:** Test with multiple calibration data points.
* **Step 9:** Assert that the MAPE reflects the mean value across all calibration points.

## `test__evaluate_model(params: Dict[str, float], ts_validation: bool, add_penalty_factor: bool, rssd_zero_penalty: bool, objective_weights: Optional[List[float]]) -> Tuple[float, float, float, float, Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame, float, float, float, float, float, float, float, int]`
### USAGE
* This method tests the `_evaluate_model` function of the RidgeModelBuilder class to ensure correct model evaluation under various configurations.

### IMPL
* **Step 1:** Mock dependencies to simulate valid input data and outputs.
* **Step 2:** Call `_evaluate_model` with specified parameters.
* **Step 3:** Assert that loss and evaluation metrics fall within expected ranges.
* **Step 4:** Test with time-series validation enabled.
* **Step 5:** Assert that evaluation metrics reflect validation results.
* **Step 6:** Test with custom objective weights.
* **Step 7:** Assert that loss calculation respects custom weights.
* **Step 8:** Test with missing optional parameters.
* **Step 9:** Assert that defaults are correctly applied in evaluation.

## `test__hyper_collector(hyperparameters: Hyperparameters, ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> Dict[str, Any]`
### USAGE
* This method tests the `_hyper_collector` function of the RidgeModelBuilder class to ensure proper collection and organization of hyperparameters.

### IMPL
* **Step 1:** Call `_hyper_collector` with fixed hyperparameters provided.
* **Step 2:** Assert that `hyper_list_all` matches the expected collected hyperparameters.
* **Step 3:** Assert that `hyper_bound_list_updated` accurately reflects updated hyperparameters.
* **Step 4:** Test without fixed hyperparameters.
* **Step 5:** Assert that `dt_hyper_fixed_mod` is empty and `all_fixed` is False.
* **Step 6:** Test with empty hyperparameters.
* **Step 7:** Assert that all collections remain empty.
* **Step 8:** Test with non-list parameter values.
* **Step 9:** Assert that `hyper_bound_list_fixed` captures non-list values.

## `test__model_refit(x_train: np.ndarray, y_train: np.ndarray, x_val: Optional[np.ndarray], y_val: Optional[np.ndarray], x_test: Optional[np.ndarray], y_test: Optional[np.ndarray], lambda_: float, lower_limits: Optional[List[float]], upper_limits: Optional[List[float]], intercept: bool, intercept_sign: str) -> ModelRefitOutput`
### USAGE
* This method tests the `_model_refit` function of the RidgeModelBuilder class to ensure proper model refitting and output generation.

### IMPL
* **Step 1:** Call `_model_refit` with basic train data only.
* **Step 2:** Assert that training metrics fall within expected ranges.
* **Step 3:** Test with train and validation data.
* **Step 4:** Assert that validation metrics are correctly calculated.
* **Step 5:** Test with train, validation, and test data.
* **Step 6:** Assert that all metrics are calculated for each dataset.
* **Step 7:** Test with intercept set to False.
* **Step 8:** Assert that model coefficients and predictions reflect intercept absence.

## `test__lambda_seq(x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float) -> np.ndarray`
### USAGE
* This method tests the `_lambda_seq` function of the RidgeModelBuilder class to ensure correct generation of lambda sequences for regularization.

### IMPL
* **Step 1:** Call `_lambda_seq` with default sequence length and lambda_min_ratio on small dataset.
* **Step 2:** Assert that the length of the sequence matches the expected value.
* **Step 3:** Assert that all values in the sequence are positive.
* **Step 4:** Test with custom sequence length and lambda_min_ratio on larger dataset.
* **Step 5:** Assert that the sequence length matches the specified custom length.
* **Step 6:** Assert that the first value is greater than the last value in the sequence.
* **Step 7:** Test with x and y as zero arrays for edge handling.
* **Step 8:** Assert that all sequence values are non-positive, reflecting edge case.
* **Step 9:** Test with high dimensional dataset and very small lambda_min_ratio.
* **Step 10:** Assert that the sequence length matches the expected large length.
* **Step 11:** Assert that all values in the sequence are positive.