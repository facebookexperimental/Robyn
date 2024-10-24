# CLASS
## FeatureEngineering
* A class used to perform feature engineering for Marketing Mix Modeling (MMM) data.
* This class is part of a module intended for data processing and feature engineering.
* It handles various data transformations and model fittings necessary for MMM.

# CONSTRUCTORS
## `__init__(mmm_data: MMMData, hyperparameters: Hyperparameters, holidays_data: Optional[HolidaysData] = None)`
* Initializes the FeatureEngineering class with MMM data, hyperparameters, and optional holidays data.
### USAGE
* Use this constructor to create an instance of FeatureEngineering with the required data and parameters.
### IMPL
* Mock the MMMData, Hyperparameters, and HolidaysData objects if necessary.
* Ensure the logger is properly initialized and can capture logs during the test execution.

# METHODS
## `perform_feature_engineering(quiet: bool = False) -> FeaturizedMMMData`
### USAGE
* Performs feature engineering on the MMM data using the specified hyperparameters and holidays data.
### IMPL
* Mock dependencies such as the MMMData and HolidaysData methods to control their output.
* Call the `perform_feature_engineering` method with the specified inputs.
* Assert that the resulting `dt_mod.columns` contains the expected columns.
* Assert that the resulting `dt_modRollWind.columns` contains the expected columns.
* Check that `modNLS` is a non-empty dictionary.

## `test_perform_feature_engineering(quiet: bool = False) -> None`
### USAGE
* Test the feature engineering function with complete data.
### IMPL
* Mock the MMMData and HolidaysData to return specific test data.
* Call `perform_feature_engineering` with `quiet=False`.
* Verify the columns of the `dt_mod` DataFrame to match the expected columns.
* Verify the columns of the `dt_modRollWind` DataFrame to match the expected columns.
* Check that `modNLS` is a non-empty dictionary with model results.

## `test_perform_feature_engineering(quiet: bool = False) -> None`
### USAGE
* Test the feature engineering function with missing prophet variables.
### IMPL
* Mock the MMMData and HolidaysData to simulate missing prophet variables.
* Call `perform_feature_engineering` with `quiet=False`.
* Verify the columns of the `dt_mod` DataFrame for expected results.
* Verify the columns of the `dt_modRollWind` DataFrame for expected results.
* Ensure `modNLS` contains a non-empty dictionary with model results.

## `test_perform_feature_engineering(quiet: bool = False) -> None`
### USAGE
* Test the feature engineering function with no holidays data.
### IMPL
* Mock the MMMData without any holidays data.
* Call `perform_feature_engineering` with `quiet=True`.
* Check that the `dt_mod` DataFrame has expected columns.
* Check that the `dt_modRollWind` DataFrame has expected columns.
* Ensure `modNLS` is a non-empty dictionary with model results.

## `test_perform_feature_engineering(quiet: bool = False) -> None`
### USAGE
* Test the feature engineering function with empty data.
### IMPL
* Mock the MMMData and HolidaysData to provide empty data.
* Call `perform_feature_engineering` with `quiet=True`.
* Verify the `dt_mod` DataFrame columns to include only 'ds' and 'dep_var'.
* Verify the `dt_modRollWind` DataFrame columns to include only 'ds' and 'dep_var'.
* Assert `modNLS` is an empty dictionary.

## `_prepare_data() -> pd.DataFrame`
### USAGE
* Prepares the initial dataset by transforming date and dependent variable columns.
### IMPL
* Mock the MMMData to provide a specific dataset.
* Call `_prepare_data` and assert the transformation of date and dependent variable.
* Compare the resulting DataFrame against expected data.

## `test__prepare_data() -> None`
### USAGE
* Test _prepare_data with valid MMMData input.
### IMPL
* Mock the MMMData data and mmmdata_spec methods to provide specific test data.
* Call `_prepare_data`.
* Assert that the resulting DataFrame matches the expected structure and values.

## `test__prepare_data() -> None`
### USAGE
* Test _prepare_data with empty MMMData input.
### IMPL
* Mock the MMMData data and mmmdata_spec methods to provide empty data.
* Call `_prepare_data`.
* Assert that the resulting DataFrame is empty and matches the expected structure.

## `test__prepare_data() -> None`
### USAGE
* Test _prepare_data with missing columns in MMMData.
### IMPL
* Mock the MMMData to simulate missing columns.
* Call `_prepare_data`.
* Expect a KeyError and verify the exception message.

## `test__prepare_data() -> None`
### USAGE
* Test _prepare_data with non-date formatted date column.
### IMPL
* Mock the MMMData to provide a non-date formatted date column.
* Call `_prepare_data`.
* Expect a ValueError and verify the exception message contains date format information.

## `_create_rolling_window_data(dt_transform: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Creates a rolling window dataset based on the specified window start and end dates.
### IMPL
* Mock the MMMData spec to control window start and end dates.
* Call `_create_rolling_window_data` and verify the resulting DataFrame matches expected data.

## `test__create_rolling_window_data() -> None`
### USAGE
* Test with both window_start and window_end as None.
### IMPL
* Mock MMMData to set both window_start and window_end to None.
* Call `_create_rolling_window_data`.
* Assert that the resulting DataFrame matches the input DataFrame.

## `test__create_rolling_window_data() -> None`
### USAGE
* Test with window_end before data start.
### IMPL
* Mock MMMData to set window_end before the start of the data.
* Call `_create_rolling_window_data`.
* Expect a ValueError and verify the exception message.

## `test__create_rolling_window_data() -> None`
### USAGE
* Test with window_start after data end.
### IMPL
* Mock MMMData to set window_start after the end of the data.
* Call `_create_rolling_window_data`.
* Expect a ValueError and verify the exception message.

## `test__create_rolling_window_data() -> None`
### USAGE
* Test with window_start after window_end.
### IMPL
* Mock MMMData to set window_start after window_end.
* Call `_create_rolling_window_data`.
* Expect a ValueError and verify the exception message.

## `test__create_rolling_window_data() -> None`
### USAGE
* Test with valid window_start and window_end within data range.
### IMPL
* Mock MMMData to set window_start and window_end within the data range.
* Call `_create_rolling_window_data`.
* Verify the resulting DataFrame matches the expected range.

## `_calculate_media_cost_factor(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* Calculates the media cost factor for the given rolling window dataset.
### IMPL
* Mock the MMMData spec to provide paid media spends.
* Call `_calculate_media_cost_factor` and verify the resulting Series matches expected values.

## `test__calculate_media_cost_factor() -> None`
### USAGE
* Calculate media cost factor with valid input.
### IMPL
* Mock MMMData and dt_input_roll_wind to provide valid media spends.
* Call `_calculate_media_cost_factor`.
* Assert that the resulting media cost factor matches expected values.

## `test__calculate_media_cost_factor() -> None`
### USAGE
* Calculate media cost factor with zero total spend.
### IMPL
* Mock MMMData and dt_input_roll_wind to simulate zero total spend.
* Call `_calculate_media_cost_factor`.
* Assert that the resulting media cost factor contains NaN values.

## `test__calculate_media_cost_factor() -> None`
### USAGE
* Calculate media cost factor with negative spends.
### IMPL
* Mock MMMData and dt_input_roll_wind to provide negative spends.
* Call `_calculate_media_cost_factor`.
* Assert that the resulting media cost factor matches expected values.

## `test__calculate_media_cost_factor() -> None`
### USAGE
* Calculate media cost factor with one media spend.
### IMPL
* Mock MMMData and dt_input_roll_wind to provide single media spend.
* Call `_calculate_media_cost_factor`.
* Assert that the resulting media cost factor is 1.0 for the single spend.

## `_run_models(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict[str, Dict[str, Any]]`
### USAGE
* Runs the models for each paid media variable and returns the results.
### IMPL
* Mock MMMData to provide paid media spends.
* Mock the _fit_spend_exposure method to control its output.
* Call `_run_models` and verify the resulting modNLS dictionary matches expectations.

## `test__run_models() -> None`
### USAGE
* Successful execution with valid MMMData and media cost factor.
### IMPL
* Mock MMMData and _fit_spend_exposure to provide valid results for paid media.
* Call `_run_models`.
* Assert `modNLS['results']` contains expected results for each media.
* Check that `modNLS['yhat']` is a concatenated DataFrame.

## `test__run_models() -> None`
### USAGE
* Handles empty paid_media_spends list.
### IMPL
* Mock MMMData with an empty paid_media_spends list.
* Call `_run_models`.
* Assert `modNLS['results']` is an empty dictionary.
* Check that `modNLS['yhat']` is an empty DataFrame.

## `test__run_models() -> None`
### USAGE
* Handles missing result from _fit_spend_exposure.
### IMPL
* Mock MMMData with one paid_media variable.
* Mock _fit_spend_exposure to return None.
* Call `_run_models`.
* Assert `modNLS['results']` is an empty dictionary.
* Check that `modNLS['yhat']` is an empty DataFrame.

## `test__run_models() -> None`
### USAGE
* Execution with a large set of paid media variables.
### IMPL
* Mock MMMData and _fit_spend_exposure to handle multiple media variables.
* Call `_run_models`.
* Assert that `modNLS['results']` contains expected results for each media.
* Verify `modNLS['yhat']` is a concatenated DataFrame of all plot DataFrames.

## `_fit_spend_exposure(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict[str, Any]`
### USAGE
* Fits the spend-exposure model for a given paid media variable and returns the results.
### IMPL
* Mock the data to simulate different scenarios for spend and exposure.
* Call `_fit_spend_exposure` and verify the resulting dictionary matches expectations.

## `test__fit_spend_exposure() -> None`
### USAGE
* Test with valid spend and exposure data - Michaelis-Menten model is better.
### IMPL
* Mock MMMData with valid spend and exposure data.
* Call `_fit_spend_exposure`.
* Assert the model type is 'nls'.
* Check that `res.rsq` is 1.0 and coefficients are greater than 0.

## `test__fit_spend_exposure() -> None`
### USAGE
* Test with valid spend and exposure data - Linear model is better.
### IMPL
* Mock MMMData with linear spend and exposure data.
* Call `_fit_spend_exposure`.
* Assert the model type is 'lm'.
* Check that `res.rsq` is 1.0 and coefficient is 1.0.

## `test__fit_spend_exposure() -> None`
### USAGE
* Test with empty exposure data.
### IMPL
* Mock MMMData with empty exposure data.
* Call `_fit_spend_exposure`.
* Assert that the result is None.

## `test__fit_spend_exposure() -> None`
### USAGE
* Test with invalid data causing exception in Michaelis-Menten fitting.
### IMPL
* Mock MMMData with invalid data for Michaelis-Menten.
* Call `_fit_spend_exposure`.
* Assert the fallback to 'lm' model type.
* Check that `res.rsq` is less than 1.0 and coefficient is greater than 0.

## `_hill_function(x, alpha, gamma)`
### USAGE
* Static method to apply the Hill function transformation.
### IMPL
* Provide test values for x, alpha, and gamma.
* Call `_hill_function` and verify the result matches expected output.

## `test__hill_function() -> None`
### USAGE
* Test hill function with positive alpha and gamma.
### IMPL
* Provide values x=2.0, alpha=1.0, gamma=1.0.
* Call `_hill_function`.
* Assert the output is approximately 0.6667.

## `test__hill_function() -> None`
### USAGE
* Test hill function with alpha equal to zero, should return zero.
### IMPL
* Provide values x=2.0, alpha=0.0, gamma=1.0.
* Call `_hill_function`.
* Assert the output is 0.0.

## `test__hill_function() -> None`
### USAGE
* Test hill function with gamma equal to zero, should return one.
### IMPL
* Provide values x=2.0, alpha=1.0, gamma=0.0.
* Call `_hill_function`.
* Assert the output is 1.0.

## `test__hill_function() -> None`
### USAGE
* Test hill function with large x value.
### IMPL
* Provide values x=1000.0, alpha=1.0, gamma=1.0.
* Call `_hill_function`.
* Assert the output is approximately 0.9990.

## `test__hill_function() -> None`
### USAGE
* Test hill function with small x value.
### IMPL
* Provide values x=0.01, alpha=1.0, gamma=1.0.
* Call `_hill_function`.
* Assert the output is approximately 0.0099.

## `test__hill_function() -> None`
### USAGE
* Test hill function with negative alpha value.
### IMPL
* Provide values x=2.0, alpha=-1.0, gamma=1.0.
* Call `_hill_function`.
* Assert the output is approximately 0.6.

## `test__hill_function() -> None`
### USAGE
* Test hill function with negative gamma value.
### IMPL
* Provide values x=2.0, alpha=1.0, gamma=-1.0.
* Call `_hill_function`.
* Assert the output is approximately 0.6667.

## `_prophet_decomposition(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Performs Prophet decomposition on the dataset and returns the transformed data.
### IMPL
* Mock the HolidaysData to provide specific prophet variables.
* Call `_prophet_decomposition` and verify the resulting DataFrame matches expectations.

## `test__prophet_decomposition() -> None`
### USAGE
* Basic test with all Prophet variables enabled.
### IMPL
* Mock HolidaysData and MMMData with all prophet variables.
* Call `_prophet_decomposition`.
* Assert that each prophet component column contains non-null values.

## `test__prophet_decomposition() -> None`
### USAGE
* Test with no holidays and trend variable.
### IMPL
* Mock HolidaysData to exclude holidays and trend.
* Call `_prophet_decomposition`.
* Assert that 'trend' and 'holiday' columns are null, others are non-null.

## `test__prophet_decomposition() -> None`
### USAGE
* Test with no factor variables.
### IMPL
* Mock HolidaysData and MMMData to exclude factor variables.
* Call `_prophet_decomposition`.
* Assert that 'monthly' and 'weekday' columns are null, others are non-null.

## `test__prophet_decomposition() -> None`
### USAGE
* Test with custom parameters overriding defaults.
### IMPL
* Mock HolidaysData and MMMData with custom prophet parameters.
* Call `_prophet_decomposition`.
* Assert that 'season' is null due to override, others are as expected.

## `_set_holidays(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* Sets the holidays in the dataset based on the specified interval type.
### IMPL
* Mock the holidays data to control the input for different interval types.
* Call `_set_holidays` and verify the resulting DataFrame matches expectations.

## `test_set_holidays() -> None`
### USAGE
* Ensure holidays are returned unmodified when interval_type is 'day'.
### IMPL
* Provide test data with interval_type 'day'.
* Call `_set_holidays`.
* Assert that the holidays DataFrame matches the input.

## `test_set_holidays() -> None`
### USAGE
* Verify holidays are adjusted to week start when interval_type is 'week'.
### IMPL
* Provide test data with interval_type 'week'.
* Call `_set_holidays`.
* Assert that holidays are adjusted to week start.

## `test_set_holidays() -> None`
### USAGE
* Test holidays aggregation by month when interval_type is 'month'.
### IMPL
* Provide test data with interval_type 'month'.
* Call `_set_holidays`.
* Assert that holidays are aggregated by month.

## `test_set_holidays() -> None`
### USAGE
* Test error is raised when interval_type is 'month' and data does not start on the first day.
### IMPL
* Provide test data with non-first day and interval_type 'month'.
* Call `_set_holidays`.
* Expect a ValueError and verify the exception message.

## `test_set_holidays() -> None`
### USAGE
* Verify invalid interval_type raises ValueError.
### IMPL
* Provide test data with an invalid interval_type.
* Call `_set_holidays`.
* Expect a ValueError and verify the exception message.

## `_apply_transformations(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies adstock and saturation transformations to the given series.
### IMPL
* Mock the input series and parameters to control the test case.
* Call `_apply_transformations` and verify the resulting Series matches expectations.

## `test__apply_transformations() -> None`
### USAGE
* Test adstock and saturation transformations with geometric adstock type.
### IMPL
* Mock the input series with positive values and parameters for geometric adstock.
* Call `_apply_transformations`.
* Assert the resulting Series type is pd.Series and is transformed.

## `test__apply_transformations() -> None`
### USAGE
* Test adstock and saturation transformations with Weibull adstock type.
### IMPL
* Mock the input series with negative and zero values and parameters for Weibull adstock.
* Call `_apply_transformations`.
* Assert the resulting Series type is pd.Series and is transformed.

## `test__apply_transformations() -> None`
### USAGE
* Test adstock and saturation transformations with no transformation.
### IMPL
* Mock the input series with constant values and parameters for no transformation.
* Call `_apply_transformations`.
* Assert the resulting Series type is pd.Series and is not transformed.

## `test__apply_transformations() -> None`
### USAGE
* Test adstock and saturation transformations with empty series.
### IMPL
* Mock the input as an empty series.
* Call `_apply_transformations`.
* Assert the resulting Series is empty.

## `_apply_adstock(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies the specified adstock transformation to the given series.
### IMPL
* Mock the input series and parameters to simulate different adstock transformations.
* Call `_apply_adstock` and verify the resulting Series matches expectations.

## `test__apply_adstock() -> None`
### USAGE
* Apply geometric adstock transformation successfully.
### IMPL
* Mock the input series and parameters for geometric adstock.
* Call `_apply_adstock`.
* Assert the resulting Series matches expected transformed values.

## `test__apply_adstock() -> None`
### USAGE
* Apply Weibull CDF adstock transformation successfully.
### IMPL
* Mock the input series and parameters for Weibull CDF adstock.
* Call `_apply_adstock`.
* Assert the resulting Series matches expected transformed values.

## `test__apply_adstock() -> None`
### USAGE
* Apply Weibull PDF adstock transformation successfully.
### IMPL
* Mock the input series and parameters for Weibull PDF adstock.
* Call `_apply_adstock`.
* Assert the resulting Series matches expected transformed values.

## `test__apply_adstock() -> None`
### USAGE
* Raise ValueError for unsupported adstock type.
### IMPL
* Mock the input series and parameters with an unsupported adstock type.
* Call `_apply_adstock`.
* Expect a ValueError and verify the exception message.

## `_geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* Static method to apply geometric adstock transformation.
### IMPL
* Provide a series and theta value to test the geometric adstock.
* Call `_geometric_adstock` and verify the transformed Series matches expected output.

## `test__geometric_adstock() -> None`
### USAGE
* Test geometric adstock with typical series and theta value.
### IMPL
* Provide a test series and theta value.
* Call `_geometric_adstock`.
* Assert the transformed Series matches expected values.

## `test__geometric_adstock_empty_series() -> None`
### USAGE
* Test geometric adstock with an empty series.
### IMPL
* Provide an empty series and theta value.
* Call `_geometric_adstock`.
* Assert the resulting Series is empty.

## `test__geometric_adstock_single_value_series() -> None`
### USAGE
* Test geometric adstock with a single value series.
### IMPL
* Provide a single value series and theta value.
* Call `_geometric_adstock`.
* Assert the resulting Series matches the input series.

## `test__geometric_adstock_zero_theta() -> None`
### USAGE
* Test geometric adstock with theta as zero.
### IMPL
* Provide a test series and theta value of zero.
* Call `_geometric_adstock`.
* Assert the resulting Series is constant and matches expectations.

## `test__geometric_adstock_one_theta() -> None`
### USAGE
* Test geometric adstock with theta as one.
### IMPL
* Provide a test series and theta value of one.
* Call `_geometric_adstock`.
* Assert the resulting Series matches the input series.

## `_weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* Static method to apply Weibull adstock transformation.
### IMPL
* Provide a series, shape, and scale to test the Weibull adstock.
* Call `_weibull_adstock` and verify the transformed Series matches expected output.

## `test__weibull_adstock() -> None`
### USAGE
* Test Weibull adstock transformation with typical input series and parameters.
### IMPL
* Provide a test series and parameters for Weibull transformation.
* Call `_weibull_adstock`.
* Assert the transformed Series matches expected values.

## `test__weibull_adstock() -> None`
### USAGE
* Test Weibull adstock transformation with edge case of empty input series.
### IMPL
* Provide an empty series and parameters.
* Call `_weibull_adstock`.
* Assert the resulting Series is empty.

## `test__weibull_adstock() -> None`
### USAGE
* Test Weibull adstock transformation with single element input series.
### IMPL
* Provide a single element series and parameters.
* Call `_weibull_adstock`.
* Assert the resulting Series matches expected transformed value.

## `test__weibull_adstock() -> None`
### USAGE
* Test Weibull adstock transformation with large scale parameter.
### IMPL
* Provide a test series and a large scale parameter.
* Call `_weibull_adstock`.
* Assert the transformed Series matches expected values.

## `test__weibull_adstock() -> None`
### USAGE
* Test Weibull adstock transformation with large shape parameter.
### IMPL
* Provide a test series and a large shape parameter.
* Call `_weibull_adstock`.
* Assert the transformed Series matches expected values.

## `test__weibull_adstock() -> None`
### USAGE
* Test Weibull adstock transformation with zero shape parameter.
### IMPL
* Provide a test series and a zero shape parameter.
* Call `_weibull_adstock`.
* Expect an exception and verify the exception message.

## `test__weibull_adstock() -> None`
### USAGE
* Test Weibull adstock transformation with zero scale parameter.
### IMPL
* Provide a test series and a zero scale parameter.
* Call `_weibull_adstock`.
* Expect an exception and verify the exception message.

## `_apply_saturation(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Static method to apply saturation transformation.
### IMPL
* Mock the input series and parameters to simulate different saturation transformations.
* Call `_apply_saturation` and verify the resulting Series matches expectations.

## `test__apply_saturation() -> None`
### USAGE
* Test when 'x' is a regular positive series and 'alpha' and 'gamma' are valid positive floats.
### IMPL
* Provide a positive series and valid alpha and gamma values.
* Call `_apply_saturation`.
* Assert the resulting Series matches expected transformed values.

## `test__apply_saturation() -> None`
### USAGE
* Test when 'x' contains zeros and 'alpha' and 'gamma' are valid positive floats.
### IMPL
* Provide a series containing zero and positive values.
* Call `_apply_saturation`.
* Assert zeros in the input result in zeros in the output.

## `test__apply_saturation() -> None`
### USAGE
* Test when 'x' is a series with negative values and 'alpha' and 'gamma' are positive floats.
### IMPL
* Provide a negative series and positive alpha and gamma values.
* Call `_apply_saturation`.
* Assert the resulting Series may contain warnings or NaN.

## `test__apply_saturation() -> None`
### USAGE
* Test with 'alpha' and 'gamma' as zero, expecting division by zero handling.
### IMPL
* Provide a positive series and zero alpha and gamma values.
* Call `_apply_saturation`.
* Assert the resulting Series contains NaN due to division by zero.

## `test__apply_saturation() -> None`
### USAGE
* Test with large 'alpha' and 'gamma' values to check function behavior under extreme scaling.
### IMPL
* Provide a positive series and large alpha and gamma values.
* Call `_apply_saturation`.
* Assert the resulting Series values are close to zero due to extreme scaling.