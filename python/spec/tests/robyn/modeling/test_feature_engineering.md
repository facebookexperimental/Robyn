# METHODS

## `test_perform_feature_engineering(quiet: bool) -> None`
### USAGE
* This method tests the `perform_feature_engineering` function of the `FeatureEngineering` class with different configurations of input data and mocked methods.
* `quiet` parameter controls the verbosity of the function. When `false`, the function will print additional information during execution.

### IMPL
* Mock the `_prepare_data` method of `FeatureEngineering` to return a "Mocked DataFrame".
* Mock the `_prophet_decomposition` method to take "Mocked DataFrame" as input and return "Prophet Decomposed DataFrame".
* Mock the `_create_rolling_window_data` method to take "Prophet Decomposed DataFrame" as input and return "Rolling Window DataFrame".
* Mock the `_calculate_media_cost_factor` method to take "Rolling Window DataFrame" as input and return "Media Cost Factor".
* Mock the `_run_models` method to take "Rolling Window DataFrame" and "Media Cost Factor" as input and return "Model Results".
* Call `perform_feature_engineering` with `quiet` parameter set to `false`.
* Assert that the returned `dt_mod` is a "Filtered DataFrame with relevant columns".
* Assert that the returned `dt_modRollWind` is a "Filtered Rolling Window DataFrame with relevant columns".
* Assert that the returned `modNLS` is "Model Results".

## `test_perform_feature_engineering_no_prophet_vars(quiet: bool) -> None`
### USAGE
* This method tests the `perform_feature_engineering` function when no prophet variables are present.

### IMPL
* Mock the `_prepare_data` method of `FeatureEngineering` to return a "Mocked DataFrame".
* Mock the `_create_rolling_window_data` method to take "Mocked DataFrame" as input and return "Rolling Window DataFrame".
* Mock the `_calculate_media_cost_factor` method to take "Rolling Window DataFrame" as input and return "Media Cost Factor".
* Mock the `_run_models` method to take "Rolling Window DataFrame" and "Media Cost Factor" as input and return "Model Results".
* Call `perform_feature_engineering` with `quiet` parameter set to `true`.
* Assert that the returned `dt_mod` is a "Filtered DataFrame with relevant columns".
* Assert that the returned `dt_modRollWind` is a "Filtered Rolling Window DataFrame with relevant columns".
* Assert that the returned `modNLS` is "Model Results".

## `test_perform_feature_engineering_missing_columns(quiet: bool) -> None`
### USAGE
* This method tests the `perform_feature_engineering` function when some columns are missing.

### IMPL
* Mock the `_prepare_data` method of `FeatureEngineering` to return a "DataFrame with missing columns".
* Mock the `_create_rolling_window_data` method to take "DataFrame with missing columns" as input and return "Rolling Window DataFrame with missing columns".
* Mock the `_calculate_media_cost_factor` method to take "Rolling Window DataFrame with missing columns" as input and return "Media Cost Factor".
* Mock the `_run_models` method to take "Rolling Window DataFrame with missing columns" and "Media Cost Factor" as input and return "Model Results".
* Call `perform_feature_engineering` with `quiet` parameter set to `true`.
* Assert that the returned `dt_mod` is a "Filtered DataFrame with available relevant columns".
* Assert that the returned `dt_modRollWind` is a "Filtered Rolling Window DataFrame with available relevant columns".
* Assert that the returned `modNLS` is "Model Results".

## `test__prepare_data(mmm_data: Dict, hyperparameters: Dict, holidays_data: Optional) -> None`
### USAGE
* This method tests the `_prepare_data` function with a typical input of `MMMData`.

### IMPL
* Simulate `mmm_data` with `data` containing "date", "sales", and "competitor_sales_B" columns and `mmmdata_spec` specifying `date_var` as "date" and `dep_var` as "sales".
* Call `_prepare_data`.
* Assert that the columns of the returned DataFrame are `['date', 'sales', 'competitor_sales_B', 'ds', 'dep_var']`.
* Assert that the data types of "competitor_sales_B" and "ds" are `int64` and `object`, respectively.

## `test__prepare_data_missing_competitor_sales_B(mmm_data: Dict, hyperparameters: Dict, holidays_data: Optional) -> None`
### USAGE
* This method tests the `_prepare_data` function when the "competitor_sales_B" column is missing.

### IMPL
* Simulate `mmm_data` with `data` containing "date" and "sales" columns and `mmmdata_spec` specifying `date_var` as "date" and `dep_var` as "sales".
* Call `_prepare_data` and expect a `KeyError`.
* Assert that the exception message is `"'competitor_sales_B'"`.

## `test__prepare_data_empty_input(mmm_data: Dict, hyperparameters: Dict, holidays_data: Optional) -> None`
### USAGE
* This method tests the `_prepare_data` function with empty data input.

### IMPL
* Simulate `mmm_data` with empty `data` for "date", "sales", and "competitor_sales_B", and `mmmdata_spec` specifying `date_var` as "date" and `dep_var` as "sales".
* Call `_prepare_data`.
* Assert that the shape of the returned DataFrame is `[0, 5]`.

## `test__prepare_data_null_values_in_competitor_sales_B(mmm_data: Dict, hyperparameters: Dict, holidays_data: Optional) -> None`
### USAGE
* This method tests the `_prepare_data` function when the "competitor_sales_B" column contains null values.

### IMPL
* Simulate `mmm_data` with `data` containing "date", "sales", and null values in "competitor_sales_B" and `mmmdata_spec` specifying `date_var` as "date" and `dep_var` as "sales".
* Call `_prepare_data`.
* Assert that the data type of "competitor_sales_B" is `int64`.

## `test__create_rolling_window_data(dt_transform: pd.DataFrame, mmm_data: Dict) -> None`
### USAGE
* This method tests the `_create_rolling_window_data` function with both `window_start` and `window_end` as `None`.

### IMPL
* Simulate `dt_transform` DataFrame with "ds" and "value" columns and `mmm_data` with `mmmdata_spec` having `window_start` and `window_end` as `None`.
* Call `_create_rolling_window_data`.
* Assert that the output DataFrame matches the input `dt_transform`.

## `test__create_rolling_window_data_with_window_end(dt_transform: pd.DataFrame, mmm_data: Dict) -> None`
### USAGE
* This method tests the `_create_rolling_window_data` function with only `window_end` set.

### IMPL
* Simulate `dt_transform` DataFrame with "ds" and "value" columns and `mmm_data` with `mmmdata_spec` having `window_end` set.
* Call `_create_rolling_window_data`.
* Assert that the output DataFrame only includes rows with "ds" up to `window_end`.

## `test__create_rolling_window_data_with_window_start(dt_transform: pd.DataFrame, mmm_data: Dict) -> None`
### USAGE
* This method tests the `_create_rolling_window_data` function with only `window_start` set.

### IMPL
* Simulate `dt_transform` DataFrame with "ds" and "value" columns and `mmm_data` with `mmmdata_spec` having `window_start` set.
* Call `_create_rolling_window_data`.
* Assert that the output DataFrame only includes rows with "ds" starting from `window_start`.

## `test__create_rolling_window_data_with_both_window_boundaries(dt_transform: pd.DataFrame, mmm_data: Dict) -> None`
### USAGE
* This method tests the `_create_rolling_window_data` function with both `window_start` and `window_end` set.

### IMPL
* Simulate `dt_transform` DataFrame with "ds" and "value" columns and `mmm_data` with `mmmdata_spec` having both `window_start` and `window_end` set.
* Call `_create_rolling_window_data`.
* Assert that the output DataFrame only includes rows with "ds" between `window_start` and `window_end`.

## `test__create_rolling_window_data_with_invalid_window_boundaries(dt_transform: pd.DataFrame, mmm_data: Dict) -> None`
### USAGE
* This method tests the `_create_rolling_window_data` function when `window_start` is greater than `window_end`.

### IMPL
* Simulate `dt_transform` DataFrame with "ds" and "value" columns and `mmm_data` with `mmmdata_spec` having `window_start` greater than `window_end`.
* Call `_create_rolling_window_data`.
* Assert that the output DataFrame is empty.

## `test__calculate_media_cost_factor(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* This method tests the `_calculate_media_cost_factor` function with typical data input.

### IMPL
* Simulate `dt_input_roll_wind` DataFrame with "paid_media_1" and "paid_media_2" columns with numeric values.
* Call `_calculate_media_cost_factor`.
* Assert that the output is a Series with expected cost factors for each media column.

## `test__calculate_media_cost_factor_with_zero_spend(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* This method tests the `_calculate_media_cost_factor` function when the total spend is zero.

### IMPL
* Simulate `dt_input_roll_wind` DataFrame with "paid_media_1" and "paid_media_2" columns set to zero.
* Call `_calculate_media_cost_factor`.
* Assert that the output is a Series with `None` values for each media column.

## `test__calculate_media_cost_factor_with_large_numbers(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* This method tests the `_calculate_media_cost_factor` function with large numeric values.

### IMPL
* Simulate `dt_input_roll_wind` DataFrame with large numeric values in "paid_media_1" and "paid_media_2".
* Call `_calculate_media_cost_factor`.
* Assert that the output is a Series with expected cost factors for each media column.

## `test__calculate_media_cost_factor_with_missing_spends(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* This method tests the `_calculate_media_cost_factor` function with some missing media spends.

### IMPL
* Simulate `dt_input_roll_wind` DataFrame with "paid_media_1" and "paid_media_2" where some values are missing.
* Call `_calculate_media_cost_factor`.
* Assert that the output is a Series with expected cost factors, handling missing values correctly.

## `test__run_models(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_run_models` function with valid data and media cost factor.

### IMPL
* Mock `_fit_spend_exposure` to return a specific result for "media_var_1" and `None` for "media_var_2".
* Simulate `dt_modRollWind` with valid data and set `media_cost_factor` to 1.0.
* Call `_run_models`.
* Assert that `modNLS` contains results for "media_var_1" and not for "media_var_2".
* Assert that the structure of `modNLS["results"]` and `modNLS["plots"]` matches expected values.

## `test__run_models_with_no_media_spends(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_run_models` function when there are no media spends.

### IMPL
* Simulate `dt_modRollWind` with valid data and set `media_cost_factor` to 1.0.
* Call `_run_models`.
* Assert that `modNLS["results"]` is an empty dictionary.
* Assert that `modNLS["yhat"]` is an empty DataFrame.
* Assert that `modNLS["plots"]` is an empty dictionary.

## `test__run_models_with_multiple_media_spends(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_run_models` function with multiple media spends.

### IMPL
* Mock `_fit_spend_exposure` to return specific results for "media_var_1" and "media_var_2".
* Simulate `dt_modRollWind` with valid data and set `media_cost_factor` to 0.5.
* Call `_run_models`.
* Assert that `modNLS` contains results for both "media_var_1" and "media_var_2".
* Assert that the structure of `modNLS["results"]` and `modNLS["plots"]` matches expected values.

## `test__run_models_with_zero_media_cost_factor(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_run_models` function when the media cost factor is zero.

### IMPL
* Mock `_fit_spend_exposure` to return `None` when media cost factor is zero.
* Simulate `dt_modRollWind` with valid data and set `media_cost_factor` to 0.0.
* Call `_run_models`.
* Assert that `modNLS["results"]` is an empty dictionary.
* Assert that `modNLS["yhat"]` is an empty DataFrame.
* Assert that `modNLS["plots"]` is an empty dictionary.

## `test__fit_spend_exposure(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_fit_spend_exposure` function with valid data and when the Michaelis-Menten model is better.

### IMPL
* Simulate `dt_modRollWind` with "paid_media_1" and "exposure_1", with a relationship best fit by the Michaelis-Menten model.
* Call `_fit_spend_exposure` with `paid_media_var` as "paid_media_1" and `media_cost_factor` as 1.0.
* Assert that the `res` channel is "paid_media_1".
* Assert that the `res` model type is "nls".
* Assert that the `res` rsq is 1.0.
* Assert that `plot.shape[0]` is 5.

## `test__fit_spend_exposure_linear_model(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_fit_spend_exposure` function with valid data when the linear model is better.

### IMPL
* Simulate `dt_modRollWind` with "paid_media_1" and "exposure_1", with a linear relationship.
* Call `_fit_spend_exposure` with `paid_media_var` as "paid_media_1" and `media_cost_factor` as 1.0.
* Assert that the `res` channel is "paid_media_1".
* Assert that the `res` model type is "lm".
* Assert that the `res` rsq is 1.0.
* Assert that `plot.shape[0]` is 5.

## `test__fit_spend_exposure_zero_spend(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_fit_spend_exposure` function when there is zero spend data, leading to a fallback to the linear model.

### IMPL
* Simulate `dt_modRollWind` with "paid_media_1" all zeros and "exposure_1".
* Call `_fit_spend_exposure` with `paid_media_var` as "paid_media_1" and `media_cost_factor` as 1.0.
* Assert that the `res` channel is "paid_media_1".
* Assert that the `res` model type is "lm".
* Assert that `plot.shape[0]` is 5.

## `test__fit_spend_exposure_negative_spend(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_fit_spend_exposure` function with negative spend data, which should lead to a fallback to the linear model.

### IMPL
* Simulate `dt_modRollWind` with "paid_media_1" having negative values and "exposure_1".
* Call `_fit_spend_exposure` with `paid_media_var` as "paid_media_1" and `media_cost_factor` as 1.0.
* Assert that the `res` channel is "paid_media_1".
* Assert that the `res` model type is "lm".
* Assert that `plot.shape[0]` is 5.

## `test__fit_spend_exposure_nan_exposure_data(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict`
### USAGE
* This method tests the `_fit_spend_exposure` function with NaN in exposure data.

### IMPL
* Simulate `dt_modRollWind` with "paid_media_1" and NaN values in "exposure_1".
* Call `_fit_spend_exposure` with `paid_media_var` as "paid_media_1" and `media_cost_factor` as 1.0.
* Assert that the `res` channel is "paid_media_1".
* Assert that the `res` model type is "lm".
* Assert that `plot.shape[0]` is 3.

## `test__hill_function(x: float, alpha: float, gamma: float) -> float`
### USAGE
* This method tests the `_hill_function` with positive values for `x`, `alpha`, and `gamma`.

### IMPL
* Call `_hill_function` with `x` as 2.0, `alpha` as 1.0, and `gamma` as 1.0.
* Assert that the return value is approximately `0.6666666666666666`.

## `test__hill_function_with_zero_x(x: float, alpha: float, gamma: float) -> float`
### USAGE
* This method tests the `_hill_function` with `x` equal to zero.

### IMPL
* Call `_hill_function` with `x` as 0.0, `alpha` as 1.0, and `gamma` as 1.0.
* Assert that the return value is `0.0`.

## `test__hill_function_with_zero_alpha(x: float, alpha: float, gamma: float) -> float`
### USAGE
* This method tests the `_hill_function` with `alpha` equal to zero.

### IMPL
* Call `_hill_function` with `x` as 2.0, `alpha` as 0.0, and `gamma` as 1.0.
* Assert that the return value is `0.5`.

## `test__hill_function_with_zero_gamma(x: float, alpha: float, gamma: float) -> float`
### USAGE
* This method tests the `_hill_function` with `gamma` equal to zero.

### IMPL
* Call `_hill_function` with `x` as 2.0, `alpha` as 1.0, and `gamma` as 0.0.
* Assert that the return value is `1.0`.

## `test__hill_function_with_large_alpha_gamma(x: float, alpha: float, gamma: float) -> float`
### USAGE
* This method tests the `_hill_function` with large values for `alpha` and `gamma`.

### IMPL
* Call `_hill_function` with `x` as 2.0, `alpha` as 100.0, and `gamma` as 100.0.
* Assert that the return value is `0.5`.

## `test__hill_function_with_small_alpha(x: float, alpha: float, gamma: float) -> float`
### USAGE
* This method tests the `_hill_function` with very small values for `alpha`.

### IMPL
* Call `_hill_function` with `x` as 2.0, `alpha` as 1e-10, and `gamma` as 1.0.
* Assert that the return value is `0.5`.

## `test__hill_function_with_small_gamma(x: float, alpha: float, gamma: float) -> float`
### USAGE
* This method tests the `_hill_function` with very small values for `gamma`.

### IMPL
* Call `_hill_function` with `x` as 2.0, `alpha` as 1.0, and `gamma` as 1e-10.
* Assert that the return value is `1.0`.

## `test__prophet_decomposition(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* This method tests the `_prophet_decomposition` function with all prophet variables present.

### IMPL
* Mock `self.holidays_data.prophet_vars` to return `['trend', 'holiday', 'season', 'monthly', 'weekday']`.
* Mock `self.holidays_data.prophet_country` to return `['US']`.
* Mock `self._set_holidays` to process holidays data.
* Prepare `dt_mod` with columns "ds" and "dep_var", as well as other relevant columns.
* Call `_prophet_decomposition`.
* Assert that "trend", "season", "monthly", "weekday", and "holiday" columns in `dt_mod` contain expected arrays.

## `test__prophet_decomposition_no_holidays(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* This method tests the `_prophet_decomposition` function when no holidays are used.

### IMPL
* Mock `self.holidays_data.prophet_vars` to return `['trend', 'season', 'monthly', 'weekday']`.
* Mock `self.holidays_data.prophet_country` to return an empty list.
* Mock `self._set_holidays` to process holidays data.
* Prepare `dt_mod` with columns "ds" and "dep_var", as well as other relevant columns.
* Call `_prophet_decomposition`.
* Assert that "trend", "season", "monthly", and "weekday" columns in `dt_mod` contain expected arrays.
* Assert that the "holiday" column is `None`.

## `test__prophet_decomposition_only_trend_and_season(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* This method tests the `_prophet_decomposition` function with only "trend" and "season" used.

### IMPL
* Mock `self.holidays_data.prophet_vars` to return `['trend', 'season']`.
* Mock `self.holidays_data.prophet_country` to return `['US']`.
* Mock `self._set_holidays` to process holidays data.
* Prepare `dt_mod` with columns "ds" and "dep_var", as well as other relevant columns.
* Call `_prophet_decomposition`.
* Assert that "trend" and "season" columns in `dt_mod` contain expected arrays.
* Assert that "monthly", "weekday", and "holiday" columns are `None`.

## `test__prophet_decomposition_custom_seasonality(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* This method tests the `_prophet_decomposition` function with custom yearly and weekly seasonality parameters.

### IMPL
* Mock `self.holidays_data.prophet_vars` to return `['trend', 'holiday', 'season', 'monthly', 'weekday']`.
* Mock `self.holidays_data.prophet_country` to return `['US']`.
* Mock `self._set_holidays` to process holidays data.
* Mock `self.custom_params` with specific seasonality parameters.
* Prepare `dt_mod` with columns "ds" and "dep_var", as well as other relevant columns.
* Call `_prophet_decomposition`.
* Assert that "trend", "season", "monthly", "weekday", and "holiday" columns in `dt_mod` contain expected arrays.

## `test__set_holidays(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* This method tests the `_set_holidays` function for setting holidays in daily intervals.

### IMPL
* Simulate `dt_transform` with "ds" dates.
* Simulate `dt_holidays` with holidays data.
* Call `_set_holidays` with `interval_type` set to "day".
* Assert that the output contains expected holidays data for daily intervals.

## `test__set_holidays_weekly(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* This method tests the `_set_holidays` function for setting holidays in weekly intervals.

### IMPL
* Simulate `dt_transform` with "ds" dates.
* Simulate `dt_holidays` with holidays data that spans weeks.
* Call `_set_holidays` with `interval_type` set to "week".
* Assert that the output contains expected holidays data aggregated weekly.

## `test__set_holidays_monthly(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* This method tests the `_set_holidays` function for setting holidays in monthly intervals with valid data.

### IMPL
* Simulate `dt_transform` with "ds" dates starting from the first of each month.
* Simulate `dt_holidays` with holidays data that spans months.
* Call `_set_holidays` with `interval_type` set to "month".
* Assert that the output contains expected holidays data aggregated monthly.

## `test__set_holidays_monthly_invalid(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* This method tests the `_set_holidays` function for setting holidays in monthly intervals with invalid data.

### IMPL
* Simulate `dt_transform` with "ds" dates not starting from the first of each month.
* Simulate `dt_holidays` with holidays data.
* Call `_set_holidays` with `interval_type` set to "month" and expect a `ValueError`.
* Assert that the exception message is `"Monthly data should have first day of month as datestamp, e.g.'2020-01-01'"`.

## `test__set_holidays_invalid_interval_type(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* This method tests the `_set_holidays` function with an invalid interval type.

### IMPL
* Simulate `dt_transform` and `dt_holidays` with appropriate data.
* Call `_set_holidays` with an unsupported `interval_type` and expect a `ValueError`.
* Assert that the exception message is `"Invalid interval_type. Must be 'day', 'week', or 'month'."`.

## `test__apply_transformations(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_transformations` function with valid time series data and transformation parameters.

### IMPL
* Mock `_apply_adstock` to return a specific series of adstocked values.
* Mock `_apply_saturation` to return a specific series of saturated values.
* Simulate `x` as a `pd.Series` with numeric values.
* Simulate `params` with specific adstock and saturation parameters.
* Call `_apply_transformations`.
* Assert that `transformed_series` matches expected saturated values.

## `test__apply_transformations_with_empty_data(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_transformations` function with empty time series data.

### IMPL
* Mock `_apply_adstock` to return an empty series.
* Mock `_apply_saturation` to return an empty series.
* Simulate `x` as an empty `pd.Series`.
* Simulate `params` with specific adstock and saturation parameters.
* Call `_apply_transformations`.
* Assert that `transformed_series` is an empty `pd.Series`.

## `test__apply_transformations_with_null_params(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_transformations` function with null transformation parameters.

### IMPL
* Simulate `x` as a `pd.Series` with numeric values.
* Set `params` to `None`.
* Call `_apply_transformations` and expect a `TypeError`.
* Assert that a `TypeError` is raised due to null parameters.

## `test__apply_transformations_with_negative_values(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_transformations` function with negative values in the time series data.

### IMPL
* Mock `_apply_adstock` to return a specific series of adstocked negative values.
* Mock `_apply_saturation` to return a specific series of saturated negative values.
* Simulate `x` as a `pd.Series` with negative values.
* Simulate `params` with specific adstock and saturation parameters.
* Call `_apply_transformations`.
* Assert that `transformed_series` matches expected saturated negative values.

## `test__apply_adstock(x: pd.Series, params: ChannelHyperparameters, hyperparameters: Dict) -> pd.Series`
### USAGE
* This method tests the `_apply_adstock` function with geometric adstock application and valid parameters.

### IMPL
* Set `hyperparameters` to use `AdstockType.GEOMETRIC`.
* Simulate `x` as a `pd.Series` with numeric values.
* Simulate `params` with specific theta values.
* Call `_apply_adstock`.
* Assert that the result is a `pd.Series`.
* Assert that the length of the result is correct.

## `test__apply_adstock_weibull_cdf(x: pd.Series, params: ChannelHyperparameters, hyperparameters: Dict) -> pd.Series`
### USAGE
* This method tests the `_apply_adstock` function with Weibull CDF adstock application and valid parameters.

### IMPL
* Set `hyperparameters` to use `AdstockType.WEIBULL_CDF`.
* Simulate `x` as a `pd.Series` with numeric values.
* Simulate `params` with specific shape and scale values.
* Call `_apply_adstock`.
* Assert that the result is a `pd.Series`.
* Assert that the length of the result is correct.

## `test__apply_adstock_weibull_pdf(x: pd.Series, params: ChannelHyperparameters, hyperparameters: Dict) -> pd.Series`
### USAGE
* This method tests the `_apply_adstock` function with Weibull PDF adstock application and valid parameters.

### IMPL
* Set `hyperparameters` to use `AdstockType.WEIBULL_PDF`.
* Simulate `x` as a `pd.Series` with numeric values.
* Simulate `params` with specific shape and scale values.
* Call `_apply_adstock`.
* Assert that the result is a `pd.Series`.
* Assert that the length of the result is correct.

## `test__apply_adstock_unsupported_type(x: pd.Series, params: ChannelHyperparameters, hyperparameters: Dict) -> pd.Series`
### USAGE
* This method tests the `_apply_adstock` function with an unsupported adstock type, expecting a `ValueError`.

### IMPL
* Set `hyperparameters` to use an unsupported `AdstockType.UNKNOWN`.
* Simulate `x` as a `pd.Series` with numeric values.
* Simulate `params` with specific theta values.
* Call `_apply_adstock` and expect a `ValueError`.
* Assert that the exception message is `"Unsupported adstock type: AdstockType.UNKNOWN"`.

## `test__geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function with regular time series data and `theta` within a normal range.

### IMPL
* Simulate `x` as a `pd.Series` with numeric values.
* Set `theta` to 0.5.
* Call `_geometric_adstock`.
* Assert that `computed_series` matches expected values.

## `test__geometric_adstock_with_theta_zero(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function with `theta` at the boundary value (0).

### IMPL
* Simulate `x` as a `pd.Series` with numeric values.
* Set `theta` to 0.0.
* Call `_geometric_adstock`.
* Assert that `computed_series` matches expected constant values.

## `test__geometric_adstock_with_theta_one(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function with `theta` at the boundary value (1).

### IMPL
* Simulate `x` as a `pd.Series` with numeric values.
* Set `theta` to 1.0.
* Call `_geometric_adstock`.
* Assert that `computed_series` matches the original series.

## `test__geometric_adstock_with_empty_series(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function with an empty series.

### IMPL
* Simulate `x` as an empty `pd.Series`.
* Set `theta` to 0.5.
* Call `_geometric_adstock`.
* Assert that `computed_series` is an empty `pd.Series`.

## `test__geometric_adstock_with_single_element_series(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function with a single-element series.

### IMPL
* Simulate `x` as a `pd.Series` with a single numeric value.
* Set `theta` to 0.5.
* Call `_geometric_adstock`.
* Assert that `computed_series` matches the original single-element series.

## `test__geometric_adstock_with_decreasing_series(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function with decreasing time series data.

### IMPL
* Simulate `x` as a `pd.Series` with decreasing numeric values.
* Set `theta` to 0.7.
* Call `_geometric_adstock`.
* Assert that `computed_series` matches expected values.

## `test__geometric_adstock_with_theta_slightly_above_zero(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* This method tests the `_geometric_adstock` function with `theta` slightly above 0.

### IMPL
* Simulate `x` as a `pd.Series` with numeric values.
* Set `theta` to 0.01.
* Call `_geometric_adstock`.
* Assert that `computed_series` matches expected values.

## `test__weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* This method tests the `_weibull_adstock` function with a simple increasing series.

### IMPL
* Simulate `x` as a `pd.Series` with increasing numeric values.
* Set `shape` to 1.5 and `scale` to 1.0.
* Call `_weibull_adstock`.
* Assert that the output is a `pd.Series`.
* Assert that the length of the output is correct.
* Assert that the first and last elements of the output are positive floats.

## `test__weibull_adstock_with_constant_series(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* This method tests the `_weibull_adstock` function with a constant series.

### IMPL
* Simulate `x` as a `pd.Series` with constant numeric values.
* Set `shape` to 1.5 and `scale` to 1.0.
* Call `_weibull_adstock`.
* Assert that the output is a `pd.Series`.
* Assert that the length of the output is correct.
* Assert that the first and last elements of the output are positive floats.

## `test__weibull_adstock_with_zero_series(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* This method tests the `_weibull_adstock` function with a series of zeros.

### IMPL
* Simulate `x` as a `pd.Series` of zeros.
* Set `shape` to 1.5 and `scale` to 1.0.
* Call `_weibull_adstock`.
* Assert that the output is a `pd.Series`.
* Assert that the length of the output is correct.
* Assert that all elements of the output are zero.

## `test__weibull_adstock_with_decreasing_series(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* This method tests the `_weibull_adstock` function with a decreasing series.

### IMPL
* Simulate `x` as a `pd.Series` with decreasing numeric values.
* Set `shape` to 1.5 and `scale` to 1.0.
* Call `_weibull_adstock`.
* Assert that the output is a `pd.Series`.
* Assert that the length of the output is correct.
* Assert that the first and last elements of the output are positive floats.

## `test__weibull_adstock_with_shape_zero(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* This method tests the `_weibull_adstock` function with the shape parameter as zero.

### IMPL
* Simulate `x` as a `pd.Series` with numeric values.
* Set `shape` to 0.0 and `scale` to 1.0.
* Call `_weibull_adstock`.
* Assert that the output is a `pd.Series`.
* Assert that the length of the output is correct.
* Assert that all elements of the output are close to zero.

## `test__weibull_adstock_with_scale_zero(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* This method tests the `_weibull_adstock` function with the scale parameter as zero.

### IMPL
* Simulate `x` as a `pd.Series` with numeric values.
* Set `shape` to 1.5 and `scale` to 0.0.
* Call `_weibull_adstock` and expect a `ZeroDivisionError`.
* Assert that a `ZeroDivisionError` is raised due to zero scale.

## `test__weibull_adstock_with_empty_series(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* This method tests the `_weibull_adstock` function with an empty series.

### IMPL
* Simulate `x` as an empty `pd.Series`.
* Set `shape` to 1.5 and `scale` to 1.0.
* Call `_weibull_adstock`.
* Assert that the output is a `pd.Series`.
* Assert that the length of the output is zero.

## `test__apply_saturation(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_saturation` function with positive values and valid parameters.

### IMPL
* Simulate `x` as a `pd.Series` with positive numeric values.
* Simulate `params` with specific alphas and gammas.
* Call `_apply_saturation`.
* Assert that the `output` matches expected saturated values.

## `test__apply_saturation_with_zero_values(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_saturation` function with zero values in `x`.

### IMPL
* Simulate `x` as a `pd.Series` of zeros.
* Simulate `params` with specific alphas and gammas.
* Call `_apply_saturation`.
* Assert that the `output` is a `pd.Series` of zeros.

## `test__apply_saturation_with_negative_values(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_saturation` function with negative values in `x`.

### IMPL
* Simulate `x` as a `pd.Series` with negative numeric values.
* Simulate `params` with specific alphas and gammas.
* Call `_apply_saturation`.
* Assert that the `output` matches expected saturated negative values.

## `test__apply_saturation_with_large_values(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* This method tests the `_apply_saturation` function with very large values in `x`.

### IMPL
* Simulate `x` as a `pd.Series` with very large numeric values.
* Simulate `params` with specific alphas and gammas.
* Call `_apply_saturation`.
* Assert that the `output` is a `pd.Series` with all elements close to 1.0.