# CLASS
## HillCalculator
* The `HillCalculator` class is designed to compute parameters related to the Hill function for marketing mix models (MMM). It processes input data to determine specific attributes such as alphas, inflexions, and sorted coefficients, which are vital for modeling media spend effects.
* This class is part of a module that likely interacts with data entities like `MMMData` and `ModelOutputs` to perform its operations.
* The class encapsulates functionality to extract adstocked media data and calculate Hill function parameters.

# CONSTRUCTORS
## `__init__` `(mmmdata: MMMData, model_outputs: ModelOutputs, dt_hyppar: pd.DataFrame, dt_coef: pd.DataFrame, media_spend_sorted: List[str], select_model: str, chn_adstocked: pd.DataFrame = None)`
* **Parameters:**
  * `mmmdata (MMMData)`: An object containing the marketing mix model data specifications, including the window start and end indices for data slicing.
  * `model_outputs (ModelOutputs)`: An object containing model outputs, specifically the media vector collections needed for adstocked media calculations.
  * `dt_hyppar (pd.DataFrame)`: A DataFrame holding the hyperparameters required for calculating the Hill function parameters, namely alphas and gammas.
  * `dt_coef (pd.DataFrame)`: A DataFrame that contains coefficients for the models, used in sorting and extracting relevant data.
  * `media_spend_sorted (List[str])`: A sorted list of media spend column names that define the order of processing and output generation.
  * `select_model (str)`: A string identifier for the selected model, which is used to filter the relevant data from the model outputs.
  * `chn_adstocked (pd.DataFrame, optional)`: Optional precomputed DataFrame of adstocked media, which can be provided to skip recalculation.

### USAGE
* Use this constructor when creating an instance of `HillCalculator` by providing all necessary data dependencies and configurations. It initializes the object with data and parameters required for subsequent calculations.
* This setup is essential before invoking methods like `_get_chn_adstocked_from_output_collect` and `get_hill_params`, as it ensures the class has all the necessary context and data to function correctly.

### IMPL
* Ensure that upon instantiation, the constructor initializes all attributes with the provided parameters, setting up the internal state of the `HillCalculator` object.
* The constructor should not perform any computations besides storing the input data; it merely prepares the object for later method calls.
* In testing, verify that all attributes are correctly assigned and that the optional parameter `chn_adstocked` defaults appropriately if not provided.

# METHODS


## `test_empty_media_vec_collect() -> None`
### USAGE
* This test checks the behavior of the `_get_chn_adstocked_from_output_collect` function when the `media_vec_collect` is empty.
* It ensures that the function returns an empty DataFrame as expected.
### IMPL
* Mock the `media_vec_collect` method of `ModelOutputs` to return an empty DataFrame.
* Prepare the input data with `mmmdata` specification having a window from 0 to 10.
* Ensure `model_outputs` has an empty DataFrame for `media_vec_collect`.
* Set `media_spend_sorted` as an empty list.
* Set `select_model` to a dummy value like 'model_1'.
* Invoke `_get_chn_adstocked_from_output_collect`.
* Assert that the result `chn_adstocked` is an empty DataFrame.

## `test_no_matching_solID() -> None`
### USAGE
* This test checks the behavior when there is no matching `solID` in `media_vec_collect`.
* It ensures that the function returns an empty DataFrame when the selected model does not match any records.
### IMPL
* Mock the `media_vec_collect` method of `ModelOutputs` to return a DataFrame with non-matching solIDs.
* Prepare the input data with `mmmdata` specification having a window from 0 to 10.
* Ensure `model_outputs` has the mocked DataFrame.
* Set `media_spend_sorted` as `['media_1', 'media_2']`.
* Set `select_model` to 'model_1', which does not match any `solID`.
* Invoke `_get_chn_adstocked_from_output_collect`.
* Assert that the result `chn_adstocked` is an empty DataFrame.

## `test_valid_matching_solID_and_window_slicing() -> None`
### USAGE
* This test checks the function behavior with valid matching `solID` and proper window slicing.
* It ensures that the function returns a correctly sliced DataFrame.
### IMPL
* Mock the `media_vec_collect` method of `ModelOutputs` to return a DataFrame with matching `solID` values.
* Prepare the input data with `mmmdata` specification having a window from 1 to 2.
* Ensure `model_outputs` has the mocked DataFrame.
* Set `media_spend_sorted` as `['media_1', 'media_2']`.
* Set `select_model` to 'model_1'.
* Invoke `_get_chn_adstocked_from_output_collect`.
* Assert that the result `chn_adstocked` matches the expected DataFrame with sliced data for indices 1 to 2.

## `test_non_existing_media_columns() -> None`
### USAGE
* This test checks the behavior when `media_spend_sorted` contains non-existing media columns.
* It ensures that the function returns an empty DataFrame.
### IMPL
* Mock the `media_vec_collect` method of `ModelOutputs` to return a DataFrame with existing media columns.
* Prepare the input data with `mmmdata` specification having a window from 0 to 1.
* Ensure `model_outputs` has the mocked DataFrame.
* Set `media_spend_sorted` to non-existing columns `['media_3', 'media_4']`.
* Set `select_model` to 'model_1'.
* Invoke `_get_chn_adstocked_from_output_collect`.
* Assert that the result `chn_adstocked` is an empty DataFrame.

## `test_get_hill_params_with_normal_input_data() -> None`
### USAGE
* This test checks the behavior of `get_hill_params` when provided with normal input data.
* Mocks are used for the `pd.DataFrame` methods `filter` and `agg` to simulate the behavior of extracting and aggregating data.
* The parameters include a populated `media_spend_sorted` list and valid data objects for `MMMData`, `ModelOutputs`, `dt_hyppar`, `dt_coef`, and `chn_adstocked`.

### IMPL
1. **Mock Setup:**
   - Mock the `filter` method of `pd.DataFrame` to return a DataFrame with specific alphas and gammas.
   - Mock the `agg` method of `pd.DataFrame` to simulate the `min` and `max` aggregation for each media.

2. **Input Preparation:**
   - Create mock instances of `MMMData`, `ModelOutputs`, `dt_hyppar`, `dt_coef`, and `chn_adstocked`.
   - Populate `media_spend_sorted` with media identifiers (e.g., `['media1', 'media2']`).

3. **Invoke Function:**
   - Call `get_hill_params` with the prepared inputs.

4. **Assertions:**
   - Assert that the returned `alphas` match the expected values `[0.5, 0.7]`.
   - Assert that calculated `inflexions` are `[170.0, 190.0]`. This involves computing weighted sums based on mocked `agg` outputs and `gammas`.
   - Verify that `coefs_sorted` matches the expected order `['coef1', 'coef2']`.

## `test_get_hill_params_with_chn_adstocked_none() -> None`
### USAGE
* This test evaluates `get_hill_params` when `chn_adstocked` is initially `None`.
* It mocks the method `_get_chn_adstocked_from_output_collect` to supply a DataFrame for `chn_adstocked`.

### IMPL
1. **Mock Setup:**
   - Mock the `_get_chn_adstocked_from_output_collect` method to return a DataFrame containing media spend data.

2. **Input Preparation:**
   - Assign `None` to `chn_adstocked` to test its handling within the function.
   - Prepare other inputs similarly to the previous test case.

3. **Invoke Function:**
   - Execute `get_hill_params` with the prepared inputs.

4. **Assertions:**
   - Confirm the `alphas` returned are `[0.5, 0.7]`, matching expected values from the mock setup.
   - Ensure `inflexions` are computed as `[170.0, 190.0]` through the mocked aggregation and `gammas`.
   - Check that `coefs_sorted` maintains the correct order `['coef1', 'coef2']`.

## `test_get_hill_params_with_empty_media_spend_sorted() -> None`
### USAGE
* This test examines the function's response when `media_spend_sorted` is empty.
* No mocks are needed, as the method should gracefully handle the empty list.

### IMPL
1. **Input Preparation:**
   - Use empty list `[]` for `media_spend_sorted`.
   - Provide valid instances of other inputs (`MMMData`, `ModelOutputs`, `dt_hyppar`, `dt_coef`, and `chn_adstocked`).

2. **Invoke Function:**
   - Call `get_hill_params` with these inputs.

3. **Assertions:**
   - Assert that `alphas` is an empty list `[]`, as there are no media to extract parameters for.
   - Confirm `inflexions` returns an empty list `[]`, given no media spend data to compute.
   - Verify `coefs_sorted` is also an empty list `[]`, due to the absence of media coefficients.