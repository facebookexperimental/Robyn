# CLASS
## HillCalculator
* The `HillCalculator` class is responsible for computing the necessary parameters for Hill function modeling, particularly in the domain of marketing mix modeling (MMM).
* It processes input data related to MMM and the output from models to derive the required alphas, inflexions, and sorted coefficients.
* Utilizes Pandas DataFrames and NumPy libraries for data manipulation and mathematical operations.

# CONSTRUCTORS
## HillCalculator `(mmmdata: MMMData, model_outputs: ModelOutputs, dt_hyppar: pd.DataFrame, dt_coef: pd.DataFrame, media_spend_sorted: List[str], select_model: str, chn_adstocked: pd.DataFrame = None)`
* **mmmdata**: This is an instance of the `MMMData` class, which includes the marketing mix modeling data.
* **model_outputs**: An instance of `ModelOutputs` that contains results from the model execution.
* **dt_hyppar**: A Pandas DataFrame that holds hyperparameter data, specifically alphas and gammas.
* **dt_coef**: A Pandas DataFrame that includes coefficients for different media channels.
* **media_spend_sorted**: A list of strings representing the sorted order of media spending channels.
* **select_model**: A string specifying the chosen model ID for filtering purposes.
* **chn_adstocked**: An optional DataFrame for precomputed adstocked media data; if not provided, it will be computed later.

### USAGE
* This constructor is used to initialize a `HillCalculator` instance with all the essential data for performing calculations.
* It should be employed when you need to compute Hill parameters based on a specific set of MMM data and model outputs.

### IMPL
* Initializes class attributes `mmmdata`, `model_outputs`, `dt_hyppar`, `dt_coef`, `media_spend_sorted`, `select_model`, and `chn_adstocked`.
* If `chn_adstocked` is not provided, it will be computed using the `_get_chn_adstocked_from_output_collect` method at a later stage.

# METHODS
## `_get_chn_adstocked_from_output_collect() -> pd.DataFrame`
### USAGE
* This method retrieves adstocked media data from the `model_outputs` if `chn_adstocked` is not precomputed.
* It filters and slices the `media_vec_collect` DataFrame to acquire adstocked media data for the selected model.

### IMPL
* Filters the `media_vec_collect` DataFrame from `model_outputs` to include only rows where the 'type' column is 'adstockedMedia' and 'solID' matches `select_model`.
* Selects columns based on the `media_spend_sorted` list to ensure only relevant media channels are included.
* Slices the DataFrame according to the `window_start` and `window_end` indices specified in `mmmdata.mmmdata_spec`.
* Returns the filtered and sliced DataFrame.

## `get_hill_params() -> Dict[str, Any]`
### USAGE
* Calculates Hill parameters: alphas, inflexions, and sorted coefficients, which are crucial for constructing Hill functions within the MMM framework.
* This method leverages input data and may utilize `_get_chn_adstocked_from_output_collect` if `chn_adstocked` was not initially provided.

### IMPL
* Extracts alphas and gammas for each media channel from `dt_hyppar` by using a regex pattern to match column names `*_alphas` and `*_gammas`.
* If `chn_adstocked` is `None`, invokes `_get_chn_adstocked_from_output_collect()` to calculate it.
* Computes inflexions for each media channel by performing a weighted summation of the minimum and maximum values in the `chn_adstocked` DataFrame using the retrieved gammas.
* Retrieves coefficients from `dt_coef` and sorts them according to the order specified in `media_spend_sorted`.
* Returns a dictionary containing:
  - A list of `alphas`.
  - A list of computed `inflexions`.
  - A list of `coefs_sorted` based on the sorted media spend.