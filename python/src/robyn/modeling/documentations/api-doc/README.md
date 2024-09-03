# API Documentation for MMM Modeling Component

## MMMModelExecutor

The main class for executing the MMM (Marketing Mix Modeling) process.

### Methods

#### model_run

```python
def model_run(
    self,
    mmmdata_collection: MMMDataCollection,
    trials_config: TrialsConfig,
    seed: int = 123,
    quiet: bool = False
) -> ModelOutputCollection:
```

Runs the MMM model with the given data and configuration.

- `mmmdata_collection`: An instance of `MMMDataCollection` containing all necessary data.
- `trials_config`: An instance of `TrialsConfig` specifying the number of trials and iterations.
- `seed`: Random seed for reproducibility.
- `quiet`: If True, suppresses output messages.

Returns a `ModelOutputCollection` containing the model output.

## ParetoOptimizer

Class for optimizing the Pareto front of model solutions.

### Methods

#### pareto_optimize

```python
def pareto_optimize(
    self,
    mmmdata_collection: MMMDataCollection,
    modeloutput: ModelOutputCollection,
    pareto_fronts: Union[str, int] = "auto",
    min_candidates: int = 100
) -> Dict[str, Any]:
```

Optimizes the Pareto front of model solutions.

- `mmmdata_collection`: An instance of `MMMDataCollection`.
- `modeloutput`: The `ModelOutputCollection` from the model run.
- `pareto_fronts`: Number of Pareto fronts to consider or "auto".
- `min_candidates`: Minimum number of candidates to consider.

Returns a dictionary containing Pareto-optimal solutions and related information.

## ModelClustersAnalyzer

Class for analyzing clusters of models.

### Methods

#### model_clusters_analyze

```python
def model_clusters_analyze(
    self,
    input_data: ModelOutput,
    dep_var_type: str,
    cluster_by: str = "hyperparameters",
    k: Union[str, int] = "auto",
    quiet: bool = True
) -> Optional[pd.DataFrame]:
```

Analyzes clusters of models based on their characteristics.

- `input_data`: The `ModelOutput` to analyze.
- `dep_var_type`: Type of the dependent variable.
- `cluster_by`: The characteristic to cluster by.
- `k`: Number of clusters or "auto".
- `quiet`: If True, suppresses output messages.

Returns a DataFrame with cluster analysis results, or None if clustering couldn't be performed.

## ModelEvaluator

Class for evaluating model performance.

### Methods

#### evaluate_model

```python
def evaluate_model(self, model_output: ModelOutput) -> Dict[str, Any]:
```

Evaluates the performance of the model.

- `model_output`: The `ModelOutput` to evaluate.

Returns a dictionary containing average metrics across all trials and metrics for each trial.

## ModelRefresh

Class for refreshing the model with new data.

### Methods

#### model_refresh

```python
def model_refresh(
    self,
    mmmdata_collection: MMMDataCollection,
    model_output_collection: ModelOutputCollection,
    refresh_config: ModelRefreshConfig,
    calibration_input: Optional[CalibrationInput] = None,
    objective_weights: Optional[Dict[str, float]] = None
) -> Any:
```

Refreshes the model with new data.

- `mmmdata_collection`: Updated `MMMDataCollection`.
- `model_output_collection`: Previous `ModelOutputCollection`.
- `refresh_config`: Configuration for the refresh process.
- `calibration_input`: Optional calibration input.
- `objective_weights`: Optional weights for objectives.

Returns the refreshed model output.
