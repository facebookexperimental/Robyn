# CLASS
## ParetoOptimizer
* Performs Pareto optimization on marketing mix models.
* This class orchestrates the Pareto optimization process, including data aggregation, Pareto front calculation, response curve calculation, and plot data preparation.
* Attributes:
  * `mmm_data (MMMData)`: Input data for the marketing mix model.
  * `model_outputs (ModelOutputs)`: Output data from the model runs.
  * `response_calculator (ResponseCurveCalculator)`: Calculator for response curves.
  * `carryover_calculator (ImmediateCarryoverCalculator)`: Calculator for immediate and carryover effects.
  * `pareto_utils (ParetoUtils)`: Utility functions for Pareto-related calculations.

# CONSTRUCTORS
## ParetoOptimizer `(mmm_data: MMMData, model_outputs: ModelOutputs, hyper_parameter: Hyperparameters, featurized_mmm_data: FeaturizedMMMData, holidays_data: HolidaysData)`
* Use this constructor when you need to perform Pareto optimization on marketing mix models using the specified inputs.