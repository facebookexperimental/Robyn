# model_evaluation.py
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from robyn.modeling.entities.modeloutput import ModelOutput


class ModelEvaluator:
    def __init__(self):
        pass

    def calculate_rsquared(
        self,
        true: np.ndarray,
        predicted: np.ndarray,
        p: int,
        df_int: int,
        n_train: Optional[int] = None,
    ) -> float:
        """
        Calculate the R-squared value.

        :param true: True values
        :param predicted: Predicted values
        :param p: Number of predictors
        :param df_int: Degrees of freedom for intercept
        :param n_train: Number of training samples (optional)
        :return: R-squared value
        """
        residuals = true - predicted
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((true - np.mean(true)) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        # Adjust R-squared if n_train is provided
        if n_train is not None:
            adjusted_r_squared = 1 - (
                (1 - r_squared) * (n_train - df_int) / (n_train - p - df_int)
            )
            return adjusted_r_squared

        return r_squared

    def calculate_nrmse(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate the Normalized Root Mean Square Error (NRMSE).

        :param true: True values
        :param predicted: Predicted values
        :return: NRMSE value
        """
        rmse = np.sqrt(np.mean((true - predicted) ** 2))
        range_true = np.max(true) - np.min(true)
        return rmse / range_true if range_true != 0 else np.inf

    def calculate_mape(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE).

        :param true: True values
        :param predicted: Predicted values
        :return: MAPE value
        """
        return np.mean(np.abs((true - predicted) / true)) * 100

    def calculate_decomp_rssd(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate the Residual Sum of Squares Decomposition (RSSD).

        :param true: True values
        :param predicted: Predicted values
        :return: RSSD value
        """
        return np.sum((true - predicted) ** 2) / len(true)

    def evaluate_model(self, model_output: ModelOutput) -> Dict[str, float]:
        """
        Evaluate the model using multiple metrics from the ModelOutput object.

        :param model_output: ModelOutput object containing model results
        :return: Dictionary containing evaluation metrics
        """
        metrics = {}
        for trial in model_output.trials:
            metrics[trial.solID] = {
                "nrmse": trial.nrmse,
                "mape": trial.mape,
                "rsq_train": trial.rsq_train,
                "rsq_val": trial.rsq_val,
                "rsq_test": trial.rsq_test,
                "nrmse_train": trial.nrmse_train,
                "nrmse_val": trial.nrmse_val,
                "nrmse_test": trial.nrmse_test,
                "decomp_rssd": trial.decomp_rssd,
            }

        # Calculate average metrics across all trials
        avg_metrics = {
            "avg_nrmse": np.mean([m["nrmse"] for m in metrics.values()]),
            "avg_mape": np.mean([m["mape"] for m in metrics.values()]),
            "avg_rsq_train": np.mean([m["rsq_train"] for m in metrics.values()]),
            "avg_rsq_val": np.mean([m["rsq_val"] for m in metrics.values()]),
            "avg_rsq_test": np.mean([m["rsq_test"] for m in metrics.values()]),
            "avg_nrmse_train": np.mean([m["nrmse_train"] for m in metrics.values()]),
            "avg_nrmse_val": np.mean([m["nrmse_val"] for m in metrics.values()]),
            "avg_nrmse_test": np.mean([m["nrmse_test"] for m in metrics.values()]),
            "avg_decomp_rssd": np.mean([m["decomp_rssd"] for m in metrics.values()]),
        }

        return {"per_trial_metrics": metrics, "average_metrics": avg_metrics}

    def cross_validate(
        self, model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.

        :param model: The model to be evaluated
        :param X: Feature matrix
        :param y: Target variable
        :param cv: Number of cross-validation folds
        :return: Dictionary containing cross-validation results
        """
        # Implement cross-validation logic here
        # This is a placeholder and should be implemented based on your specific requirements
        return {"r_squared": [], "nrmse": [], "mape": []}

    def plot_actual_vs_predicted(self, true: np.ndarray, predicted: np.ndarray) -> Any:
        """
        Create a plot of actual vs predicted values.

        :param true: True values
        :param predicted: Predicted values
        :return: Plot object
        """
        # Implement plotting logic here
        # This is a placeholder and should be implemented based on your specific requirements
        pass

    def plot_residuals(self, true: np.ndarray, predicted: np.ndarray) -> Any:
        """
        Create a plot of residuals.

        :param true: True values
        :param predicted: Predicted values
        :return: Plot object
        """
        # Implement residual plotting logic here
        # This is a placeholder and should be implemented based on your specific requirements
        pass


# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator()

    # Example data
    true_values = np.array([1, 2, 3, 4, 5])
    predicted_values = np.array([1.1, 2.2, 2.9, 3.8, 5.2])

    # Calculate individual metrics
    r_squared = evaluator.calculate_rsquared(
        true_values, predicted_values, p=1, df_int=1
    )
    nrmse = evaluator.calculate_nrmse(true_values, predicted_values)
    mape = evaluator.calculate_mape(true_values, predicted_values)

    print(f"R-squared: {r_squared}")
    print(f"NRMSE: {nrmse}")
    print(f"MAPE: {mape}")

    # Evaluate model
    evaluation_results = evaluator.evaluate_model(
        true_values, predicted_values, p=1, df_int=1
    )
    print("Evaluation results:", evaluation_results)
