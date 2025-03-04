# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from sklearn import datasets

# Activate automatic conversion between R and NumPy arrays
numpy2ri.activate()

# Import required R packages
base = importr("base")
glmnet = importr("glmnet")


# Create wrapper function similar to Robyn's create_ridge_model
def create_ridge_model_r(lambda_value, X, y, fit_intercept=True, standardize=True):
    """Use R's glmnet to create and fit a ridge regression model.

    Args:
        lambda_value: Regularization parameter (lambda)
        X: Input features (numpy array)
        y: Target values (numpy array)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features

    Returns:
        Fitted R glmnet model
    """
    print(f"Creating R glmnet Ridge model - lambda: {lambda_value:.6f}")
    print(f"Intercept fitting: {fit_intercept}, Standardization: {standardize}")

    # Convert to R objects
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_X = ro.conversion.py2rpy(X)
        r_y = ro.conversion.py2rpy(y)

    # Convert X to R matrix explicitly
    r_matrix = ro.r["matrix"]
    x_train = r_matrix(r_X, nrow=X.shape[0], ncol=X.shape[1])

    # Create penalty_factor (1 for each feature)
    penalty_factor = ro.FloatVector([1.0] * X.shape[1])

    # Fit glmnet model
    model = glmnet.glmnet(
        x=x_train,
        y=r_y,
        family="gaussian",
        alpha=0,  # 0 for ridge regression
        lambda_=ro.FloatVector([lambda_value]),
        penalty_factor=penalty_factor,
        intercept=fit_intercept,
        standardize=standardize,
    )

    return model, x_train


def get_coefficients(model):
    """Extract coefficients from an R glmnet model.

    Args:
        model: Fitted R glmnet model

    Returns:
        NumPy array of coefficients
    """
    # Extract beta (coefficients) from the model
    beta = model.rx2("beta")

    # Convert to numpy array
    with localconverter(ro.default_converter + numpy2ri.converter):
        coefficients = ro.conversion.rpy2py(beta)

    # If intercept is included, get it
    intercept = 0
    if model.rx2("intercept"):
        intercept = model.rx2("a0")[0]

    return coefficients, intercept


def predict(model, X):
    """Make predictions using an R glmnet model.

    Args:
        model: Fitted R glmnet model
        X: Input features as R matrix

    Returns:
        NumPy array of predictions
    """
    # Use R's predict function
    predictions = glmnet.predict_glmnet(model, X, s=model.rx2("lambda")[0])

    # Convert to numpy array
    with localconverter(ro.default_converter + numpy2ri.converter):
        predictions_np = ro.conversion.rpy2py(predictions)

    return predictions_np
