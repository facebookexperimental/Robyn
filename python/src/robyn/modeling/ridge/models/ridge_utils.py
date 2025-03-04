import logging
import numpy as np
from sklearn.linear_model import Ridge


def create_ridge_model_sklearn(
    lambda_value, n_samples, fit_intercept=True, standardize=True
):
    """Create a Ridge regression model using scikit-learn.

    Args:
        lambda_value: Regularization parameter (lambda) from glmnet
        n_samples: Number of samples (needed for proper scaling)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features

    Returns:
        A configured sklearn Ridge model that behaves like glmnet
    """

    # Create a wrapper class that matches glmnet's behavior
    class GlmnetLikeRidge:
        def __init__(self):
            self.feature_means = None
            self.feature_stds = None
            self.y_mean = None
            self.coef_ = None
            self.intercept_ = 0.0
            # Lambda value used for regularization
            self.lambda_value = lambda_value
            self.fit_intercept = fit_intercept
            self.standardize = standardize

        def fit(self, X, y):
            # Make copies to avoid modifying original data
            X_processed = X.copy()
            y_processed = y.copy()

            # Center y if fitting intercept
            if self.fit_intercept:
                self.y_mean = np.mean(y_processed)
                y_processed = y_processed - self.y_mean
            else:
                self.y_mean = 0.0

            # Standardize X if needed
            if self.standardize:
                self.feature_means = np.mean(X_processed, axis=0)
                self.feature_stds = np.std(X_processed, axis=0, ddof=1)  # R uses ddof=1
                # Avoid division by zero
                self.feature_stds[self.feature_stds == 0] = 1.0
                X_processed = (X_processed - self.feature_means) / self.feature_stds
            else:
                self.feature_means = np.zeros(X_processed.shape[1])
                self.feature_stds = np.ones(X_processed.shape[1])

            # ADJUST THE LAMBDA VALUE based on StackExchange findings
            # From the top answer: To match glmnet, use λ = 2*α_sklearn/N
            # So working backwards, α_sklearn = λ*N/2
            # This accounts for how both libraries scale the regularization term
            sklearn_alpha = self.lambda_value * n_samples / 2

            # Use cholesky solver for better numerical stability, same as glmnet
            model = Ridge(alpha=sklearn_alpha, fit_intercept=False, solver="cholesky")
            model.fit(X_processed, y_processed)

            # Extract coefficients and adjust for standardization
            self.coef_ = model.coef_.copy()

            if self.standardize:
                # Scale coefficients back to original scale
                self.coef_ = self.coef_ / self.feature_stds

            # Calculate intercept manually (similar to how glmnet does it)
            if self.fit_intercept:
                self.intercept_ = self.y_mean - np.dot(self.feature_means, self.coef_)
            else:
                self.intercept_ = 0.0

            return self

        def predict(self, X):
            if self.coef_ is None:
                raise ValueError("Model must be fitted before making predictions")

            # Direct prediction using coefficients and intercept
            return np.dot(X, self.coef_) + self.intercept_

    return GlmnetLikeRidge()


def create_ridge_model_rpy2(
    lambda_value,
    n_samples,
    fit_intercept=True,
    standardize=True,
    lower_limits=None,
    upper_limits=None,
    penalty_factor=None,
    **kwargs,  # This catches any additional arguments
):
    """Create a Ridge regression model using R's glmnet via rpy2.

    This implementation exactly matches R's glmnet behavior by directly
    calling the R functions through rpy2.

    Args:
        lambda_value: Regularization parameter (lambda)
        n_samples: Number of samples (not directly used for glmnet)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features
        lower_limits: Lower bounds for coefficients (default None)
        upper_limits: Upper bounds for coefficients (default None)
        penalty_factor: Custom penalty factors for variables (default None)
        **kwargs: Additional arguments to pass to glmnet

    Returns:
        A model object with fit() and predict() methods
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    # Activate automatic conversion between R and numpy
    numpy2ri.activate()

    # Import necessary R packages
    utils = importr("utils")

    # Make sure the glmnet package is available
    try:
        ro.r("library(glmnet)")
    except Exception as e:
        print(f"Error loading glmnet: {e}")
        print("Installing glmnet package...")
        utils.install_packages("glmnet")
        ro.r("library(glmnet)")

    class GlmnetRidgeWrapper:
        def __init__(self):
            self.fitted_model = None
            self.coef_ = None
            self.intercept_ = 0.0
            self.lambda_value = lambda_value
            self.kwargs = kwargs

        def fit(self, X, y):
            # Ensure numpy arrays
            X = np.asarray(X)
            y = np.asarray(y)

            # Convert to R matrices directly using R code
            with localconverter(ro.default_converter + numpy2ri.converter):
                # Pass the data to R environment
                ro.r.assign("X_r", X)
                ro.r.assign("y_r", y)
                ro.r.assign("lambda_value", self.lambda_value)

                # Set up parameters
                fit_intercept_r = "TRUE" if fit_intercept else "FALSE"
                standardize_r = "TRUE" if standardize else "FALSE"

                # Prepare R command for optional parameters
                optional_params = []

                if penalty_factor is not None:
                    ro.r.assign("penalty_factor_r", penalty_factor)
                    optional_params.append("penalty.factor = penalty_factor_r")

                if lower_limits is not None:
                    ro.r.assign("lower_limits_r", lower_limits)
                    optional_params.append("lower.limits = lower_limits_r")

                if upper_limits is not None:
                    ro.r.assign("upper_limits_r", upper_limits)
                    optional_params.append("upper.limits = upper_limits_r")

                # Add any additional parameters
                for k, v in self.kwargs.items():
                    if v is not None:
                        k_r = k.replace("_", ".")
                        ro.r.assign(f"{k}_param", v)
                        optional_params.append(f"{k_r} = {k}_param")

                # Join optional parameters
                optional_str = ", ".join(optional_params)
                if optional_str:
                    optional_str = ", " + optional_str

                # Fit the model using direct R code
                r_code = f"""
                # Use global assignment operator to ensure objects persist
                r_model <<- glmnet(
                    x = X_r,
                    y = y_r,
                    family = "gaussian",
                    alpha = 0,  # 0 for ridge regression
                    lambda = lambda_value,
                    standardize = {standardize_r},
                    intercept = {fit_intercept_r},
                    type.measure = "mse"{optional_str}
                )
                coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                """
                ro.r(r_code)

                # Get the model and coefficients from R
                self.fitted_model = ro.r["r_model"]
                coef_array = np.array(ro.r["coef_values"])

                # First coefficient is intercept, rest are feature coefficients
                if fit_intercept:
                    self.intercept_ = float(coef_array[0])
                    self.coef_ = coef_array[1:]
                else:
                    self.intercept_ = 0.0
                    self.coef_ = coef_array[1:]  # Still skip first element as it's 0

            return self

        def predict(self, X):
            if self.fitted_model is None:
                raise ValueError("Model must be fitted before making predictions")

            # Ensure numpy array
            X = np.asarray(X)

            # Make predictions using R code directly
            with localconverter(ro.default_converter + numpy2ri.converter):
                # Pass the data to R environment
                ro.r.assign("X_new", X)
                ro.r.assign("lambda_value", self.lambda_value)

                # Make predictions using R code
                ro.r(
                    """
                predictions <<- as.numeric(predict(r_model, newx = X_new, s = lambda_value, type = "response"))
                """
                )

                # Get predictions from R
                predictions = np.array(ro.r["predictions"])

                return predictions

    return GlmnetRidgeWrapper()
