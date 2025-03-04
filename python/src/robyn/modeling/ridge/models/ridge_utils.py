import logging
import numpy as np
from sklearn.linear_model import Ridge
import time


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
    lambda_value, n_samples, fit_intercept=True, standardize=True, **kwargs
):
    """Create a Ridge regression model using rpy2 to access glmnet.

    Args:
        lambda_value: Regularization parameter
        n_samples: Number of samples (not directly used, but kept for API consistency)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features
        **kwargs: Additional arguments to pass to glmnet

    Returns:
        A Ridge regression model using rpy2 to access glmnet.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        # Import glmnet only once per Python session
        global glmnet_imported
        if "glmnet_imported" not in globals():
            try:
                importr("glmnet")
                glmnet_imported = True
            except Exception as e:
                logging.warning(f"Failed to import glmnet: {e}")
                logging.warning("Falling back to sklearn implementation")
                return create_ridge_model_sklearn(
                    lambda_value, n_samples, fit_intercept, standardize
                )
    except ImportError:
        logging.warning("rpy2 not available, using sklearn implementation")
        return create_ridge_model_sklearn(
            lambda_value, n_samples, fit_intercept, standardize
        )

    class GlmnetRidgeWrapper:
        def __init__(self):
            self.lambda_value = lambda_value
            self.fit_intercept = fit_intercept
            self.standardize = standardize
            self.kwargs = kwargs
            self.fitted_model = None
            self.coef_ = None
            self.intercept_ = 0.0

            # Cache for performance
            self._X_matrix_cache = {}
            self._prediction_cache = {}

        def fit(self, X, y):
            # Ensure numpy arrays
            X = np.asarray(X)
            y = np.asarray(y)

            fit_intercept_r = "TRUE" if self.fit_intercept else "FALSE"
            standardize_r = "TRUE" if self.standardize else "FALSE"

            # Collect optional parameters
            optional_params = []

            # Generate key for caching
            cache_key = (
                hash(X.tobytes()),
                hash(y.tobytes()),
                self.lambda_value,
                self.fit_intercept,
                self.standardize,
            )

            # Convert Python objects to R
            with localconverter(ro.default_converter + numpy2ri.converter):
                # Pass the data to R environment
                ro.r.assign("X_r", X)
                ro.r.assign("y_r", y)
                ro.r.assign("lambda_value", self.lambda_value)

                # Extract optional parameters
                lower_limits = kwargs.get("lower_limits", None)
                upper_limits = kwargs.get("upper_limits", None)

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

                # Execute R code for model fitting
                ro.r(r_code)

                # Get the model and coefficients from R
                self.fitted_model = ro.r["r_model"]
                coef_array = np.array(ro.r["coef_values"])

                # Store X matrix for future predictions
                self._X_matrix_cache[cache_key] = X

                # First coefficient is intercept, rest are feature coefficients
                if self.fit_intercept:
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

            # For small matrices (fewer than 1000 rows), it's faster to just
            # compute the prediction in Python directly
            if X.shape[0] < 1000:
                return np.dot(X, self.coef_) + self.intercept_

            # For larger matrices, use R but check cache first
            X_hash = hash(X.tobytes())
            if X_hash in self._prediction_cache:
                return self._prediction_cache[X_hash]

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

                # Cache the predictions
                self._prediction_cache[X_hash] = predictions

                return predictions

    return GlmnetRidgeWrapper()
