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
            self.logger = logging.getLogger(__name__)
            self.lambda_value = lambda_value  # Use raw lambda value
            self.fit_intercept = fit_intercept
            self.standardize = standardize
            self.feature_means = None
            self.feature_stds = None
            self.y_mean = None
            self.coef_ = None
            self.intercept_ = 0.0

        def mysd(self, y):
            """R-like standard deviation"""
            return np.sqrt(np.sum((y - np.mean(y)) ** 2) / len(y))

        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y)

            # Debug prints matching R
            self.logger.debug("Lambda calculation debug:")
            self.logger.debug(f"x_means: {np.mean(np.abs(X))}")
            x_sds = np.apply_along_axis(self.mysd, 0, X)
            self.logger.debug(f"x_sds mean: {np.mean(x_sds)}")

            # Center and scale like R's glmnet
            if self.standardize:
                self.feature_means = np.mean(X, axis=0)
                self.feature_stds = np.apply_along_axis(self.mysd, 0, X)
                self.feature_stds[self.feature_stds == 0] = 1.0
                X_scaled = (X - self.feature_means) / self.feature_stds
            else:
                X_scaled = X
                self.feature_means = np.zeros(X.shape[1])
                self.feature_stds = np.ones(X.shape[1])

            if self.fit_intercept:
                self.y_mean = np.mean(y)
                y_centered = y - self.y_mean
            else:
                y_centered = y
                self.y_mean = 0.0

            self.logger.debug(f"sx mean: {np.mean(np.abs(X_scaled))}")
            self.logger.debug(f"sy mean: {np.mean(np.abs(y_centered))}")
            self.logger.debug(f"lambda: {self.lambda_value}")

            # Fit model using raw lambda (not scaled)
            model = Ridge(
                alpha=self.lambda_value,
                fit_intercept=False,  # We handle centering manually
                solver="cholesky",
            )

            model.fit(X_scaled, y_centered)

            # Transform coefficients back to original scale
            if self.standardize:
                self.coef_ = model.coef_ / self.feature_stds
            else:
                self.coef_ = model.coef_

            if self.fit_intercept:
                self.intercept_ = self.y_mean - np.dot(self.feature_means, self.coef_)

            self.logger.debug(
                f"Coefficients range: [{np.min(self.coef_):.6f}, {np.max(self.coef_):.6f}]"
            )
            self.logger.debug(f"Intercept: {self.intercept_:.6f}")

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
    intercept=True,
    intercept_sign="non_negative",
    penalty_factor=None,
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

    Raises:
        ImportError: If rpy2 is not available
        RuntimeError: If glmnet R package cannot be imported
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
    except ImportError:
        raise ImportError(
            "rpy2 is required for using the R implementation. Please install rpy2."
        )

    # Import glmnet only once per Python session
    global glmnet_imported
    if "glmnet_imported" not in globals():
        try:
            importr("glmnet")
            glmnet_imported = True
        except Exception as e:
            raise RuntimeError(f"Failed to import glmnet R package: {e}")

    class GlmnetRidgeWrapper:
        def __init__(self):
            self.lambda_value = lambda_value
            self.fit_intercept = fit_intercept
            self.standardize = standardize
            self.intercept_sign = intercept_sign
            self.fitted_model = None
            self.coef_ = None
            self.intercept_ = 0.0
            self.logger = logging.getLogger(__name__)
            self._prediction_cache = {}
            # Cache for performance
            self._X_matrix_cache = {}
            self.full_coef_ = None  # Add this to store full coefficient array
            self.df_int = 1  # Initialize to 1

        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y)

            # Convert Python objects to R
            with localconverter(ro.default_converter + numpy2ri.converter):
                ro.r.assign("X_r", X)
                ro.r.assign("y_r", y)
                ro.r.assign("lambda_value", self.lambda_value)
                ro.r.assign(
                    "lower_limits_r",
                    lower_limits if lower_limits is not None else ro.r("NULL"),
                )
                ro.r.assign(
                    "upper_limits_r",
                    upper_limits if upper_limits is not None else ro.r("NULL"),
                )
                ro.r.assign(
                    "penalty_factor_r",
                    penalty_factor if penalty_factor is not None else ro.r("NULL"),
                )

                # First attempt: Fit with intercept
                r_code = """
                # First fit with intercept
                r_model <<- glmnet(
                    x = X_r,
                    y = y_r,
                    family = "gaussian",
                    alpha = 0,
                    lambda = lambda_value,
                    lower.limits = lower_limits_r,
                    upper.limits = upper_limits_r,
                    type.measure = "mse",
                    penalty.factor = penalty_factor_r,
                    intercept = TRUE
                )
                coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                """
                ro.r(r_code)

                # Check intercept sign constraint
                coef_array = np.array(ro.r["coef_values"])
                if self.intercept_sign == "non_negative" and coef_array[0] < 0:
                    # Second attempt: Refit without intercept
                    r_code = """
                    # Refit without intercept
                    r_model <<- glmnet(
                        x = X_r,
                        y = y_r,
                        family = "gaussian",
                        alpha = 0,
                        lambda = lambda_value,
                        lower.limits = lower_limits_r,
                        upper.limits = upper_limits_r,
                        type.measure = "mse",
                        penalty.factor = penalty_factor_r,
                        intercept = FALSE
                    )
                    coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                    """
                    ro.r(r_code)
                    coef_array = np.array(ro.r["coef_values"])
                    self.fit_intercept = False
                    self.df_int = 0  # Set df_int to 0 when intercept is dropped
                else:
                    self.df_int = 1  # Keep df_int as 1 when intercept is kept

                # Store model and coefficients
                self.fitted_model = ro.r["r_model"]
                if self.fit_intercept:
                    self.intercept_ = float(coef_array[0])
                    self.coef_ = coef_array[1:]
                    self.full_coef_ = coef_array  # Store full array including intercept
                else:
                    self.intercept_ = 0.0
                    self.coef_ = coef_array[1:]
                    # Create full coefficient array with 0 intercept
                    self.full_coef_ = np.concatenate([[0.0], self.coef_])

            return self

        def predict(self, X):
            X = np.asarray(X)

            if X.shape[0] < 1000:
                predictions = np.dot(X, self.coef_) + self.intercept_
                self.logger.debug(f"Using direct computation")
            else:
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
                    self.logger.debug("\n=== Prediction Output ===")
                    self.logger.debug(
                        f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]"
                    )
                    self.logger.debug(f"Predictions mean: {predictions.mean():.6f}")
                    # Cache the predictions
                    self._prediction_cache[X_hash] = predictions

                    self.logger.debug(f"Using R computation")

            self.logger.debug(
                f"Predictions stats - min: {predictions.min():.6f}, max: {predictions.max():.6f}, mean: {predictions.mean():.6f}"
            )
            return predictions

        def get_full_coefficients(self):
            """Get full coefficient array including intercept (R-style)"""
            return self.full_coef_

    return GlmnetRidgeWrapper()
