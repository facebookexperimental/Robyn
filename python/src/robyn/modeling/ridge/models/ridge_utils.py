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
            self.logger = logging.getLogger(__name__)

            # Cache for performance
            self._X_matrix_cache = {}
            self._prediction_cache = {}

        def fit(self, X, y):
            # Ensure numpy arrays
            X = np.asarray(X)
            y = np.asarray(y)

            self.logger.debug("\n=== Model Fitting Debug ===")
            self.logger.debug(f"Input shapes - X: {X.shape}, y: {y.shape}")
            self.logger.debug(
                f"X stats - min: {X.min():.6f}, max: {X.max():.6f}, mean: {X.mean():.6f}"
            )
            self.logger.debug(
                f"y stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
            )
            self.logger.debug(f"lambda_value: {self.lambda_value}")

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

                # After getting coefficients
                self.logger.debug("\n=== Coefficient Debug ===")
                self.logger.debug(f"Raw coefficients shape: {coef_array.shape}")
                self.logger.debug(
                    f"Raw coefficients range: [{coef_array.min():.6f}, {coef_array.max():.6f}]"
                )

                # First coefficient is intercept, rest are feature coefficients
                if self.fit_intercept:
                    self.intercept_ = float(coef_array[0])
                    self.coef_ = coef_array[1:]
                else:
                    self.intercept_ = 0.0
                    self.coef_ = coef_array[1:]

                self.logger.debug(f"Final intercept: {self.intercept_:.6f}")
                self.logger.debug(
                    f"Final coefficients range: [{self.coef_.min():.6f}, {self.coef_.max():.6f}]"
                )

            return self

        def predict(self, X):
            X = np.asarray(X)
            self.logger.debug("\n=== Prediction Input ===")
            self.logger.debug(f"X shape: {X.shape}")
            self.logger.debug(f"X range: [{X.min():.6f}, {X.max():.6f}]")
            self.logger.debug(f"X mean: {X.mean():.6f}")
            self.logger.debug(
                f"X stats - min: {X.min():.6f}, max: {X.max():.6f}, mean: {X.mean():.6f}"
            )

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

    return GlmnetRidgeWrapper()
