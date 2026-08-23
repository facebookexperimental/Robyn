import logging

import numpy as np

from glmnetpp import fit_gaussian_ridge


def create_ridge_model_glmnet(
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
    """Create a Ridge regression model using the native ``glmnetpp`` core.

    This replaces the previous rpy2/R implementation (and the transient
    ``glmnet_python`` wrapper). It calls the same C++ ``glmnetpp`` core that
    R's ``glmnet`` >= 4.0 uses (the header-only reimplementation behind
    ``glmnet``), so the fitted coefficients, intercept and predictions are
    numerically identical to R's ``glmnet(family="gaussian", alpha=0,
    lambda=...)`` — including the ``intercept=FALSE`` refit path, which the
    older Fortran ``GLMnet.f`` did not match.

    The returned object mirrors the interface previously exposed by the rpy2
    wrapper so the rest of Robyn (``RidgeModelEvaluator`` and
    ``RidgeMetricsCalculator``) can use it unchanged:

    - ``fit(X, y)``           -> fits the model (X: 2-D, y: 1-D)
    - ``predict(X)``          -> 1-D predictions on the original scale
    - ``coef_``               -> coefficients on the original scale (without intercept)
    - ``intercept_``          -> intercept (float)
    - ``get_full_coefficients()`` -> ``[intercept, *coef_]`` (R-style ordering)
    - ``df_int``              -> 1 when the intercept is kept, 0 when dropped

    The ``intercept_sign="non_negative"`` behaviour is preserved: the model is
    first fit with an intercept; if that intercept is negative it is refit
    without one and ``df_int`` becomes 0 (matching R's ``intercept=FALSE``
    refit path).

    Args:
        lambda_value: Regularization parameter (single lambda) passed to glmnet.
        n_samples: Number of samples (kept for API consistency, unused).
        fit_intercept: Whether an intercept is intended to be fit.
        standardize: Whether glmnet standardizes the predictors internally
            (R's default and Robyn's behaviour is ``True``).
        lower_limits: per-coefficient lower bounds (R's ``lower.limits``).
        upper_limits: per-coefficient upper bounds (R's ``upper.limits``).
        intercept: kept for API consistency with the previous signature.
        intercept_sign: sign constraint on the intercept
            (``"non_negative"`` drops a negative intercept).
        penalty_factor: per-coefficient penalty multipliers (R's
            ``penalty.factor``).

    Returns:
        A fitted-on-demand wrapper object with the interface described above.
    """
    # Capture the limit / penalty configuration at construction time, exactly
    # as the previous rpy2 wrapper did through its closure.
    _lower_limits = lower_limits
    _upper_limits = upper_limits
    _penalty_factor = penalty_factor

    class GlmnetRidgeWrapper:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.lambda_value = float(lambda_value)
            self.fit_intercept = fit_intercept
            self.standardize = standardize
            self.intercept_sign = intercept_sign
            self.fitted_model = None
            self.coef_ = None
            self.intercept_ = 0.0
            self.full_coef_ = None
            # R-style degrees of freedom for the intercept: 1 when kept, 0 when
            # dropped because of the non-negative intercept constraint.
            self.df_int = 1

        def _fit_one(self, X, y, intr):
            """Run a single glmnet fit; return (intercept, coef, jerr).

            ``coef`` is length-p on the original (unstandardized) scale, which
            is what the C++ core returns directly (it unstandardizes
            internally, even when ``standardize=True``).
            """
            n_vars = np.asarray(X).shape[1]
            if _lower_limits is None:
                lower = np.full(n_vars, -np.inf, dtype=np.float64)
            else:
                lower = np.asarray(_lower_limits, dtype=np.float64).reshape(-1)
            if _upper_limits is None:
                upper = np.full(n_vars, np.inf, dtype=np.float64)
            else:
                upper = np.asarray(_upper_limits, dtype=np.float64).reshape(-1)
            if _penalty_factor is None:
                pf = np.ones(n_vars, dtype=np.float64)
            else:
                pf = np.asarray(_penalty_factor, dtype=np.float64).reshape(-1)

            intercept_, coef, jerr = fit_gaussian_ridge(
                X,
                y,
                self.lambda_value,
                alpha=0.0,
                lower_limits=lower,
                upper_limits=upper,
                penalty_factor=pf,
                standardize=self.standardize,
                intercept=intr,
                thresh=np.float64(1e-7),
            )
            return float(intercept_), np.asarray(coef, dtype=np.float64), jerr

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            # First attempt: fit with an intercept (matches the previous R
            # wrapper's first glmnet call with intercept = TRUE).
            intercept_, coef, jerr = self._fit_one(X, y, intr=True)

            # If the intercept must be non-negative but came out negative, refit
            # without an intercept and drop the intercept degrees of freedom.
            if self.intercept_sign == "non_negative" and intercept_ < 0:
                intercept_, coef, jerr = self._fit_one(X, y, intr=False)
                self.fit_intercept = False
                self.df_int = 0
            else:
                self.df_int = 1

            self.fitted_model = jerr  # status code, retained for diagnostics
            self.intercept_ = intercept_
            self.coef_ = coef
            if self.fit_intercept:
                self.full_coef_ = np.concatenate([[intercept_], coef])
            else:
                self.intercept_ = 0.0
                self.full_coef_ = np.concatenate([[0.0], coef])

            self.logger.debug(
                "Coefficients range: [%.6f, %.6f], intercept: %.6f, df_int: %d, jerr: %d",
                np.min(self.coef_),
                np.max(self.coef_),
                self.intercept_,
                self.df_int,
                jerr,
            )
            return self

        def predict(self, X):
            if self.coef_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X, dtype=np.float64)
            # For family="gaussian" the response is the linear predictor on the
            # original scale, i.e. X @ beta + a0, which is exactly what R's
            # predict(..., type="response") computes.
            return X @ self.coef_ + self.intercept_

        def get_full_coefficients(self):
            """Get full coefficient array including intercept (R-style)."""
            return self.full_coef_

    return GlmnetRidgeWrapper()