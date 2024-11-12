#pyre-strict
"""Optimization module for budget allocation."""

from typing import Callable, List, Tuple, Dict
import logging
import numpy as np
from scipy.optimize import minimize
from robyn.allocator.entities.enums import ConstrMode

class AllocationOptimizer:
    """Handles numerical optimization for budget allocation."""

    def __init__(self):
        """Initialize optimizer."""
        self.supported_methods = {
            "SLSQP_AUGLAG": "SLSQP", #TODO: both constant values are same. is this correct?
            "MMA_AUGLAG": "SLSQP",  
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AllocationOptimizer")

    def optimize(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        constraints: List[Dict],
        initial_guess: np.ndarray,
        method: str = "SLSQP_AUGLAG",
        maxeval: int = 100000,
        constr_mode: ConstrMode = ConstrMode.EQUALITY,
    ) -> Dict:
        """Run optimization with given objective and constraints."""
        
        self.logger.info("Starting optimization with method: %s", method)
        self.logger.debug("Optimization parameters: maxeval=%d, constr_mode=%s", maxeval, constr_mode)
        self.logger.debug("Initial guess shape: %s", initial_guess.shape)
        self.logger.debug("Number of bounds: %d", len(bounds))
        self.logger.debug("Number of constraints: %d", len(constraints))

        # Validate method
        if method not in self.supported_methods:
            err_msg = f"Unsupported optimization method: {method}. Supported methods: {list(self.supported_methods.keys())}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        scipy_method = self.supported_methods[method]
        self.logger.debug("Using scipy solver: %s", scipy_method)

        # Setup optimization options
        options = {"maxiter": maxeval, "ftol": 1e-10, "disp": False}
        self.logger.debug("Optimization options: %s", options)

        # Process constraints
        processed_constraints = []
        for i, c in enumerate(constraints):
            try:
                constraint_type = c.get("type", "eq")
                if constraint_type not in ["eq", "ineq"]:
                    err_msg = f"Invalid constraint type: {constraint_type}"
                    self.logger.error(err_msg)
                    raise ValueError(err_msg)

                processed_constraints.append({
                    "type": constraint_type,
                    "fun": c["fun"],
                    "jac": c.get("jac")
                })
                self.logger.debug("Processed constraint %d: type=%s", i+1, constraint_type)
            except KeyError as e:
                self.logger.error("Error processing constraint %d: %s", i+1, str(e))
                raise

        try:
            self.logger.info("Running optimization")
            result = minimize(
                fun=objective_func,
                x0=initial_guess,
                method=scipy_method,
                bounds=bounds,
                constraints=processed_constraints,
                options=options,
            )

            if not result.success:
                self.logger.warning("Optimization may not have converged. Message: %s", result.message)
            else:
                self.logger.info("Optimization completed successfully")
                self.logger.debug("Final objective value: %s", result.fun)
                self.logger.debug("Number of iterations: %d", result.nit)
                self.logger.debug("Number of function evaluations: %d", result.nfev)

            optimization_result = {
                "x": result.x,
                "fun": result.fun,
                "success": result.success,
                "message": result.message,
                "nfev": result.nfev,
                "nit": result.nit if hasattr(result, "nit") else None,
            }

            self.logger.debug("Optimization result shape: %s", result.x.shape)
            return optimization_result

        except Exception as e:
            self.logger.error("Optimization failed: %s", str(e), exc_info=True)
            raise ValueError(f"Optimization failed: {str(e)}") from e