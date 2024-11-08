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
            "SLSQP_AUGLAG": "SLSQP", 
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
        
        self.logger.info(f"Starting optimization with method: {method}")
        self.logger.debug(f"Optimization parameters: maxeval={maxeval}, constr_mode={constr_mode}")
        self.logger.debug(f"Initial guess shape: {initial_guess.shape}")
        self.logger.debug(f"Number of bounds: {len(bounds)}")
        self.logger.debug(f"Number of constraints: {len(constraints)}")

        # Validate method
        if method not in self.supported_methods:
            err_msg = f"Unsupported optimization method: {method}. Supported methods: {list(self.supported_methods.keys())}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        scipy_method = self.supported_methods[method]
        self.logger.debug(f"Using scipy solver: {scipy_method}")

        # Setup optimization options
        options = {"maxiter": maxeval, "ftol": 1e-10, "disp": False}
        self.logger.debug(f"Optimization options: {options}")

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
                self.logger.debug(f"Processed constraint {i+1}: type={constraint_type}")
            except KeyError as e:
                self.logger.error(f"Error processing constraint {i+1}: {str(e)}")
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
                self.logger.warning(f"Optimization may not have converged. Message: {result.message}")
            else:
                self.logger.info("Optimization completed successfully")
                self.logger.debug(f"Final objective value: {result.fun}")
                self.logger.debug(f"Number of iterations: {result.nit}")
                self.logger.debug(f"Number of function evaluations: {result.nfev}")

            optimization_result = {
                "x": result.x,
                "fun": result.fun,
                "success": result.success,
                "message": result.message,
                "nfev": result.nfev,
                "nit": result.nit if hasattr(result, "nit") else None,
            }

            self.logger.debug(f"Optimization result shape: {result.x.shape}")
            return optimization_result

        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            raise ValueError(f"Optimization failed: {str(e)}")