from typing import Callable, List, Tuple, Dict
import numpy as np
from scipy.optimize import minimize
from robyn.allocator.entities.enums import ConstrMode


class AllocationOptimizer:
    """Handles numerical optimization for budget allocation."""

    def __init__(self):
        """Initialize optimizer."""
        self.supported_methods = {
            "SLSQP_AUGLAG": "SLSQP",  # Map Robyn method names to scipy method names
            "MMA_AUGLAG": "SLSQP",  # For now, map both to SLSQP as it's most similar
        }

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
        """
        Run optimization with given objective and constraints.

        Args:
            objective_func: Function to minimize
            bounds: List of (lower, upper) bounds for each variable
            constraints: List of constraint dictionaries
            initial_guess: Initial values for optimization
            method: Optimization method (SLSQP_AUGLAG or MMA_AUGLAG)
            maxeval: Maximum number of function evaluations
            constr_mode: Constraint mode (equality or inequality)

        Returns:
            Dictionary containing optimization results
        """
        # Convert method name to scipy solver name
        if method not in self.supported_methods:
            raise ValueError(
                f"Unsupported optimization method: {method}. "
                f"Supported methods are: {list(self.supported_methods.keys())}"
            )

        scipy_method = self.supported_methods[method]

        # Setup optimization options
        options = {"maxiter": maxeval, "ftol": 1e-10, "disp": False}

        # Process constraints
        processed_constraints = []
        for c in constraints:
            constraint_type = c.get("type", "eq")
            if constraint_type not in ["eq", "ineq"]:
                raise ValueError(f"Invalid constraint type: {constraint_type}")

            processed_constraints.append({"type": constraint_type, "fun": c["fun"], "jac": c.get("jac")})

        try:
            # Run optimization
            result = minimize(
                fun=objective_func,
                x0=initial_guess,
                method=scipy_method,
                bounds=bounds,
                constraints=processed_constraints,
                options=options,
            )

            if not result.success:
                print(f"Warning: Optimization may not have converged. Message: {result.message}")

            # Return results in a consistent format
            return {
                "x": result.x,
                "fun": result.fun,
                "success": result.success,
                "message": result.message,
                "nfev": result.nfev,
                "nit": result.nit if hasattr(result, "nit") else None,
            }

        except Exception as e:
            raise ValueError(f"Optimization failed: {str(e)}")
