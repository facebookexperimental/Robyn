from typing import Callable, List, Tuple, Dict
import numpy as np
from scipy.optimize import minimize
from ..entities.enums import ConstrMode


class AllocationOptimizer:
    """Handles numerical optimization for budget allocation."""

    def optimize(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        constraints: List[Dict],
        initial_guess: np.ndarray,
        method: str = "SLSQP",
        maxeval: int = 100000,
        constr_mode: ConstrMode = ConstrMode.EQUALITY,
    ) -> Dict:
        """
        Run optimization with given objective and constraints.

        Implementation follows optimization logic from allocator.R
        """
        # Setup optimization options
        options = {"maxiter": maxeval, "ftol": 1e-10, "disp": False}

        # Setup constraint dictionaries based on mode
        if constr_mode == ConstrMode.EQUALITY:
            constraints = [{"type": "eq", "fun": c["fun"], "jac": c.get("jac")} for c in constraints]
        else:
            constraints = [{"type": "ineq", "fun": c["fun"], "jac": c.get("jac")} for c in constraints]

        # Run optimization
        result = minimize(
            objective_func, initial_guess, method=method, bounds=bounds, constraints=constraints, options=options
        )

        return {
            "x": result.x,
            "fun": result.fun,
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "nit": result.nit,
        }
