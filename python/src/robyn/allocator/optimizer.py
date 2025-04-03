import nlopt
import numpy as np
from typing import Optional, Dict, Any, List, Union


def run_optimization(
    x0: np.ndarray,
    eval_f: callable,
    eval_g_eq: Optional[callable],
    eval_g_ineq: Optional[callable],
    lb: np.ndarray,
    ub: np.ndarray,
    maxeval: int,
    local_opts: Dict[str, Any],
    target_value: Optional[float] = None,
    eval_dict: Optional[Dict] = None,
) -> Dict:
    """Run nlopt optimization, matching R's nloptr interface"""

    # Create optimizer with AUGLAG algorithm
    opt = nlopt.opt(nlopt.LD_AUGLAG, len(x0))

    # Set local optimizer based on local_opts
    if local_opts["algorithm"] == "NLOPT_LD_SLSQP":
        local_optimizer = nlopt.opt(nlopt.LD_SLSQP, len(x0))
    elif local_opts["algorithm"] == "NLOPT_LD_MMA":
        local_optimizer = nlopt.opt(nlopt.LD_MMA, len(x0))
    else:
        raise ValueError(f"Unsupported local optimizer: {local_opts['algorithm']}")

    local_optimizer.set_xtol_rel(local_opts["xtol_rel"])
    opt.set_local_optimizer(local_optimizer)

    # Wrap evaluation functions to handle dictionary returns
    def objective_wrapper(x, grad):
        result = eval_f(x, grad)  # Match lambda signature
        if grad.size > 0:
            grad[:] = result["gradient"]
        return result["objective"]

    def eq_constraint_wrapper(x, grad):
        result = eval_g_eq(x, grad)  # Match lambda signature
        if grad.size > 0:
            grad[:] = result["jacobian"]
        return result["constraints"]

    def ineq_constraint_wrapper(x, grad):
        result = eval_g_ineq(x, grad)  # Match lambda signature
        if grad.size > 0:
            grad[:] = result["jacobian"]
        return result["constraints"]

    # Set objective and constraints with wrappers
    opt.set_min_objective(objective_wrapper)

    # Set bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Set constraints using wrappers
    if eval_g_eq is not None:
        opt.add_equality_constraint(eq_constraint_wrapper)
    if eval_g_ineq is not None:
        opt.add_inequality_constraint(ineq_constraint_wrapper)

    # Set options
    opt.set_maxeval(maxeval)
    opt.set_xtol_rel(1.0e-10)

    # Run optimization
    try:
        x = opt.optimize(x0)
        result = {
            "solution": x,
            "objective": opt.last_optimum_value(),
            "status": opt.last_optimize_result(),
        }
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        result = {"solution": x0, "objective": float("inf"), "status": -1}

    return result


def eval_f(
    X: np.ndarray,
    grad_or_eval_dict: Union[np.ndarray, Dict[str, Any]],
    target_value: Optional[float] = None,
) -> Union[Dict, float]:
    """
    Evaluation function for optimization. Can be called in two ways:
    1. eval_f(X, eval_dict, target_value) - direct call
    2. eval_f(x, grad) - called through lambda in nlopt

    Args:
        X: Input array of spend values
        grad_or_eval_dict: Either gradient array (nlopt) or eval_dict (direct)
        target_value: Optional target value

    Returns:
        Dictionary containing objective, gradient, and per-channel objectives
    """
    # Determine if this is a direct call or through nlopt
    if isinstance(grad_or_eval_dict, dict):
        # Direct call with eval_dict
        eval_dict = grad_or_eval_dict
        grad = None
    else:
        # Called through nlopt with grad
        eval_dict = getattr(eval_f, "eval_dict", None)
        grad = grad_or_eval_dict

    # Unpack evaluation dictionary
    coefs_eval = eval_dict["coefs_eval"]
    alphas_eval = eval_dict["alphas_eval"]
    inflexions_eval = eval_dict["inflexions_eval"]
    hist_carryover_eval = eval_dict["hist_carryover_eval"]

    # Calculate mean carryover
    x_hist_carryover = {k: np.mean(v) for k, v in hist_carryover_eval.items()}

    # Calculate objective for each channel and convert to numpy array
    objective_channel = np.array(
        [
            fx_objective(
                x=x,
                coeff=coefs_eval[channel],
                alpha=alphas_eval[f"{channel}_alphas"],
                inflexion=inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=x_hist_carryover[channel],
            )
            for channel, x in zip(coefs_eval.keys(), X)
        ]
    )

    # Calculate total objective
    objective = -np.sum(objective_channel)

    # Calculate gradient
    gradient = np.array(
        [
            fx_gradient(
                x=x,
                coeff=coefs_eval[channel],
                alpha=alphas_eval[f"{channel}_alphas"],
                inflexion=inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=x_hist_carryover[channel],
            )
            for channel, x in zip(coefs_eval.keys(), X)
        ]
    )

    # Update gradient if provided
    if grad is not None and grad.size > 0:
        grad[:] = gradient

    return {
        "objective": objective,
        "gradient": gradient,
        "objective_channel": objective_channel,
    }


def fx_objective(
    x: float,
    coeff: float,
    alpha: float,
    inflexion: float,
    x_hist_carryover: float,
    get_sum: bool = True,
) -> float:
    """
    Calculate objective function using Hill transformation

    Args:
        x: Input value
        coeff: Coefficient
        alpha: Alpha parameter (with _alphas suffix in name)
        inflexion: Inflexion parameter (with _gammas suffix in name)
        x_hist_carryover: Historical carryover value
        get_sum: Whether to sum the result

    Returns:
        float: Transformed value
    """
    # Adstock scales
    x_adstocked = x + x_hist_carryover

    # Hill transformation
    if get_sum:
        x_out = coeff * np.sum((1 + inflexion**alpha / x_adstocked**alpha) ** -1)
    else:
        x_out = coeff * ((1 + inflexion**alpha / x_adstocked**alpha) ** -1)

    return x_out


def fx_gradient(
    x: float, coeff: float, alpha: float, inflexion: float, x_hist_carryover: float
) -> float:
    """
    Calculate gradient of the objective function
    Source: https://www.derivative-calculator.net/ on the objective function 1/(1+gamma^alpha / x^alpha)

    Args:
        x: Input value
        coeff: Coefficient
        alpha: Alpha parameter
        inflexion: Inflexion parameter (gamma)
        x_hist_carryover: Historical carryover value

    Returns:
        float: Gradient value
    """
    # Adstock scales
    x_adstocked = x + x_hist_carryover

    # Calculate gradient
    x_out = -coeff * np.sum(
        (alpha * (inflexion**alpha) * (x_adstocked ** (alpha - 1)))
        / (x_adstocked**alpha + inflexion**alpha) ** 2
    )

    return x_out


def fx_objective_channel(
    x: float, coeff: float, alpha: float, inflexion: float, x_hist_carryover: float
) -> float:
    """
    Calculate per-channel objective

    Args:
        x: Input value
        coeff: Coefficient
        alpha: Alpha parameter
        inflexion: Inflexion parameter
        x_hist_carryover: Historical carryover value

    Returns:
        float: Channel objective value
    """
    # Adstock scales
    x_adstocked = x + x_hist_carryover

    # Calculate channel objective
    x_out = -coeff * np.sum((1 + inflexion**alpha / x_adstocked**alpha) ** -1)

    return x_out


def eval_g_eq(
    X: np.ndarray,
    grad_or_eval_dict: Union[np.ndarray, Dict[str, Any]],
    target_value: Optional[float] = None,
) -> Union[Dict, float]:
    """
    Equality constraint evaluation. Can be called in two ways:
    1. eval_g_eq(X, eval_dict, target_value) - direct call
    2. eval_g_eq(x, grad) - called through lambda in nlopt
    """
    # Determine if this is a direct call or through nlopt
    if isinstance(grad_or_eval_dict, dict):
        eval_dict = grad_or_eval_dict
        grad = None
    else:
        eval_dict = getattr(eval_g_eq, "eval_dict", None)
        grad = grad_or_eval_dict

    constr = np.sum(X) - eval_dict["total_budget_unit"]
    gradient = np.ones(len(X))

    # Update gradient if provided
    if grad is not None and grad.size > 0:
        grad[:] = gradient

    return {"constraints": constr, "jacobian": gradient}


def eval_g_ineq(
    X: np.ndarray, eval_dict: Dict[str, Any], target_value: float = None
) -> Dict:
    """
    Inequality constraint evaluation

    Args:
        X: Input array
        eval_dict: Dictionary containing evaluation parameters
        target_value: Optional target value

    Returns:
        Dict with constraints and jacobian
    """
    constr = np.sum(X) - eval_dict["total_budget_unit"]
    grad = np.ones(len(X))

    return {"constraints": constr, "jacobian": grad}


def eval_g_eq_effi(
    X: np.ndarray, eval_dict: Dict[str, Any], target_value: float = None
) -> Dict:
    """
    Efficiency constraint evaluation

    Args:
        X: Input array
        eval_dict: Dictionary containing evaluation parameters
        target_value: Optional target value

    Returns:
        Dict with constraints and jacobian
    """
    # Calculate sum response
    channels = list(eval_dict["coefs_eval"].keys())
    sum_response = np.sum(
        fx_objective(
            x=x,
            coeff=eval_dict["coefs_eval"][channel],
            alpha=eval_dict["alphas_eval"][f"{channel}_alphas"],
            inflexion=eval_dict["inflexions_eval"][f"{channel}_gammas"],
            x_hist_carryover=eval_dict["hist_carryover_eval"][channel],
        )
        for channel, x in zip(channels, X)
    )

    # Calculate constraint based on target value and dep_var_type
    target = target_value if target_value is not None else eval_dict["target_value"]
    if eval_dict["dep_var_type"] == "conversion":
        constr = np.sum(X) - sum_response * target
    else:
        constr = np.sum(X) - sum_response / target

    # Calculate gradient
    grad = np.ones(len(X)) - np.array(
        [
            fx_gradient(
                x=x,
                coeff=eval_dict["coefs_eval"][channel],
                alpha=eval_dict["alphas_eval"][f"{channel}_alphas"],
                inflexion=eval_dict["inflexions_eval"][f"{channel}_gammas"],
                x_hist_carryover=eval_dict["hist_carryover_eval"][channel],
            )
            for channel, x in zip(channels, X)
        ]
    )

    return {"constraints": constr, "jacobian": grad}
