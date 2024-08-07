# pyre-strict

from typing import TypedDict, Any
import matplotlib.pyplot as plt

class ConvergenceResult(TypedDict):
    """
    ConvergenceResult is a typed dictionary that stores the results of the convergence analysis.

    Attributes:
        moo_distrb_plot (plt.Figure): The distribution plot for the multi-objective optimization.
        moo_cloud_plot (plt.Figure): The cloud plot for the multi-objective optimization.
        errors (Any): Errors encountered during the convergence analysis.
        conv_msg (str): Convergence message.
        sd_qtref (float): Standard deviation quantile reference.
        med_lowb (float): Median lower bound.
    """
    moo_distrb_plot: plt.Figure
    moo_cloud_plot: plt.Figure
    errors: Any
    conv_msg: str
    sd_qtref: float
    med_lowb: float

# Example usage:
if __name__ == "__main__":
    # Example matplotlib figures
    fig1 = plt.figure()
    fig2 = plt.figure()

    # Example ConvergenceResult
    convergence_result: ConvergenceResult = {
        'moo_distrb_plot': fig1,
        'moo_cloud_plot': fig2,
        'errors': None,
        'conv_msg': "Convergence successful.",
        'sd_qtref': 3.0,
        'med_lowb': 2.0
    }

    print(convergence_result)
