from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize


class RobynAllocator:
    def __init__(self, mmmdata_collection, model_output, select_model):
        self.mmmdata_collection = mmmdata_collection
        self.model_output = model_output
        self.select_model = select_model

        self.paid_media_spends = mmmdata_collection.mmmdata.mmmdata_spec.paid_media_vars
        self.dep_var_type = mmmdata_collection.mmmdata.mmmdata_spec.dep_var_type
        self.media_data = None
        self.eval_list = {}

    def allocate(
        self,
        scenario: str = "max_response",
        total_budget: float = None,
        target_value: float = None,
        date_range: str = "all",
        channel_constr_low: float = None,
        channel_constr_up: float = None,
        channel_constr_multiplier: float = 3,
        optim_algo: str = "SLSQP",
        maxeval: int = 100000,
        constr_mode: str = "eq",
        plots: bool = True,
        export: bool = True,
        quiet: bool = False,
    ) -> Dict:
        self.prepare_media_data(date_range)
        if self.media_data is None:
            raise ValueError("Media data is not prepared properly.")
        objective_function = self.create_objective_function()
        constraints = self.create_constraints(total_budget, constr_mode)
        result = self.run_optimization(
            objective_function, constraints, optim_algo, maxeval
        )

        allocation_results = self.process_results(result, total_budget, target_value)

        if plots:
            plot_results = self.generate_plots(allocation_results)
        else:
            plot_results = None

        if export:
            self.export_results(allocation_results)

        return {"allocation_results": allocation_results, "plots": plot_results}

    def prepare_media_data(self, date_range):
        # Example initialization, adjust according to your actual data structure
        if date_range == "all":
            self.media_data = self.mmmdata_collection.intermediate_data.dt_mod.copy()
        else:
            # Filter based on date_range if necessary
            pass
        # Ensuring media_data is not None and has the expected columns
        if self.media_data is not None and "initSpendUnit" in self.media_data.columns:
            # Initialize 'initSpendUnit' if not already initialized
            self.media_data["initSpendUnit"] = np.random.rand(
                len(self.media_data)
            )  # or other logic
            self.media_data["lb"] = (
                self.media_data["initSpendUnit"] * 0.8
            )  # Example lower bound
            self.media_data["ub"] = (
                self.media_data["initSpendUnit"] * 1.2
            )  # Example upper bound
        else:
            print("media_data is not properly set or missing 'initSpendUnit'")
            return  # Handle or raise error as appropriate

    def set_constraints(self, scenario, channel_constr_low, channel_constr_up):
        # Implementation of constraint setting
        pass

    def create_objective_function(self):
        def objective(x):
            return -np.sum(self.fx_objective(x, self.eval_list))

        return objective

    def create_constraints(self, total_budget, constr_mode):
        if constr_mode == "eq":
            return [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget}]
        else:
            return [{"type": "ineq", "fun": lambda x: total_budget - np.sum(x)}]

    def run_optimization(self, objective_function, constraints, optim_algo, maxeval):
        return minimize(
            objective_function,
            x0=self.media_data["initSpendUnit"],
            method=optim_algo,
            bounds=list(zip(self.media_data["lb"], self.media_data["ub"])),
            constraints=constraints,
            options={"maxiter": maxeval},
        )

    def process_results(self, result, total_budget, target_value):
        # Implementation of result processing
        pass

    def generate_plots(self, allocation_results):
        # Implementation of plot generation
        pass

    def export_results(self, allocation_results):
        # Implementation of result export
        pass

    @staticmethod
    def fx_objective(x, eval_list):
        coefs = eval_list["coefs_eval"]
        alphas = eval_list["alphas_eval"]
        inflexions = eval_list["inflexions_eval"]
        x_hist_carryover = eval_list["hist_carryover_eval"]

        x_adstocked = x + np.mean(x_hist_carryover)
        return coefs * ((1 + inflexions**alphas / x_adstocked**alphas) ** -1)

    @staticmethod
    def fx_gradient(x, eval_list):
        coefs = eval_list["coefs_eval"]
        alphas = eval_list["alphas_eval"]
        inflexions = eval_list["inflexions_eval"]
        x_hist_carryover = eval_list["hist_carryover_eval"]

        x_adstocked = x + np.mean(x_hist_carryover)
        return (
            -coefs
            * alphas
            * (inflexions**alphas)
            * (x_adstocked ** (alphas - 1))
            / (x_adstocked**alphas + inflexions**alphas) ** 2
        )
