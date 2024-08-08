# pyre-strict

from typing import Optional


class BudgetAllocatorConfig:
    """
    Configuration for budget allocation.

    Attributes:
        scenario (str): The scenario for the budget allocation.
        total_budget (float): The total budget for the allocation.
        target_value (float): The target value for the allocation.
        date_range (str): The date range for the allocation.
        channel_constr_low (float): The lower constraint for the channels.
        channel_constr_up (float): The upper constraint for the channels.
        channel_constr_multiplier (int): The multiplier for the channel constraints.
        optim_algo (str): The optimization algorithm to use.
        maxeval (int): The maximum number of evaluations for the optimization.
        constr_mode (str): The constraint mode for the optimization.
    """

    def __init__(
        self,
        scenario: str,
        total_budget: float,
        target_value: float,
        date_range: str,
        channel_constr_low: float,
        channel_constr_up: float,
        channel_constr_multiplier: int,
        optim_algo: str,
        maxeval: int,
        constr_mode: str
    ) -> None:
        self.scenario: str = scenario
        self.total_budget: float = total_budget
        self.target_value: float = target_value
        self.date_range: str = date_range
        self.channel_constr_low: float = channel_constr_low
        self.channel_constr_up: float = channel_constr_up
        self.channel_constr_multiplier: int = channel_constr_multiplier
        self.optim_algo: str = optim_algo
        self.maxeval: int = maxeval
        self.constr_mode: str = constr_mode

    def __str__(self) -> str:
        return (
            f"BudgetAllocatorConfig("
            f"scenario={self.scenario}, "
            f"total_budget={self.total_budget}, "
            f"target_value={self.target_value}, "
            f"date_range={self.date_range}, "
            f"channel_constr_low={self.channel_constr_low}, "
            f"channel_constr_up={self.channel_constr_up}, "
            f"channel_constr_multiplier={self.channel_constr_multiplier}, "
            f"optim_algo={self.optim_algo}, "
            f"maxeval={self.maxeval}, "
            f"constr_mode={self.constr_mode}"
            f")"
        )

    def update(
        self,
        scenario: Optional[str] = None,
        total_budget: Optional[float] = None,
        target_value: Optional[float] = None,
        date_range: Optional[str] = None,
        channel_constr_low: Optional[float] = None,
        channel_constr_up: Optional[float] = None,
        channel_constr_multiplier: Optional[int] = None,
        optim_algo: Optional[str] = None,
        maxeval: Optional[int] = None,
        constr_mode: Optional[str] = None
    ) -> None:
        """
        Update the BudgetAllocatorConfig parameters.

        :param scenario: The new scenario for the budget allocation.
        :param total_budget: The new total budget for the allocation.
        :param target_value: The new target value for the allocation.
        :param date_range: The new date range for the allocation.
        :param channel_constr_low: The new lower constraint for the channels.
        :param channel_constr_up: The new upper constraint for the channels.
        :param channel_constr_multiplier: The new multiplier for the channel constraints.
        :param optim_algo: The new optimization algorithm to use.
        :param maxeval: The new maximum number of evaluations for the optimization.
        :param constr_mode: The new constraint mode for the optimization.
        """
        if scenario is not None:
            self.scenario = scenario
        if total_budget is not None:
            self.total_budget = total_budget
        if target_value is not None:
            self.target_value = target_value
        if date_range is not None:
            self.date_range = date_range
        if channel_constr_low is not None:
            self.channel_constr_low = channel_constr_low
        if channel_constr_up is not None:
            self.channel_constr_up = channel_constr_up
        if channel_constr_multiplier is not None:
            self.channel_constr_multiplier = channel_constr_multiplier
        if optim_algo is not None:
            self.optim_algo = optim_algo
        if maxeval is not None:
            self.maxeval = maxeval
        if constr_mode is not None:
            self.constr_mode = constr_mode

# Example usage:
if __name__ == "__main__":
    # Initialize BudgetAllocatorConfig
    budget_config: BudgetAllocatorConfig = BudgetAllocatorConfig(
        scenario="Q1 2024",
        total_budget=1000000.0,
        target_value=500000.0,
        date_range="2024-01-01 to 2024-03-31",
        channel_constr_low=0.1,
        channel_constr_up=0.5,
        channel_constr_multiplier=2,
        optim_algo="genetic",
        maxeval=1000,
        constr_mode="strict"
    )

    # Print the BudgetAllocatorConfig object
    print(budget_config)

    # Update the BudgetAllocatorConfig
    budget_config.update(
        total_budget=1200000.0,
        maxeval=1500
    )

    # Print the updated BudgetAllocatorConfig object
    print(budget_config)
