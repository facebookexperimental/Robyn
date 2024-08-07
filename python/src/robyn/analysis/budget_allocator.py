# pyre-strict

from typing import Dict, Any, List
from typing import Optional

from robyn.analysis.budget_allocation_result import BudgetAllocationResult
from robyn.analysis.budgetallocator_config import BudgetAllocatorConfig
from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.modeling.entities.modeloutput_collection import ModelOutputCollection

class BudgetAllocator:
    def budget_allocator(
        self,
        mmmdata_collection: MMMDataCollection,
        model_output_collection: ModelOutputCollection,
        select_model: str,
        budget_allocator_config: BudgetAllocatorConfig
    ) -> BudgetAllocationResult:
        """
        Run budget allocator for given MMMDataCollection and ModelOutputsCollection.

        :param mmmdata_collection: Collection of MMM data.
        :param model_output_collection: Collection of model outputs.
        :param select_model: The model to select for allocation.
        :param budget_allocator_config: Configuration for the budget allocator.
        :return: The result of the budget allocation.
        """
        pass

    def budget_allocator_from_robyn_object(
        self,
        robyn_object: Dict[str, Any],
        select_model: str,
        budget_allocator_config: BudgetAllocatorConfig
    ) -> BudgetAllocationResult:
        """
        Run budget allocator for given robyn_object.

        :param robyn_object: Dictionary containing the Robyn object.
        :param select_model: The model to select for allocation.
        :param budget_allocator_config: Configuration for the budget allocator.
        :return: The result of the budget allocation.
        """
        pass

    def budget_allocator_from_json(
        self,
        json_file: str,
        select_model: str,
        budget_allocator_config: BudgetAllocatorConfig
    ) -> BudgetAllocationResult:
        """
        Run budget allocator - load from json file.

        :param json_file: Path to the JSON file containing the model configuration.
        :param select_model: The model to select for allocation.
        :param budget_allocator_config: Configuration for the budget allocator.
        :return: The result of the budget allocation.
        """
        pass

    def _set_constraints(
        self,
        channel_constr_low: Optional[List[float]],
        channel_constr_up: Optional[List[float]]
    ) -> None:
        # Corresponds to constraint setting in robyn_allocator()
        # Implementation details...
        pass

    def _optimize_allocation(
        self,
        scenario: str,
        total_budget: Optional[float],
        target_value: Optional[float],
        optim_algo: str,
        maxeval: int,
        constr_mode: str
    ) -> Dict[str, Any]:
        # Corresponds to optimization logic in robyn_allocator()
        # Implementation details...
        pass

    def _get_hill_params(self) -> Dict[str, Any]:
        # Corresponds to get_hill_params() function
        # Implementation details...
        pass

    def _fx_objective(
        self,
        x: float,
        coeff: float,
        alpha: float,
        inflexion: float,
        x_hist_carryover: float,
        get_sum: bool = False
    ) -> float:
        # Corresponds to fx_objective() function
        # Implementation details...
        pass
