# pyre-strict

from typing import Dict, Any
from typing import Optional

class BudgetAllocator:
    def budget_allocator(
        self,
        mmmdata_collection: MMMDataCollection,
        model_output_collection: ModelOutputsCollection,
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

    def budget_allocator(
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

    def budget_allocator(
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
