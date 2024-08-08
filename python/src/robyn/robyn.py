# pyre-strict


#This needs to be rewritten to match the new structure of the codebase

from robyn.analysis.budget_allocation_result import BudgetAllocationResult
from robyn.analysis.budgetallocator_config import BudgetAllocatorConfig
from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.modeling.entities.modeloutput import ModelOutput
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig


class Robyn:
    def __init__(self, working_dir: str):
        """
        Initializes the Robyn object with a working directory.

        Args:
            working_dir (str): The path to the working directory.
        """
        self.working_dir = working_dir

    # Load input data for the first time and validates
    def initialize(
        self,
        mmm_data: MMMData,
        holidays_data: HolidaysData,
        hyperparameters: HyperParametersConfig,
        calibration_input: CalibrationInputConfig,
    ) -> MMMDataCollection:
        """
        Loads input data for the first time and validates it.

        Args:
            mmm_data (MMMData): The MMM data object.
            holidays_data (HolidaysData): The holidays data object.
            hyperparameters (HyperParametersConfig): The hyperparameters configuration object.
            calibration_input (CalibrationInputConfig): The calibration input configuration object.
        """
        pass

    # Load previous state from Json file
    def reload_from_json(self, mmmdata_collection_json_file: str) -> None:
        """
        Loads the previous state from a JSON file and validates it.

        Args:
            mmmdata_collection_json_file (str): The path to the JSON file containing the previous state.
        """
        pass

    # Run models for all trials and iterations, using num of cores
    def model_run(
        self,
        num_of_cores: int,
        trials_config: TrialsConfig,
    ) -> ModelOutput:
        """
        Runs the models for all trials and iterations using the specified number of cores.

        Args:
            mmmdata_collection (MMMDataCollection): The MMM data collection object.
            num_of_cores (int): The number of cores to use for running the models.
            trials_config (TrialsConfig): The trials configuration object.

        Returns:
            OutputModels: The output models object.
        """
        pass

    # Run budget allocator for given MMMDataCollection and ModelOutputsCollection
    def budget_allocator(
        self,
        select_model: str,
        budger_allocator_config: BudgetAllocatorConfig,
    ) -> BudgetAllocationResult:
        """
        Runs the budget allocator for the given MMMDataCollection and ModelOutputsCollection.

        Args:
            mmmdata_colllection (MMMDataCollection): The MMM data collection object.
            model_output_collection (ModelOutputsCollection): The model output collection object.
            select_model (str): The selected model to use for budget allocation.
            budger_allocator_config (BudgetAllocatorConfig): The budget allocator configuration object.

        Returns:
            BudgetAllocationResult: The budget allocation result object.
        """
        pass

    # Run budget allocator for given robyn_object?
    def budget_allocator(
        self,
        robyn_object: object,
        select_model: str,
        budger_allocator_config: BudgetAllocatorConfig,
    ) -> BudgetAllocationResult:
        """
        Runs the budget allocator for the given robyn_object.

        Args:
            robyn_object (object): The robyn object to use for budget allocation.
            select_model (str): The selected model to use for budget allocation.
            budger_allocator_config (BudgetAllocatorConfig): The budget allocator configuration object.

        Returns:
            BudgetAllocationResult: The budget allocation result object.
        """
        pass

    # Run budget allocator - load from json file? # Using json file from robyn_write() for allocation
    def budget_allocator(
        self,
        robyn_object_json: str,
        select_model: str,
        budger_allocator_config: BudgetAllocatorConfig,
    ) -> BudgetAllocationResult:
        """
        Runs the budget allocator using the specified JSON file.

        Args:
            robyn_object_json (str): The path to the JSON file containing the robyn object.
            select_model (str): The selected model to use for budget allocation.
            budger_allocator_config (BudgetAllocatorConfig): The budget allocator configuration object.

        Returns:
            BudgetAllocationResult: The budget allocation result object.
        """
        pass

    # Refresh
    def model_refresh(
        self,
        mmmdata_colllection: MMMDataCollection,
        model_output_collection: ModelOutputsCollection,
        refresh_config: ModelRefreshConfig,
        calibration_input: CalibrationInputConfig = None,
        objective_weights=None,
    ) -> BudgetAllocationResult:
        """
        Refreshes the model using the specified configuration and input.

        Args:
            mmmdata_colllection (MMMDataCollection): The MMM data collection object.
            model_output_collection (ModelOutputsCollection): The model output collection object.
            refresh_config (ModelRefreshConfig): The refresh configuration object.
            calibration_input (CalibrationInputConfig, optional): The calibration input configuration object. Defaults to None.
            objective_weights (dict, optional): The objective weights dictionary. Defaults to None.

        Returns:
            BudgetAllocationResult: The budget allocation result object.
        """
        pass

    # def model_refresh(self, robyn_object? -> ?, refresh_config:ModelRefreshConfig, calibration_input:CalibrationInputConfig=None, objective_weights=None) -> ? :
    #         pass
    # def model_refresh(seld, json_file -> ?, refresh_config:ModelRefreshConfig, calibration_input:CalibrationInputConfig=None, objective_weights=None) -> ? :
    #     pass

    def robyn_response(
        mmm_data_collection: MMMDataCollection = None,
        model_output_collection: ModelOutputCollection = None,
        select_build=None,
        select_model=None,
        metric_name=None,
        metric_value=None,
        date_range=None,
        dt_hyppar=None,
        dt_coef=None,
        quiet=False,
    ):
        """
        Generates a response for the given input parameters.

        Args:
            mmm_data_collection (MMMDataCollection, optional): The MMM data collection object. Defaults to None.
            model_output_collection (ModelOutputCollection, optional): The model output collection object. Defaults to None.
            select_build (_type_, optional): The selected build. Defaults to None.
            select_model (_type_, optional): The selected model. Defaults to None.
            metric_name (_type_, optional): The metric name. Defaults to None.
            metric_value (_type_, optional): The metric value. Defaults to None.
            date_range (_type_, optional): The date range. Defaults to None.
            dt_hyppar (_type_, optional): The dt hyppar. Defaults to None.
            dt_coef (_type_, optional): The dt coef. Defaults to None.
            quiet (bool, optional): Whether to suppress output. Defaults to False.
        """
        pass

    def robyn_response(
        json_file=None,
        select_build=None,
        select_model=None,
        metric_name=None,
        metric_value=None,
        date_range=None,
        dt_hyppar=None,
        dt_coef=None,
        quiet=False,
    ):
        """
        Generates a response for the given input parameters using a JSON file.

        Args:
            json_file (_type_, optional): The path to the JSON file. Defaults to None.
            select_build (_type_, optional): The selected build. Defaults to None.
            select_model (_type_, optional): The selected model. Defaults to None.
            metric_name (_type_, optional): The metric name. Defaults to None.
            metric_value (_type_, optional): The metric value. Defaults to None.
            date_range (_type_, optional): The date range. Defaults to None.
            dt_hyppar (_type_, optional): The dt hyppar. Defaults to None.
            dt_coef (_type_, optional): The dt coef. Defaults to None.
            quiet (bool, optional): Whether to suppress output. Defaults to False.
        """
        pass

    def robyn_response(
        robyn_object=None,
        select_build=None,
        select_model=None,
        metric_name=None,
        metric_value=None,
        date_range=None,
        dt_hyppar=None,
        dt_coef=None,
        quiet=False,
    ):
        """
        Generates a response for the given input parameters using a Robyn object.

        Args:
            robyn_object (_type_, optional): The Robyn object. Defaults to None.
            select_build (_type_, optional): The selected build. Defaults to None.
            select_model (_type_, optional): The selected model. Defaults to None.
            metric_name (_type_, optional): The metric name. Defaults to None.
            metric_value (_type_, optional): The metric value. Defaults to None.
            date_range (_type_, optional): The date range. Defaults to None.
            dt_hyppar (_type_, optional): The dt hyppar. Defaults to None.
            dt_coef (_type_, optional): The dt coef. Defaults to None.
            quiet (bool, optional): Whether to suppress output. Defaults to False.
        """
        pass
