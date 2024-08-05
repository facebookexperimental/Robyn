#pyre-strict

from typing import Optional, Dict, Any
from pyre_extensions import none_throws
import os
import re
import pandas as pd
import numpy as np

class MarketingMixModelResponse:
    def __init__(self) -> None:
        pass

    def get_response_from_inputcollect(
        self,
        data_collection: MMMDataCollection,
        model_output: ModelOutputCollection,
        select_model: str,
        metric_name: str,
        metric_value: Optional[float],
        date_range: Optional[str],
        dt_hyppar: Optional[pd.DataFrame] = None,
        dt_coef: Optional[pd.DataFrame] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        if dt_hyppar is None:
            dt_hyppar = model_output.get('resultHypParam')
        if dt_coef is None:
            dt_coef = model_output.get('xDecompAgg')
        if any(x is None for x in [dt_hyppar, dt_coef, data_collection, model_output]):
            raise ValueError("When 'robyn_object' is not provided, 'InputCollect' & 'OutputCollect' must be provided")

        return self._get_response(
            data_collection,
            select_model,
            metric_name,
            metric_value,
            date_range,
            dt_hyppar,
            dt_coef,
            quiet,
        )

    def get_response_from_robyn_object(
        self,
        robyn_object: str,
        select_build: Optional[int],
        select_model: str,
        metric_name: str,
        metric_value: Optional[float],
        date_range: Optional[str],
        dt_hyppar: Optional[pd.DataFrame] = None,
        dt_coef: Optional[pd.DataFrame] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        if not os.path.exists(robyn_object):
            raise FileNotFoundError(f"File does not exist or is somewhere else. Check: {robyn_object}")
        else:
            Robyn = readRDS(robyn_object)  # type: ignore
            objectPath = os.path.dirname(robyn_object)
            objectName = re.sub(r'\..*$', '', os.path.basename(robyn_object))

        select_build_all = range(len(Robyn))
        if select_build is None:
            select_build = max(select_build_all)
            if not quiet and len(select_build_all) > 1:
                print(f"Using latest model: {'initial model' if select_build == 0 else f'refresh model #{select_build}'} for the response function. Use parameter 'select_build' to specify which run to use")

        if select_build not in select_build_all or not isinstance(select_build, int):
            raise ValueError(f"'select_build' must be one value of {', '.join(map(str, select_build_all))}")

        listName = "listInit" if select_build == 0 else f"listRefresh{select_build}"
        data_collection = MMMDataCollection(Robyn[listName]["InputCollect"])
        model_output = ModelOutputCollection(Robyn[listName]["OutputCollect"])
        dt_hyppar = model_output.get('resultHypParam')
        dt_coef = model_output.get('xDecompAgg')

        return self._get_response(
            data_collection,
            select_model,
            metric_name,
            metric_value,
            date_range,
            dt_hyppar,
            dt_coef,
            quiet,
        )

    def get_response_from_json_file(
        self,
        json_file: str,
        select_model: str,
        metric_name: str,
        metric_value: Optional[float],
        date_range: Optional[str],
        dt_hyppar: Optional[pd.DataFrame] = None,
        dt_coef: Optional[pd.DataFrame] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        data_collection = MMMDataCollection(robyn_inputs(json_file=json_file))  # type: ignore
        model_output = ModelOutputCollection(robyn_run(InputCollect=data_collection, json_file=json_file, export=False, quiet=quiet))  # type: ignore
        if dt_hyppar is None:
            dt_hyppar = model_output.get('resultHypParam')
        if dt_coef is None:
            dt_coef = model_output.get('xDecompAgg')

        return self._get_response(
            data_collection,
            select_model,
            metric_name,
            metric_value,
            date_range,
            dt_hyppar,
            dt_coef,
            quiet,
        )

    def _get_response(
        self,
        data_collection: MMMDataCollection,
        select_model: str,
        metric_name: str,
        metric_value: Optional[float],
        date_range: Optional[str],
        dt_hyppar: pd.DataFrame,
        dt_coef: pd.DataFrame,
        quiet: bool,
    ) -> Dict[str, Any]:
        dt_input = data_collection.get("robyn_inputs")["dt_input"]
        startRW = data_collection.get("robyn_inputs")["rollingWindowStartWhich"]
        endRW = data_collection.get("robyn_inputs")["rollingWindowEndWhich"]
        adstock = data_collection.get("robyn_inputs")["adstock"]
        spendExpoMod = data_collection.get("robyn_inputs")["modNLS"]["results"]
        paid_media_vars = data_collection.get("robyn_inputs")["paid_media_vars"]
        paid_media_spends = data_collection.get("robyn_inputs")["paid_media_spends"]
        exposure_vars = data_collection.get("robyn_inputs")["exposure_vars"]
        organic_vars = data_collection.get("robyn_inputs")["organic_vars"]
        allSolutions = dt_hyppar['solID'].unique()
        dayInterval = data_collection.get("robyn_inputs")["dayInterval"]

        # ... (rest of the implementation remains the same)

        ret = {
            'metric_name': metric_name,
            'date': date_range_updated,
            'input_total': input_total,
            'input_carryover': input_carryover,
            'input_immediate': input_immediate,
            'response_total': response_total,
            'response_carryover': response_carryover,
            'response_immediate': response_immediate,
            'usecase': usecase,
            'plot': None
        }
        return ret
