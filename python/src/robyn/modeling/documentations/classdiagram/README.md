```mermaid
classDiagram
    class MMMDataCollection {
        +dt_input: DataFrame
        +dt_holidays: DataFrame
        +mmmdata_spec: MMMDataSpec
        +data: DataFrame
    }

    class ModelOutputCollection {
        +resultHypParam: ResultHypParam
        +xDecompAgg: XDecompAgg
        +mediaVecCollect: DataFrame
        +xDecompVecCollect: DataFrame
        +resultCalibration: DataFrame
        +model_output: ModelOutput
        +allSolutions: List[str]
        +allPareto: Dict
        +calibration_constraint: float
        +pareto_fronts: int
        +selectID: str
        +cores: int
        +iterations: int
        +trials: List
        +intercept_sign: str
        +nevergrad_algo: str
        +add_penalty_factor: bool
        +seed: int
        +hyper_fixed: bool
        +hyper_updated: Dict
        +update(kwargs)
    }

    class ModelOutput {
        +trials: List[Trial]
        +metadata: Metadata
        +seed: int
        +create(model_output_dict: Dict) ModelOutput
    }

    class FeatureEngineering {
        -mmm_data_collection: MMMDataCollection
        +feature_engineering(quiet: bool) MMMDataCollection
        -__prepare_data() DataFrame
        -__create_rolling_window_data(dt_transform: DataFrame) DataFrame
        -__calculate_media_cost_factor(dt_input_roll_wind: DataFrame) List[float]
        -__fit_spend_exposure(dt_spend_mod_input: DataFrame, media_cost_factor: float, media_var: str) Dict
        -__run_models(dt_input_roll_wind: DataFrame, media_cost_factor: List[float]) Tuple
    }

    class MMMModelExecutor {
        +model_run(mmmdata_collection: MMMDataCollection, ...)
        +print_robyn_models(x: Any)
        +model_train(resultHyper, hyper_collect: Dict, ...)
        +run_nevergrad_optimization(mmmdata_collection: MMMDataCollection, ...)
        -model_fit_iteration(iteration: int, ...) ModelOutputTrialResult
        +model_decomp(coefs: Any, y_pred: Any, ...)
        +model_refit(x_train: Any, y_train: Any, ...)
        -__get_rsq(true: Any, predicted: Any, ...) float
        -__lambda_seq(x: Any, y: Any, ...) Any
        -__hyper_collector(InputCollect: Dict, ...)
        -__init_msgs_run(InputCollect: Dict, ...)
    }

    class ParetoOptimizer {
        +pareto_optimize(mmmdata_collection: MMMDataCollection, modeloutput: ModelOutput, ...) Dict
        +pareto_front(x: np.ndarray, y: np.ndarray, ...) DataFrame
        +get_pareto_fronts(pareto_fronts: Union[str, int]) int
        +run_dt_resp(respN: int, mmmdata_collection: MMMDataCollection, ...) DataFrame
    }

    class ModelClustersAnalyzer {
        +model_clusters_analyze(input: Dict, dep_var_type: str, ...) Dict
        -_determine_optimal_k(df: DataFrame, max_clusters: int, ...) int
        -_clusterKmeans_auto(df: DataFrame, ...) Tuple
        -_plot_wss_and_save(wss: List[float], path: str, ...)
        -_prepare_df(x: DataFrame, all_media: List[str], ...) DataFrame
        -_clusters_df(df: DataFrame, all_paid: List[str], ...) DataFrame
        -_confidence_calcs(xDecompAgg: DataFrame, df: DataFrame, ...) Dict
        -_plot_clusters_ci(sim_collect: DataFrame, df_ci: DataFrame, ...) Any
    }

    class ModelRefresh {
        +model_refresh(mmmdata_collection: MMMDataCollection, model_output_collection: ModelOutputCollection, ...) Any
        +model_refresh_from_robyn_object(robyn_object: Dict, refresh_config: ModelRefreshConfig, ...) Any
        +model_refresh_from_reloadedstate(json_file: str, refresh_config: ModelRefreshConfig, ...) Any
    }

    class ModelResponse {
        +robyn_response(InputCollect, OutputCollect, ...)
        +robyn_response_from_robyn_object(robyn_object, ...)
        +robyn_response_from_json(json_file, ...)
    }

    MMMDataCollection -- ModelOutputCollection
    ModelOutputCollection -- ModelOutput
    FeatureEngineering -- MMMDataCollection
    MMMModelExecutor -- MMMDataCollection
    MMMModelExecutor -- ModelOutputCollection
    ParetoOptimizer -- MMMDataCollection
    ParetoOptimizer -- ModelOutput
    ModelClustersAnalyzer -- ModelOutputCollection
    ModelRefresh -- MMMDataCollection
    ModelRefresh -- ModelOutputCollection
    ModelResponse -- MMMDataCollection
    ModelResponse -- ModelOutputCollection
