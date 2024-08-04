# pyre-strict

from typing import Dict, Any, Optional, TypedDict

class HyperParameters(TypedDict, total=False):
    thetas: List[float]
    shapes: List[float]
    scales: List[float]
    alphas: List[float]
    gammas: List[float]
    penalty: List[float]
    
class HyperParametersConfig:
    def __init__(self, hyperparameters: Optional[HyperParameters] = None) -> None:
        self.hyperparameters: HyperParameters = hyperparameters if hyperparameters is not None else {}

    def set_hyperparameter(self, key: str, value: Any) -> None:
        if key in HyperParameters.__annotations__:
            self.hyperparameters[key] = value
        else:
            raise KeyError(f"{key} is not a valid hyperparameter")

    def get_hyperparameter(self, key: str) -> Any:
        return self.hyperparameters.get(key)

    def __str__(self) -> str:
        return f"HyperParametersConfig: {self.hyperparameters}"
