from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class Hyperparameter:
    """
    Hyperparameter is an immutable data class that holds the hyperparameters for a model.

    Attributes:
        thetas (List[float]): List of theta values.
        shapes (List[float]): List of shape values.
        scales (List[float]): List of scale values.
        alphas (List[float]): List of alpha values.
        gammas (List[float]): List of gamma values.
        penalty (List[float]): List of penalty values.
    """
    thetas: List[float] = field(default_factory=list)
    shapes: List[float] = field(default_factory=list)
    scales: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    gammas: List[float] = field(default_factory=list)
    penalty: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Hyperparameter(\n"
            f"  thetas={self.thetas},\n"
            f"  shapes={self.shapes},\n"
            f"  scales={self.scales},\n"
            f"  alphas={self.alphas},\n"
            f"  gammas={self.gammas},\n"
            f"  penalty={self.penalty}\n"
            f")"
        )
