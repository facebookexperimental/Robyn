# pyre-strict

from dataclasses import dataclass, field
from typing import Dict, List

from robyn.data.entities.enums import AdstockType


@dataclass
class ChannelHyperparameters:
    """
    ChannelHyperparameters is an immutable data class that holds the hyperparameters for a model.

    Attributes:
        thetas (List[float]): List of theta values.
        shapes (List[float]): List of shape values.
        scales (List[float]): List of scale values.
        alphas (List[float]): List of alpha values.
        gammas (List[float]): List of gamma values.
        penalty (List[float]): List of penalty values.
    """

    thetas: List[float] = None  # if adstock is geometric
    shapes: List[float] = None  # if adstock is weibull
    scales: List[float] = None  # if adstock is weibull
    alphas: List[float] = None  # Mandatory
    gammas: List[float] = None  # Mandatory
    penalty: List[bool] = (
        None  # optional. User only provides if they want to use it. They don't provide values. Model run calculates it.
    )

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


@dataclass
class Hyperparameters:
    """
    Hyperparameters is an immutable data class that holds a dictionary of hyperparameters for multiple channels.

    Attributes:
        hyperparameters (Dict[str, Hyperparameter]): A dictionary of hyperparameters where the key is the channel name and the value is a Hyperparameter object.
    """

    hyperparameters: Dict[str, ChannelHyperparameters] = field(default_factory=dict)
    adstock: AdstockType = AdstockType.GEOMETRIC  # Mandatory. User provides this.
    lambda_: float = 0.0  # User does not provide this. Model run calculates it.
    train_size: List[float] = field(default_factory=lambda: [0.5, 0.8])
    hyper_bound_list_updated: Dict[str, List[float]] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Hyperparameters(\n"
            + "\n".join(
                f"  {channel}={hyperparameter}"
                for channel, hyperparameter in self.hyperparameters.items()
            )
            + "\n)"
        )

    def copy(self):
        # Create a new instance with the same data
        return Hyperparameters(
            hyperparameters=self.hyperparameters.copy(),
            adstock=self.adstock,
            lambda_=self.lambda_,
            train_size=self.train_size.copy() if self.train_size else None,
        )

    def __post_init__(self):
        self.update_hyper_bounds()

    def update_hyper_bounds(self):
        self.hyper_bound_list_updated = {}
        for key, value in self.hyperparameters.items():
            if isinstance(value, list) and len(value) == 2:
                self.hyper_bound_list_updated[key] = value
        if isinstance(self.lambda_, list) and len(self.lambda_) == 2:
            self.hyper_bound_list_updated["lambda"] = self.lambda_

    def get_hyperparameter(self, channel: str) -> ChannelHyperparameters:
        """
        Get the hyperparameter for a specific channel.

        Args:
        channel (str): The name of the channel.

        Returns:
        Hyperparameter: The hyperparameter for the specified channel.

        Raises:
        KeyError: If the channel is not found in the hyperparameters dictionary.
        """
        return self.hyperparameters[channel]

    def has_channel(self, channel: str) -> bool:
        """
        Check if a channel exists in the hyperparameters dictionary.

        Args:
        channel (str): The name of the channel.

        Returns:
        bool: True if the channel exists, False otherwise.
        """
        return channel in self.hyperparameters

    @staticmethod
    def get_hyperparameter_limits() -> Dict[str, List[str]]:
        """
        Returns the hyperparameter limits.
        Returns:
            pd.DataFrame: The hyperparameter limits.
        """
        return {
            "thetas": [">=0", "<1"],
            "alphas": [">0", "<10"],
            "gammas": [">0", "<=1"],
            "shapes": [">=0", "<20"],
            "scales": [">=0", "<=1"],
        }

    @property
    def hyper_list_all(self):
        return {
            **self.hyperparameters,
            "lambda": self.lambda_,
            "train_size": self.train_size,
        }

    @property
    def all_fixed(self):
        return len(self.hyper_bound_list_updated) == 0
