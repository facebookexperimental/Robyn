#pyre-strict

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

@dataclass(frozen=True)
class Hyperparameters:
    """
    Hyperparameters is an immutable data class that holds a dictionary of hyperparameters for multiple channels.

    Attributes:
        hyperparameters (Dict[str, Hyperparameter]): A dictionary of hyperparameters where the key is the channel name and the value is a Hyperparameter object.
    """
    hyperparameters: Dict[str, Hyperparameter] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Hyperparameters(\n"
            + "\n".join(f"  {channel}={hyperparameter}" for channel, hyperparameter in self.hyperparameters.items())
            + "\n)"
        )

    def get_hyperparameter(self, channel: str) -> Hyperparameter:
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
