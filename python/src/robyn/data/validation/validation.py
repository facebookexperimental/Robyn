from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationResult:
    """Result of a validation operation"""

    status: bool  # True if validation was successful, False otherwise
    error_message: Optional[str]  # Error message if validation failed
    error_details: Optional[dict]  # Additional error details if validation failed


class Validation(ABC):
    """Abstract base class for validation operations"""

    @abstractmethod
    def validate(self) -> List[ValidationResult]:
        """Perform validation and return the result"""
