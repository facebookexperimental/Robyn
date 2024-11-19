# pyre-strict

import logging

from typing import Optional
import multiprocessing
from typing_extensions import Final


class CommonUtils:
    """Utility class for common utilities."""

    # Constants
    MIN_CORES: Final[int] = 1

    @staticmethod
    def get_cores_available(requested_cores: Optional[int] = None) -> int:
        """
        Determines the number of CPU cores to use based on available cores and requested cores.

        Args:
            requested_cores: Optional number of cores requested. If None, uses all available cores.

        Returns:
            int: Number of cores to use

        Raises:
            ValueError: If requested_cores is less than 1
        """
        logger = logging.getLogger(__name__)
        available_cores = multiprocessing.cpu_count()

        # If no cores requested, use all available
        if requested_cores is None:
            return available_cores

        # Validate requested cores
        if requested_cores < CommonUtils.MIN_CORES:
            requested_cores = CommonUtils.MIN_CORES
            logger.warning(
                "Requested cores must be at least %d. Got: %d. Will use %d cores.",
                CommonUtils.MIN_CORES,
                requested_cores,
                CommonUtils.MIN_CORES,
            )

        # Log warning if requested cores exceed available cores
        if requested_cores > available_cores:
            logger.warning(
                "Requested cores (%d) exceeds available cores (%d). Will use %d cores.",
                requested_cores,
                available_cores,
                available_cores,
            )

        # Return minimum of requested vs available cores
        return min(requested_cores, available_cores)
