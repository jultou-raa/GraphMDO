"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Mapping
from typing import Any


class OptimizationConfigurationError(ValueError):
    """Raised when the optimization request is invalid for the current backend."""


class OptimizationExecutionError(RuntimeError):
    """Raised when optimization cannot produce a valid result.

    Attributes:
        partial_result: What the run produced before it was aborted, for
            example its trial history; ``None`` if nothing is available.
    """

    def __init__(
        self, message: str, *, partial_result: Mapping[str, Any] | None = None
    ) -> None:
        """Build the error.

        Args:
            message: Why the optimization failed.
            partial_result: What the run produced before it was aborted.
        """
        super().__init__(message)
        self.partial_result = partial_result
