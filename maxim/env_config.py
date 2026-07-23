"""
Helpers for reading numeric tuning knobs from the environment.

These are operational backstops (cache TTLs, retention windows) that a very
small number of deployments need to override - e.g. workloads with legitimately
long-idle, human-in-the-loop traces. They are read lazily at construction time,
not import time, so a test or an embedding application can set them before the
component that reads them is created.
"""

import os
from typing import Optional

from .scribe import scribe


def env_int(name: str, default: int, minimum: int = 1) -> int:
    """
    Read a positive integer from environment variable ``name``.

    Falls back to ``default`` when the variable is unset, empty, non-numeric,
    or below ``minimum``. A malformed override is warned about rather than
    raised, so a bad env value can never take a logging integration down.
    """
    raw: Optional[str] = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        scribe().warning(
            "[MaximSDK] %s=%r is not an integer; using default %s.",
            name,
            raw,
            default,
        )
        return default
    if value < minimum:
        scribe().warning(
            "[MaximSDK] %s=%s is below the minimum of %s; using minimum.",
            name,
            value,
            minimum,
        )
        return minimum
    return value
