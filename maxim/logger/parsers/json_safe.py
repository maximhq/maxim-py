"""
Shared conversions for values that json.dumps cannot encode natively.

Two serializers in the SDK need these: ``default_json_serializer`` (model
parameters) and ``CustomEncoder`` (the log payload that goes over the wire).
They deliberately differ in how they treat *objects* - model parameters are
metadata about a call and are recorded by type name, while payload content
(messages, results) must be expanded - but the scalar conversions below have
to be identical in both. Keeping them in one place is what stops a type from
being handled on one path and raising on the other, which is how
"Object of type mappingproxy is not JSON serializable" survived a fix that
only touched the model-parameter path.
"""

import datetime
import decimal
import enum
import pathlib
import types
import uuid
from typing import Any

# Returned when a value needs no scalar conversion, so callers can distinguish
# "not handled here" from a legitimate None/falsy conversion result.
UNHANDLED = object()


def json_safe_scalar(o: Any) -> Any:
    """
    Convert a value with no native JSON representation into one that has.

    Returns UNHANDLED if the value is not one of these types, leaving the
    caller to apply its own policy for objects.
    """
    if isinstance(o, enum.Enum):
        return o.value
    # Read-only dict wrappers (e.g. a class' __dict__) are not JSON
    # serializable by default.
    if isinstance(o, types.MappingProxyType):
        return dict(o)
    if isinstance(o, (datetime.datetime, datetime.date, datetime.time)):
        return o.isoformat()
    if isinstance(o, datetime.timedelta):
        return o.total_seconds()
    if isinstance(o, decimal.Decimal):
        return float(o)
    if isinstance(o, uuid.UUID):
        return str(o)
    if isinstance(o, pathlib.PurePath):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    return UNHANDLED
