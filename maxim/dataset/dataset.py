import json
from typing import Any, Dict, Optional

from ..models.dataset import (
    ContextToEvaluateColumn,
    DataStructure,
    ExpectedOutputColumn,
    InputColumn,
)


def create_data_structure(data_structure: DataStructure) -> DataStructure:
    """Create and validate a data structure.

    Takes a data structure, sanitizes it to ensure it meets validation requirements,
    and returns the sanitized data structure.

    Args:
        data_structure (DataStructure): The data structure to create and validate.

    Returns:
        DataStructure: The validated data structure.

    Raises:
        Exception: If the data structure contains validation errors (e.g., multiple
            input columns, multiple expected output columns, or multiple context
            to evaluate columns).
    """
    sanitize_data_structure(data_structure)
    return data_structure


def sanitize_data_structure(data_structure: Optional[DataStructure]) -> None:
    """Sanitize and validate a data structure for correctness.

    Ensures that the data structure contains at most one of each required column type:
    - InputColumn: Only one input column is allowed
    - ExpectedOutputColumn: Only one expected output column is allowed
    - ContextToEvaluateColumn: Only one context to evaluate column is allowed

    Args:
        data_structure (Optional[DataStructure]): The data structure to sanitize.
            Can be None, in which case no validation is performed.

    Raises:
        Exception: If the data structure contains more than one input column,
            more than one expected output column, or more than one context
            to evaluate column. The exception includes the full data structure
            for debugging purposes.
    """
    encountered_input = False
    encountered_expected_output = False
    encountered_context_to_evaluate = False
    if data_structure:
        for value in data_structure.values():
            if value == InputColumn:
                if encountered_input:
                    raise Exception(
                        "Data structure contains more than one input",
                        {"dataStructure": json.dumps(data_structure, indent=2)},
                    )
                else:
                    encountered_input = True
            elif value == ExpectedOutputColumn:
                if encountered_expected_output:
                    raise Exception(
                        "Data structure contains more than one expectedOutput",
                        {"dataStructure": json.dumps(data_structure, indent=2)},
                    )
                else:
                    encountered_expected_output = True
            elif value == ContextToEvaluateColumn:
                if encountered_context_to_evaluate:
                    raise Exception(
                        "Data structure contains more than one contextToEvaluate",
                        {"dataStructure": json.dumps(data_structure, indent=2)},
                    )
                else:
                    encountered_context_to_evaluate = True


def validate_data_structure(
    data_structure: Dict[str, Any], against_data_structure: Dict[str, Any]
) -> None:
    """Validate that a data structure matches the expected structure schema.

    Ensures that all keys present in the provided data structure also exist
    in the reference data structure (typically from the platform/dataset).
    This prevents attempting to use columns that don't exist in the target dataset.

    Args:
        data_structure (Dict[str, Any]): The data structure to validate.
        against_data_structure (Dict[str, Any]): The reference data structure
            to validate against (e.g., from the platform dataset).

    Raises:
        Exception: If the provided data structure contains any keys that are
            not present in the reference data structure. The exception includes
            both the provided keys and the expected keys for debugging.
    """
    data_structure_keys = set(data_structure.keys())
    against_data_structure_keys = set(against_data_structure.keys())
    for key in data_structure_keys:
        if key not in against_data_structure_keys:
            raise Exception(
                f"The provided data structure contains key '{key}' which is not present in the dataset on the platform",
                {
                    "providedDataStructureKeys": list(data_structure_keys),
                    "platformDataStructureKeys": list(against_data_structure_keys),
                },
            )
