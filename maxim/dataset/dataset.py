import json
from typing import Any, Dict, Optional

from ..models.dataset import (
    ContextToEvaluateColumn,
    DataStructure,
    ExpectedOutputColumn,
    InputColumn,
)


def create_data_structure(data_structure: DataStructure) -> DataStructure:
    sanitize_data_structure(data_structure)
    return data_structure


def sanitize_data_structure(data_structure: Optional[DataStructure]) -> None:
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
