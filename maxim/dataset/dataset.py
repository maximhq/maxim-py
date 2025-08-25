import json
from typing import Any, Optional, Union
from ..models.dataset import (
    ContextToEvaluateColumn,
    DatasetEntry,
    DatasetEntryWithRowNo,
    DataStructure,
    ExpectedOutputColumn,
    InputColumn,
    FileVariablePayload,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..apis.maxim_apis import MaximAPI



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
    data_structure: dict[str, Any], against_data_structure: dict[str, Any]
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


def _build_updates_for_attachments(
    entry_id_map: dict[str, "DatasetEntryWithRowNo"],
    uploaded_attachments: list[tuple[str, FileVariablePayload]],
) -> list[dict[str, Any]]:
    """
    Build the updates payload for uploaded file attachments.

    Args:
        entry_id_map: Mapping from entryId to the original `DatasetEntryWithRowNo` used to derive column names
        uploaded_attachments: List of tuples of (entryId, uploaded_attachment_payload)

    Returns:
        A list of update dicts conforming to the API schema
    """
    updates: list[dict[str, Any]] = []
    for entry_id, uploaded_attachment in uploaded_attachments:
        # Create the file variable payload according to the API schema
        file_variable_payload = {
            "text": uploaded_attachment.text,
            "files": [file.to_dict() for file in uploaded_attachment.files],
            "entryId": uploaded_attachment.entry_id,
        }

        # Find the column name from the original attachment entry
        attachment_entry = entry_id_map[entry_id]
        column_name = attachment_entry.column_name

        # Create update entry matching the API schema
        update_entry: dict[str, Any] = {
            "entryId": entry_id,
            "columnName": column_name,
            "value": {"type": "file", "payload": file_variable_payload},
        }
        updates.append(update_entry)

    return updates


def add_entries(api: "MaximAPI", dataset_id: str, dataset_entries: list[Union[DatasetEntry, dict[str, Any]]]) -> dict[str, Any]:
    """
    Add entries to a dataset.

    Args:
        api (MaximAPI): The MaximAPI instance to use for API calls
        dataset_id (str): The ID of the dataset to add entries to
        dataset_entries (list[DatasetEntry | dict[str, Any]]): 
            Entries to add. Each item can be a DatasetEntry or a dict convertible via DatasetEntry.from_dict.

    Returns:
        dict[str, Any]: Response data from the API

    Raises:
        TypeError: If entry type is not DatasetEntry or dict
        Exception: If API call fails
    """
    total_rows = api.get_dataset_total_rows(dataset_id)

    converted_entries: list[DatasetEntry] = []
    for entry in dataset_entries:
        if isinstance(entry, DatasetEntry):
            converted_entries.append(entry)
        elif isinstance(entry, dict):
            # Convert dictionary to DatasetEntry using the new from_dict method
            converted_entries.append(DatasetEntry.from_dict(entry))
        else:
            raise TypeError(f"Invalid entry type: {type(entry).__name__}. Expected DatasetEntry or dict.")

    entries_with_row_no: list[DatasetEntryWithRowNo] = []
    for i, entry in enumerate(converted_entries):
        entries_with_row_no.extend(DatasetEntryWithRowNo.from_dataset_entry(entry, i + 1 + total_rows))

    response_data = api.create_dataset_entries(
        dataset_id=dataset_id,
        entries=[entry.to_dict() for entry in entries_with_row_no],
    )
    # Map entry IDs to attachment entries using simple nested loops
    cells = response_data.get("data", {}).get("cells", []) if isinstance(response_data, dict) else []

    entry_id_map: dict[str, DatasetEntryWithRowNo] = {}
    cell_lookup = {
        (cell.get("columnName"), cell.get("rowNo")): cell
        for cell in cells
        if cell.get("entryId")
    }

    attachment_queue: list[DatasetEntryWithRowNo] = []
    for entry in entries_with_row_no:
        if entry.type == "file":
            attachment_queue.append(entry)

    for attachment_entry in attachment_queue:
        cell = cell_lookup.get((attachment_entry.column_name, attachment_entry.row_no))
        if cell:
            entry_id = cell.get("entryId")
            column_id = cell.get("columnId")
            if column_id:
                attachment_entry.column_id = column_id
            if entry_id:
                entry_id_map[entry_id] = attachment_entry


    uploaded_attachments = []
    for entry_id, attachment_entry in entry_id_map.items():
        uploaded_attachment = api.upload_dataset_entry_attachments(dataset_id, entry_id, attachment_entry)
        if uploaded_attachment:
            uploaded_attachments.append((entry_id, uploaded_attachment))

    # Update the dataset with the uploaded attachments
    if uploaded_attachments:
        updates = _build_updates_for_attachments(entry_id_map, uploaded_attachments)
        api.update_dataset_entries(dataset_id, updates)
        if isinstance(response_data, dict):
            response_data["attachmentsUpdated"] = True
    return response_data
