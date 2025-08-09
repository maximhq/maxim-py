import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from ..models.dataset import (
    ContextToEvaluateColumn,
    DatasetEntry,
    DatasetEntryWithRowNo,
    DataStructure,
    ExpectedOutputColumn,
    InputColumn,
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


def _map_attachment_entries_to_entry_ids(
    cells: List[Dict[str, Any]],
    attachment_queue: List[DatasetEntryWithRowNo],
) -> Dict[str, DatasetEntryWithRowNo]:
    """
    Build a mapping from entryId -> DatasetEntryWithRowNo for file attachments.

    Prefers direct (columnName, rowNo) matches; falls back to relative index
    matching within a column when row numbers are not reliably present in the response.
    """
    # Precompute lookups
    cell_lookup: Dict[tuple[str, int], str] = {}
    column_entries: Dict[str, List[str]] = defaultdict(list)

    for cell in cells:
        entry_id = cell.get("entryId")
        column_name = cell.get("columnName")
        row_no = cell.get("rowNo")
        if entry_id and column_name:
            if isinstance(row_no, int):
                cell_lookup[(column_name, row_no)] = entry_id
            if entry_id not in column_entries[column_name]:
                column_entries[column_name].append(entry_id)

    # Precompute min row number per column for fallback
    min_row_for_column: Dict[str, int] = {}
    for entry in attachment_queue:
        col = entry.column_name
        current_min = min_row_for_column.get(col)
        if current_min is None or entry.row_no < current_min:
            min_row_for_column[col] = entry.row_no

    # Match entries in a single pass
    entry_id_map: Dict[str, DatasetEntryWithRowNo] = {}
    for attachment_entry in attachment_queue:
        column_name = attachment_entry.column_name
        row_no = attachment_entry.row_no

        # Primary: direct match via (column, row)
        matching_entry_id = cell_lookup.get((column_name, row_no))

        # Fallback: match by relative index in column_entries
        if not matching_entry_id and column_name in column_entries and column_name in min_row_for_column:
            relative_index = row_no - min_row_for_column[column_name]
            entries = column_entries[column_name]
            if 0 <= relative_index < len(entries):
                matching_entry_id = entries[relative_index]

        if matching_entry_id and matching_entry_id not in entry_id_map:
            entry_id_map[matching_entry_id] = attachment_entry

    return entry_id_map


def add_entries(api: "MaximAPI", dataset_id: str, dataset_entries: Union[List[DatasetEntry], List[Dict[str, Any]]]) -> dict[str, Any]:
    """
    Add entries to a dataset.

    Args:
        api (MaximAPI): The MaximAPI instance to use for API calls
        dataset_id (str): The ID of the dataset to add entries to
        dataset_entries (Union[List[DatasetEntry], List[Dict[str, Any]]]): 
            List of dataset entries to add. Can be DatasetEntry objects or dictionaries.

    Returns:
        dict[str, Any]: Response data from the API

    Raises:
        TypeError: If entry type is not DatasetEntry or dict
        Exception: If API call fails
    """
    total_rows = api.get_dataset_total_rows(dataset_id)

    converted_entries: List[DatasetEntry] = []
    for entry in dataset_entries:
        if isinstance(entry, DatasetEntry):
            converted_entries.append(entry)
        elif isinstance(entry, dict):
            # Convert dictionary to DatasetEntry using the new from_dict method
            converted_entries.append(DatasetEntry.from_dict(entry))
        else:
            raise TypeError(f"Invalid entry type: {type(entry).__name__}. Expected DatasetEntry or dict.")

    entries_with_row_no: List[DatasetEntryWithRowNo] = []
    for i, entry in enumerate(converted_entries):
        entries_with_row_no.extend(DatasetEntryWithRowNo.from_dataset_entry(entry, i + 1 + total_rows))

    attachment_queue: List[DatasetEntryWithRowNo] = []
    for entry in entries_with_row_no:
        if entry.type == "file":
            attachment_queue.append(entry)

    response_data = api.create_dataset_entries(
        dataset_id=dataset_id,
        entries=[entry.to_dict() for entry in entries_with_row_no]
    )
    entry_id_map: Dict[str, DatasetEntryWithRowNo] = {}

    if "data" in response_data and "cells" in response_data["data"]:
        cells = response_data["data"]["cells"]
        entry_id_map = _map_attachment_entries_to_entry_ids(cells, attachment_queue)

    uploaded_attachments = []
    for entry_id, attachment_entry in entry_id_map.items():
        uploaded_attachment = api.upload_dataset_entry_attachments(dataset_id, entry_id, attachment_entry)
        if uploaded_attachment:
            uploaded_attachments.append((entry_id, uploaded_attachment))

    # Update the dataset with the uploaded attachments
    if uploaded_attachments:
        updates = []
        for entry_id, uploaded_attachment in uploaded_attachments:
            # Create the file variable payload according to the API schema
            file_variable_payload = {
                "text": uploaded_attachment.text,
                "files": [file.to_dict() for file in uploaded_attachment.files],
                "entryId": uploaded_attachment.entryId
            }
            
            # Find the column name from the original attachment entry
            attachment_entry = entry_id_map[entry_id]
            column_name = attachment_entry.column_name
            
            # Create update entry matching the API schema
            update_entry = {
                "entryId": entry_id,
                "columnName": column_name,
                "value": {
                    "type": "file",
                    "payload": file_variable_payload
                }
            }
            updates.append(update_entry)
        
        # Make the PATCH call with all updates
        patch_response = api.update_dataset_entries(dataset_id, updates)

        response_data["patch_response"] = patch_response

    return response_data


