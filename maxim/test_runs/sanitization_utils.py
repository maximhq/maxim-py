import json
from typing import List, Union

from ..evaluators import BaseEvaluator
from ..models.dataset import Data, DataStructure


def sanitize_data(
    data_to_sanitize: Data, against_data_structure: DataStructure
) -> None:
    if data_to_sanitize:
        if against_data_structure and not isinstance(data_to_sanitize, str):
            if callable(data_to_sanitize):
                # do nothing as constructor already checks for the validity and existence of the file in case of CSVFile
                # and for function, we sanitize it while running the test run
                pass
            elif isinstance(data_to_sanitize, list):
                for data_entry in data_to_sanitize:
                    for key, value in data_entry.items():
                        if against_data_structure[key] == "INPUT":
                            if not isinstance(value, str):
                                raise ValueError(
                                    f'Input column "{key}" has a data entry which is not a string',
                                    {
                                        "cause": json.dumps(
                                            {"dataEntry": {key: value}}, indent=2
                                        )
                                    },
                                )
                        elif against_data_structure[key] == "EXPECTED_OUTPUT":
                            if not isinstance(value, str):
                                raise ValueError(
                                    f'Expected output column "{key}" has a data entry which is not a string',
                                    {
                                        "cause": json.dumps(
                                            {"dataEntry": {key: value}}, indent=2
                                        )
                                    },
                                )
                        elif against_data_structure[key] == "CONTEXT_TO_EVALUATE":
                            if not (isinstance(value, str)) and (
                                not (isinstance(value, list))
                                or not (all(isinstance(v, str) for v in value))
                            ):
                                raise ValueError(
                                    f'Context to evaluate column "{key}" has a data entry which is not a string or an array',
                                    {
                                        "cause": json.dumps(
                                            {"dataEntry": {key: value}}, indent=2
                                        )
                                    },
                                )
                        elif against_data_structure[key] == "VARIABLE":
                            if not (isinstance(value, str)) and (
                                not (isinstance(value, list))
                                or not (all(isinstance(v, str) for v in value))
                            ):
                                raise ValueError(
                                    f'Context to evaluate column "{key}" has a data entry which is not a string or an array',
                                    {
                                        "cause": json.dumps(
                                            {"dataEntry": {key: value}}, indent=2
                                        )
                                    },
                                )
                        elif against_data_structure[key] == "NULLABLE_VARIABLE":
                            if (
                                not (isinstance(value, str))
                                and value is not None
                                and (
                                    not (isinstance(value, list))
                                    or not (all(isinstance(v, str) for v in value))
                                )
                            ):
                                raise ValueError(
                                    f'Nullable variable column "{key}" has a data entry which is not null, a string or an array',
                                    {
                                        "cause": json.dumps(
                                            {"dataEntry": {key: value}}, indent=2
                                        )
                                    },
                                )
                        elif against_data_structure[key] == "FILE_URL_VARIABLE":
                            if not (isinstance(value, str)):
                                raise ValueError(
                                    f'File URL variable column "{key}" has a data entry which is not a string',
                                    {
                                        "cause": json.dumps(
                                            {"dataEntry": {key: value}}, indent=2
                                        )
                                    },
                                )
                        else:
                            raise ValueError(
                                f'Unknown column type "{against_data_structure[key]}" for column "{key}"',
                                {
                                    "cause": json.dumps(
                                        {
                                            "dataStructure": against_data_structure,
                                            "dataEntry": {key: value},
                                        },
                                        indent=2,
                                    )
                                },
                            )
            else:
                for key, value in data_to_sanitize.items():
                    if against_data_structure[key] == "INPUT":
                        if not isinstance(value, str):
                            raise ValueError(
                                f'Input column "{key}" has a data entry which is not a string',
                                {
                                    "cause": json.dumps(
                                        {"dataEntry": {key: value}}, indent=2
                                    )
                                },
                            )
                    elif against_data_structure[key] == "EXPECTED_OUTPUT":
                        if not isinstance(value, str):
                            raise ValueError(
                                f'Expected output column "{key}" has a data entry which is not a string',
                                {
                                    "cause": json.dumps(
                                        {"dataEntry": {key: value}}, indent=2
                                    )
                                },
                            )
                    elif against_data_structure[key] == "CONTEXT_TO_EVALUATE":
                        if not (isinstance(value, str)) and (
                            not (isinstance(value, list))
                            or not (all(isinstance(v, str) for v in value))
                        ):
                            raise ValueError(
                                f'Context to evaluate column "{key}" has a data entry which is not a string or an array',
                                {
                                    "cause": json.dumps(
                                        {"dataEntry": {key: value}}, indent=2
                                    )
                                },
                            )
                    elif against_data_structure[key] == "VARIABLE":
                        if not (isinstance(value, str)) and (
                            not (isinstance(value, list))
                            or not (all(isinstance(v, str) for v in value))
                        ):
                            raise ValueError(
                                f'Context to evaluate column "{key}" has a data entry which is not a string or an array',
                                {
                                    "cause": json.dumps(
                                        {"dataEntry": {key: value}}, indent=2
                                    )
                                },
                            )
                    elif against_data_structure[key] == "NULLABLE_VARIABLE":
                        if (
                            not (isinstance(value, str))
                            and value is not None
                            and (
                                not (isinstance(value, list))
                                or not (all(isinstance(v, str) for v in value))
                            )
                        ):
                            raise ValueError(
                                f'Nullable variable column "{key}" has a data entry which is not null, a string or an array',
                                {
                                    "cause": json.dumps(
                                        {"dataEntry": {key: value}}, indent=2
                                    )
                                },
                            )
                    elif against_data_structure[key] == "FILE_URL_VARIABLE":
                        if not isinstance(value, str):
                            raise ValueError(
                                f'File URL variable column "{key}" has a data entry which is not a string',
                                {
                                    "cause": json.dumps(
                                        {"dataEntry": {key: value}}, indent=2
                                    )
                                },
                            )
                    else:
                        raise ValueError(
                            f'Unknown column type "{against_data_structure[key]}" for column "{key}"',
                            {
                                "cause": json.dumps(
                                    {
                                        "dataStructure": against_data_structure,
                                        "dataEntry": {key: value},
                                    },
                                    indent=2,
                                )
                            },
                        )
        elif not isinstance(data_to_sanitize, str):
            raise ValueError(
                "Data structure is not provided and data argument is not a datasetId(string)",
                {"cause": json.dumps({"data": data_to_sanitize}, indent=2)},
            )


def sanitize_evaluators(
    evaluators: List[Union[BaseEvaluator, str]],
):
    names_encountered = set()
    all_evaluator_names = []
    for evaluator in evaluators:
        if isinstance(evaluator, BaseEvaluator):
            for name in evaluator.names:
                if name not in names_encountered:
                    for e in evaluators:
                        if isinstance(e, BaseEvaluator):
                            all_evaluator_names.extend(e.names)
                        else:
                            all_evaluator_names.append(e)
                    names_encountered.add(name)
                else:
                    raise ValueError(
                        f'Multiple evaluators with the same name "{name}" found',
                        {
                            "cause": json.dumps(
                                {"allEvaluatorNames": all_evaluator_names}, indent=2
                            )
                        },
                    )
        else:
            name = evaluator
            if name not in names_encountered:
                for e in evaluators:
                    if isinstance(e, BaseEvaluator):
                        all_evaluator_names.extend(e.names)
                    else:
                        all_evaluator_names.append(e)
                names_encountered.add(name)
            else:
                raise ValueError(
                    f'Multiple evaluators with the same name "{name}" found',
                    {
                        "cause": json.dumps(
                            {"allEvaluatorNames": all_evaluator_names}, indent=2
                        )
                    },
                )
