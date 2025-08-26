import asyncio
from typing import Awaitable, List, Optional, Dict, Callable

from maxim.evaluators.base_evaluator import BaseEvaluator

from ..models.dataset import DataStructure, LocalData, Variable
from ..models.evaluator import (
    LocalEvaluationResult,
    LocalEvaluatorResultParameter,
    LocalEvaluatorReturn,
)


async def process_awaitable(awaitable: Awaitable):
    return await awaitable

def get_variables_from_row(
    row: LocalData, 
    data_structure: DataStructure,
) -> Dict[str, Variable]:
    variables = {}
    for column_name, column_type in data_structure.items():
        if column_type == "FILE_URL_VARIABLE":
            url_val = row.get(column_name)
            if url_val is None:
                continue
            url_str = str(url_val).strip()
            if not url_str:
                continue
            
            variables[column_name] = Variable(
                type="file",
                payload={"files": [{"url": url_str, "type": "url"}]},
            )
        elif column_type in ("VARIABLE", "NULLABLE_VARIABLE"):
            # Skip nullable variables with None values to avoid invalid payloads
            val = row.get(column_name)
            if column_type == "NULLABLE_VARIABLE" and val is None:
                continue
            variables[column_name] = Variable(
                type="text",
                payload={"text": "" if val is None else str(val)},
            )
    return variables


def get_input_expected_output_and_context_from_row(
    input_key: Optional[str],
    expectedOutputKey: Optional[str],
    contextToEvaluateKey: Optional[str],
    row: LocalData,
):
    input = None
    expected_output = None
    context_to_evaluate = None
    if input_key is not None and input_key in row:
        input = str(row[input_key]) if row[input_key] is not None else None
    if expectedOutputKey is not None and expectedOutputKey in row:
        expected_output = (
            str(row[expectedOutputKey]) if row[expectedOutputKey] is not None else None
        )
    if contextToEvaluateKey is not None and contextToEvaluateKey in row:
        context_to_evaluate = (
            str(row[contextToEvaluateKey])
            if row[contextToEvaluateKey] is not None
            else None
        )
    return input, expected_output, context_to_evaluate


async def run_local_evaluations(
    evaluators: List[BaseEvaluator],
    data_entry: LocalData,
    processed_data: LocalEvaluatorResultParameter,
) -> List[LocalEvaluationResult]:
    coroutines = [
        asyncio.to_thread(evaluator.guarded_evaluate, processed_data, data_entry)
        for evaluator in evaluators
    ]
    evaluator_results = await asyncio.gather(*coroutines)
    results: List[LocalEvaluationResult] = []
    for i, evaluator in enumerate(evaluators):
        if isinstance(evaluator, BaseEvaluator):
            try:
                combined_results = evaluator_results[i]
                for name, result in combined_results.items():
                    results.append(
                        LocalEvaluationResult(
                            name=name,
                            pass_fail_criteria=evaluator.pass_fail_criteria[name],
                            result=result,
                        )
                    )
            except Exception as err:
                results.extend(
                    [
                        LocalEvaluationResult(
                            name=name,
                            pass_fail_criteria=evaluator.pass_fail_criteria[name],
                            result=LocalEvaluatorReturn(
                                score="Err",
                                reasoning=f"Error while running combined evaluator with names {evaluator.names}: {str(err)}",
                            ),
                        )
                        for name in evaluator.names
                    ]
                )
    return results
