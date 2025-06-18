from dataclasses import dataclass
from typing import Dict, List, Union

from maxim.evaluators.base_evaluator import BaseEvaluator

from ..models.evaluator import (
    Evaluator,
    EvaluatorType,
    PassFailCriteria,
)
from ..utils.utils import create_cuid_generator


@dataclass
class EvaluatorNameToIdAndPassFailCriteria:
    """
    This class represents an evaluator name to id and pass fail criteria.
    """

    id: str
    pass_fail_criteria: PassFailCriteria

    def __init__(self, id: str, pass_fail_criteria: PassFailCriteria):
        self.id = id
        self.pass_fail_criteria = pass_fail_criteria


def get_local_evaluator_name_to_id_and_pass_fail_criteria_map(
    evaluators: List[Union[BaseEvaluator, str]],
) -> Dict[str, EvaluatorNameToIdAndPassFailCriteria]:
    """
    This function returns a map of evaluator names to their corresponding ids and pass fail criteria.
    """
    all_eval_names: List[str] = []
    for evaluator in evaluators:
        if isinstance(evaluator, str):
            pass
        else:
            all_eval_names.extend(evaluator.names)

    all_pass_fail_criteria: Dict[str, PassFailCriteria] = {}
    for evaluator in evaluators:
        if isinstance(evaluator, BaseEvaluator):
            all_pass_fail_criteria.update(evaluator.pass_fail_criteria)

    name_to_id_and_pass_fail_criteria_map: Dict[
        str, EvaluatorNameToIdAndPassFailCriteria
    ] = {}
    for eval_name in all_eval_names:
        generate_cuid = create_cuid_generator()
        name_to_id_and_pass_fail_criteria_map[eval_name] = (
            EvaluatorNameToIdAndPassFailCriteria(
                id=generate_cuid(),
                pass_fail_criteria=all_pass_fail_criteria[eval_name],
            )
        )

    return name_to_id_and_pass_fail_criteria_map


def get_evaluator_config_from_evaluator_name_and_pass_fail_criteria(
    id: str, name: str, pass_fail_criteria: PassFailCriteria
) -> Evaluator:
    """
    This function returns an evaluator config from the evaluator name and pass fail criteria.
    """
    return Evaluator(
        builtin=False,
        id=id,
        name=name,
        reversed=False,
        type=EvaluatorType.LOCAL,
        config={
            "passFailCriteria": {
                "entryLevel": {
                    "value": (
                        pass_fail_criteria.on_each_entry.value
                        if not (
                            isinstance(pass_fail_criteria.on_each_entry.value, bool)
                        )
                        else (
                            "Yes"
                            if pass_fail_criteria.on_each_entry.value is True
                            else "No"
                        )
                    ),
                    "operator": pass_fail_criteria.on_each_entry.score_should_be,
                    "name": "score",
                },
                "runLevel": {
                    "value": pass_fail_criteria.for_testrun_overall.value,
                    "operator": pass_fail_criteria.for_testrun_overall.overall_should_be,
                    "name": (
                        "meanScore"
                        if pass_fail_criteria.for_testrun_overall.for_result
                        == "average"
                        else "queriesPassed"
                    ),
                },
            },
        },
    )
