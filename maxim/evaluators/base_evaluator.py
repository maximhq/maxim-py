import logging
from abc import ABC, abstractmethod
from typing import Dict, final

from ..models.dataset import ManualData
from ..models.evaluator import (
    LocalEvaluatorResultParameter,
    LocalEvaluatorReturn,
    PassFailCriteria,
)
from .utils import sanitize_pass_fail_criteria


class BaseEvaluator(ABC):
    _evaluator_names: list[str]
    _pass_fail_criteria: dict[str, PassFailCriteria]

    def __init__(self, pass_fail_criteria: dict[str, PassFailCriteria]):
        self._evaluator_names = []
        for name, pfc in pass_fail_criteria.items():
            sanitize_pass_fail_criteria(name, pfc)
            self._evaluator_names.append(name)
        self._pass_fail_criteria = pass_fail_criteria

    @property
    def names(self) -> list[str]:
        return self._evaluator_names

    @property
    def pass_fail_criteria(self):
        return self._pass_fail_criteria

    @abstractmethod
    def evaluate(
        self, result: LocalEvaluatorResultParameter, data: ManualData
    ) -> Dict[str, LocalEvaluatorReturn]:
        pass

    @final
    def guarded_evaluate(
        self, result: LocalEvaluatorResultParameter, data: ManualData
    ) -> Dict[str, LocalEvaluatorReturn]:
        response = self.evaluate(result, data)
        invalid_evaluator_names: list[str] = []
        for key in response.keys():
            if key not in self._evaluator_names:
                invalid_evaluator_names.append(key)
        if len(invalid_evaluator_names) > 0:
            logging.warning(
                f"Received results for unknown evaluator names: [{invalid_evaluator_names}]. Make sure you initialize pass fail criteria for these evaluator names"
            )
        return response
