import logging
from abc import ABC, abstractmethod
from typing import Dict, final

from ..models.dataset import LocalData
from ..models.evaluator import (
    LocalEvaluatorResultParameter,
    LocalEvaluatorReturn,
    PassFailCriteria,
)
from .utils import sanitize_pass_fail_criteria


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    _evaluator_names: list[str]
    _pass_fail_criteria: dict[str, PassFailCriteria]

    def __init__(self, pass_fail_criteria: dict[str, PassFailCriteria]):
        self._evaluator_names = []
        """Initialize the evaluator."""
        for name, pfc in pass_fail_criteria.items():
            sanitize_pass_fail_criteria(name, pfc)
            self._evaluator_names.append(name)
        self._pass_fail_criteria = pass_fail_criteria

    @property
    def names(self) -> list[str]:
        """Get the names of the evaluators."""
        return self._evaluator_names

    @property
    def pass_fail_criteria(self):
        """Get the pass fail criteria for the evaluators."""
        return self._pass_fail_criteria

    @abstractmethod
    def evaluate(
        self, result: LocalEvaluatorResultParameter, data: LocalData
    ) -> Dict[str, LocalEvaluatorReturn]:
        """Evaluate the result."""
        pass

    @final
    def guarded_evaluate(
        self, result: LocalEvaluatorResultParameter, data: LocalData
    ) -> Dict[str, LocalEvaluatorReturn]:
        """Guarded evaluate the result."""
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
