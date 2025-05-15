from dataclasses import dataclass
from typing import Any, TypedDict, Union

from typing_extensions import deprecated


class FeedbackDict(TypedDict, total=False):
    score: int
    comment: str


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class Feedback():
    score: int
    comment: str


def get_feedback_dict(feedback: Union[Feedback, FeedbackDict]) -> dict[str, Any]:
    return (
        dict(
            FeedbackDict(
                score=feedback.score,
                comment=feedback.comment,
            )
        )
        if isinstance(feedback, Feedback)
        else dict(feedback)
    )
