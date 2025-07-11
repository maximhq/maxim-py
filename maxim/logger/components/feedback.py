from dataclasses import dataclass
from typing import Any, TypedDict, Union

from typing_extensions import deprecated


class FeedbackDict(TypedDict, total=False):
    """Feedback dict.

    This class provides functionality to manage feedback dictionaries.
    """

    score: int
    comment: str


@deprecated(
    "This class will be removed in a future version. Use FeedbackDict instead."
)
@dataclass
class Feedback():
    """Feedback.

    This class represents a feedback event.
    """

    score: int
    comment: str


def get_feedback_dict(feedback: Union[Feedback, FeedbackDict]) -> dict[str, Any]:
    """Get the feedback dict.

    Args:
        feedback (Union[Feedback, FeedbackDict]): The feedback to get the dict from.

    Returns:
        dict[str, Any]: The feedback dict.
    """
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
