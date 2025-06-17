from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class Metadata():
    """
    This class represents metadata.

    Attributes:
        val: The value of the metadata.
    """

    val: Dict[str, Union[str, int, bool]]
