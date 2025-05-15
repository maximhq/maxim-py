from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class Metadata():
    val: Dict[str, Union[str, int, bool]]
