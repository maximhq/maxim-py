from typing import Any, Dict

from litellm.types.utils import ModelResponse


def parse_litellm_model_response(response: ModelResponse) -> Dict[str, Any]:
    return response.to_dict()
