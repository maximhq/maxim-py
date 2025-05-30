from typing import Any, Dict, Optional, TypedDict, Union


class Session(TypedDict):
    id: str
    object: str
    expires_at: int
    input_audio_noise_reduction: Optional[bool]
    turn_detection: dict
    input_audio_format: str
    input_audio_transcription: Optional[dict]
    client_secret: Optional[str]
    include: Optional[list]
    model: str
    modalities: list[str]
    instructions: str
    voice: str
    output_audio_format: str
    tool_choice: str
    temperature: float
    max_response_output_tokens: Union[str, int]
    speed: float
    tools: list


class SessionCreatedEvent(TypedDict):
    type: str
    event_id: str
    session: Session


def get_model_params(session: Session) -> Dict[str, Any]:    
    params = dict(session)
    # Removing some keys as they are not part of the model parameters
    params.pop("model",None)
    params.pop("instructions", None)
    return params
