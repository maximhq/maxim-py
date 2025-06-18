from typing import Union


def parse_tags(data: dict) -> Union[None, dict]:
    """
    Parse tags from a dictionary.

    Args:
        data: The dictionary to parse.

    Returns:
        The parsed tags.
    """

    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("tags must be a dict[str,str]")
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = parse_tags(value)
        elif not isinstance(value, str):
            try:
                data[key] = str(value)
            except Exception as e:
                data[key] = repr(value)
    return data
