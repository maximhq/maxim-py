def validate_optional_type(value, expected_type, field_name):
    if value is not None and not isinstance(value, expected_type):
        raise ValueError(
            f"{field_name} must be of type {expected_type.__name__}")


def validate_type(value, expected_type, field_name):
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{field_name} must be of type {expected_type.__name__}")

def validate_content(value, expected_values):
    if value not in expected_values:
        raise ValueError(
            f"Content must be one of the following values: {', '.join(expected_values)}")

def validate_type_to_be_one_of(value, expected_types, field_name):
    if not any(isinstance(value, t) for t in expected_types):
        raise ValueError(
            f"{field_name} must be one of the following types: {', '.join([t.__name__ for t in expected_types])}")
