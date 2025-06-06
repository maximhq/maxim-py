from typing import TypedDict


class SignedURLResponse(TypedDict):
    """
    Represents a signed URL response.

    Attributes:
        url (str): The signed URL for uploading a file.
    """

    url: str
