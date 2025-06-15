"""
This module contains utility functions for parsing attachments from messages.
"""

import base64
import re
from typing import TYPE_CHECKING, Tuple

from ...scribe import scribe
from .attachment import (
    Attachment,
    FileDataAttachment,
    UrlAttachment,
)

if TYPE_CHECKING:
    from .generation import (
        GenerationRequestMessage,
    )


def parse_attachments_from_messages(
    messages: list["GenerationRequestMessage"],
) -> Tuple[list["GenerationRequestMessage"], list["Attachment"]]:
    """
    Parses the attachment from the result.
    Args:
        messages (List[GenerationRequestMessage]): The messages to parse attachments from.

    Returns:
        tuple: A tuple containing the modified messages and the list of attachments.
    """
    attachments = []
    for message in messages:
        content = message.get("content", [])
        if content is None or isinstance(content, str):
            continue
        # Iterate in reverse order to safely remove items while iterating
        for i in range(len(content) - 1, -1, -1):
            item = content[i]
            if isinstance(item, str):
                continue
            if isinstance(item, dict) and (
                item.get("type") == "image" or item.get("type") == "image_url"
            ):
                # Here we will check if its actual URL
                # or base64 encoded data uri
                image_url = item.get("image_url", "")
                url = None
                if image_url is not None and isinstance(image_url, dict):
                    url = image_url.get("url", "")
                elif image_url is not None and isinstance(image_url, str):
                    url = image_url
                if url is not None and url != "":
                    # Check if its base64 encoded data uri
                    if url.startswith("data:image"):
                        # Extract base64 data from data URI
                        match = re.match(
                            r"data:image/(?P<ext>\w+);base64,(?P<data>.+)",
                            url,
                        )
                        if match:
                            ext = match.group("ext")
                            data = match.group("data")
                            try:
                                file_data = base64.b64decode(data)
                            except Exception as e:
                                scribe().error(
                                    f"[MaximSDK] Error while parsing attachment: {str(e)}"
                                )
                                continue
                            attachment = FileDataAttachment(
                                data=file_data,
                                mime_type=f"image/{ext}",
                                tags={"attach-to": "input"},
                            )
                            attachments.append(attachment)
                    else:
                        attachment = UrlAttachment(url=url, tags={"attach-to": "input"})
                        attachments.append(attachment)

                    # Remove the image item from content
                    content.pop(i)
    return messages, attachments
