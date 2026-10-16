import json
import re
from typing import List
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import BaseTool
from uuid import uuid4

def sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Sanitize a list of messages by removing any messages that are not instances of BaseMessage.
    """
    result : List[BaseMessage] = []
    for msg in messages:
        content = msg.content
        is_empty = content is None or content == "" or content == []

        if not is_empty:
            result.append(msg)
            continue

        if isinstance(msg, AIMessage):
            result.append(
                AIMessage(
                    content=" ",
                    tool_calls=getattr(msg, "tool_calls", None),
                    id=getattr(msg, "id", None),
                    response_metadata= getattr(msg, "response_metadata", {}),
                    usage_metadata= getattr(msg, "usage_metadata", None),
                )
            )
        elif isinstance(msg, HumanMessage):
            result.append(
                HumanMessage(
                    content= " ",
                    id=getattr(msg, "id", None),
                    response_metadata = getattr(msg, "response_metadata", {})
                )
            )
        elif isinstance(msg, ToolMessage):
            result.append(
                ToolMessage(
                    content=" ",
                    tool_call_id= getattr(msg, "tool_call_id",""),
                    id=getattr(msg, "id", None)
                )
            )
        elif isinstance(msg, SystemMessage):
            result.append(
                SystemMessage(
                    content=" ",
                    id= getattr(msg, "id", None)
                )
            )
        else:
            result.append(msg.__class__(content=" ", **msg.__dict__))
    return result

def normalize_text_tool_call(response: AIMessage, tools: List[BaseTool]) -> AIMessage:
    if response.tool_calls or not isinstance(response.content, str):
        return response

    tool_names = {tool.name for tool in tools}
    decoder = json.JSONDecoder()
    for start, character in enumerate(response.content):
        if character != "{":
            continue

        try:
            payload, _ = decoder.raw_decode(response.content[start:])
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if (
            not isinstance(name, str)
            or name not in tool_names
            or not isinstance(arguments, dict)
        ):
            continue

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": arguments,
                    "id": uuid4().hex,
                    "type": "tool_call",
                }
            ],
            id=response.id,
            response_metadata=response.response_metadata,
            usage_metadata=response.usage_metadata,
        )

    return response

def has_explicit_reschedule_target(messages: List[BaseMessage]) -> bool:
    reschedule_start = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
            and isinstance(messages[index].content, str)
            and "reschedul" in messages[index].content.lower()
        ),
        0,
    )

    for message in messages[reschedule_start:]:
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            continue
        for match in re.finditer(
            r"\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\b",
            message.content,
        ):
            prefix = message.content[max(0, match.start() - 12):match.start()]
            if not re.search(r"\bfrom\s*$", prefix, re.IGNORECASE):
                return True
    return False

def format_slot(dt) -> str:
    """
    Convert a pandas Timestamp (or datetime) into the canonical
    string format used by the whole system: M/D/YYYY H:MM
    """
    return (
        f"{dt.month}/{dt.day}/{dt.year} "
        f"{dt.hour}:{dt.minute:02d}"
    )