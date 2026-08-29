"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    role: str
    # content is a plain string, or a list of parts (OpenAI vision-style): text
    # parts and data-URI image parts are read; other parts are ignored.
    content: Union[str, List[dict], None] = None
    # OpenAI function-calling fields: an assistant message may carry tool_calls,
    # and a role="tool" message carries the result for tool_call_id.
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # Tools to enable for this request, independent of the model. OpenAI clients
    # pass these via extra_body: `thinking` (DeepThink), `search` (web).
    thinking: bool = False
    search: bool = False
    # OpenAI function tools. DeepSeek's web API has no native tool-call channel,
    # so these are emulated: the schemas are injected into the prompt and the
    # model's <function_call> output is parsed back into OpenAI tool_calls.
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None  # accepted, not enforced
    parallel_tool_calls: Optional[bool] = None      # always effectively False
    # Accepted for compatibility but not all are forwarded to DeepSeek.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
