"""miniserve.protocol — Request/response data models (Pydantic v2).

Defines the OpenAI-compatible API schema for completion requests,
responses, streaming chunks, and usage statistics.
Backward-compatible: all new fields have defaults.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"
    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema")

    model_config = {"populate_by_name": True}


class CompletionRequest(BaseModel):
    model: str = "stub"
    prompt: str = ""
    messages: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    seed: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    # internal — set by dedup middleware
    _dedup_key: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class StreamDelta(BaseModel):
    text: str
    index: int = 0
    finish_reason: Optional[str] = None


class StreamChunk(BaseModel):
    id: str
    object: str = "text_completion.chunk"
    model: str
    choices: List[StreamDelta]
