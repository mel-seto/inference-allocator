from enum import IntEnum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Priority(IntEnum):
    """Request priority levels. Lower value = higher priority."""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


class InferenceRequest(BaseModel):
    """Incoming inference request."""

    model_id: str = Field(..., description="Model identifier")
    prompt: str = Field(..., description="Input prompt")
    priority: Priority = Field(default=Priority.MEDIUM, description="Request priority")
    request_id: Optional[str] = Field(default=None, description="Client-provided request ID")

    @field_validator("priority", mode="before")
    @classmethod
    def parse_priority(cls, v):
        if isinstance(v, str):
            mapping = {"high": Priority.HIGH, "medium": Priority.MEDIUM, "low": Priority.LOW}
            if v.lower() in mapping:
                return mapping[v.lower()]
        return v


class InferenceResponse(BaseModel):
    """Response from inference execution."""

    request_id: str = Field(..., description="Request identifier")
    model_id: str = Field(..., description="Model used")
    output: str = Field(..., description="Generated output")
    gpu_id: int = Field(..., description="GPU that processed the request")
    inference_time_ms: float = Field(..., description="Time taken for inference in ms")
