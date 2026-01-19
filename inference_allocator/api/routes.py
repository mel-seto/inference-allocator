"""FastAPI routes for inference API."""

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from inference_allocator.config import settings
from inference_allocator.models.request import InferenceRequest, InferenceResponse
from inference_allocator.services.priority_queue import QueueFullError

router = APIRouter(prefix="/api/v1")


@router.post("/inference", response_model=InferenceResponse)
async def submit_inference(request: InferenceRequest, http_request: Request) -> InferenceResponse:
    """Submit inference request and wait for response."""
    orchestrator = http_request.app.state.orchestrator

    try:
        response = await asyncio.wait_for(
            orchestrator.submit(request),
            timeout=settings.request_timeout_seconds
        )
        return response
    except QueueFullError:
        raise HTTPException(
            status_code=503,
            detail="Queue is full. Please try again later."
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timeout. The server took too long to respond."
        )


@router.get("/status")
async def get_status(request: Request) -> Dict[str, Any]:
    """Get current system status."""
    orchestrator = request.app.state.orchestrator
    return orchestrator.get_status()
