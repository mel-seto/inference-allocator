"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from inference_allocator.api import routes
from inference_allocator.config import settings
from inference_allocator.services.orchestrator import Orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    orchestrator = Orchestrator(
        gpu_count=settings.gpu_count,
        queue_max_size=settings.queue_max_size,
        min_ms=settings.inference_min_ms,
        max_ms=settings.inference_max_ms
    )
    await orchestrator.start()
    app.state.orchestrator = orchestrator
    yield
    await orchestrator.stop()


app = FastAPI(
    title="Inference Allocator",
    description="GPU-based inference request allocation service",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(routes.router)
