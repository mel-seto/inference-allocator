"""Integration tests for API endpoints using real orchestrator."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from inference_allocator.api import routes
from inference_allocator.services.orchestrator import Orchestrator


def create_app(
    gpu_count: int = 4,
    queue_max_size: int = 100,
    min_ms: int = 50,
    max_ms: int = 100
) -> FastAPI:
    """Create a FastAPI app with custom orchestrator settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        orchestrator = Orchestrator(
            gpu_count=gpu_count,
            queue_max_size=queue_max_size,
            min_ms=min_ms,
            max_ms=max_ms
        )
        await orchestrator.start()
        app.state.orchestrator = orchestrator
        yield
        await orchestrator.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(routes.router)
    return app


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Default client with standard settings."""
    app = create_app()
    async with LifespanManager(app) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app),
            base_url="http://test"
        ) as c:
            yield c


@pytest_asyncio.fixture
async def slow_client() -> AsyncGenerator[AsyncClient, None]:
    """Client with slow inference and small queue for testing 503."""
    app = create_app(gpu_count=1, queue_max_size=2, min_ms=500, max_ms=600)
    async with LifespanManager(app) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app),
            base_url="http://test"
        ) as c:
            yield c


class TestEndToEnd:
    """End-to-end tests with real orchestrator."""

    @pytest.mark.asyncio
    async def test_submit_inference_real_orchestrator(self, client: AsyncClient):
        """Submit request through real orchestrator."""
        response = await client.post(
            "/api/v1/inference",
            json={"model_id": "llama-3-70b", "prompt": "Hello", "priority": 1}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "llama-3-70b"
        assert "request_id" in data
        assert "output" in data
        assert "gpu_id" in data
        assert "inference_time_ms" in data

    @pytest.mark.asyncio
    async def test_status_reflects_real_state(self, client: AsyncClient):
        """Status endpoint returns real orchestrator state."""
        response = await client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["gpus_available"] == 4
        assert data["gpus_busy"] == 0
        assert data["queue_size"] == 0


class TestQueueFullScenario:
    """Test 503 response when queue is full."""

    @pytest.mark.asyncio
    async def test_queue_full_returns_503(self, slow_client: AsyncClient):
        """When queue is at capacity, should return 503."""
        # Send many requests simultaneously to overwhelm the queue
        # With 1 GPU, queue_max_size=2, and slow inference (500-600ms),
        # sending 10 requests at once should trigger 503
        tasks = [
            asyncio.create_task(
                slow_client.post(
                    "/api/v1/inference",
                    json={"model_id": "test", "prompt": f"Request {i}", "priority": 2}
                )
            )
            for i in range(10)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        status_codes = [r.status_code for r in responses if not isinstance(r, Exception)]

        assert 503 in status_codes, f"Expected 503 in {status_codes}"


class TestParallelGPUUsage:
    """Test concurrent requests use multiple GPUs."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_use_multiple_gpus(self, client: AsyncClient):
        """Concurrent requests should be processed on different GPUs."""
        tasks = [
            client.post(
                "/api/v1/inference",
                json={"model_id": "test", "prompt": f"Concurrent {i}", "priority": 1}
            )
            for i in range(4)
        ]

        responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses)

        gpu_ids = [r.json()["gpu_id"] for r in responses]
        unique_gpus = set(gpu_ids)

        assert len(unique_gpus) > 1, f"Expected multiple GPUs, got {gpu_ids}"
