"""Tests for API endpoints."""

import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

from inference_allocator.main import app, lifespan
from inference_allocator.models.request import InferenceResponse, Priority
from inference_allocator.services.priority_queue import QueueFullError


class TestSubmitInference:
    """Test POST /api/v1/inference endpoint."""

    @pytest.mark.asyncio
    async def test_submit_inference_returns_response(self):
        """Submit request should return InferenceResponse fields."""
        mock_response = InferenceResponse(
            request_id="req-123",
            model_id="llama-3-70b",
            output="Generated text",
            gpu_id=0,
            inference_time_ms=150.5
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.submit.return_value = mock_response

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            app.state.orchestrator = mock_orchestrator

            response = await client.post(
                "/api/v1/inference",
                json={
                    "model_id": "llama-3-70b",
                    "prompt": "Hello, world!",
                    "priority": 1
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-123"
        assert data["model_id"] == "llama-3-70b"
        assert data["output"] == "Generated text"
        assert data["gpu_id"] == 0
        assert data["inference_time_ms"] == 150.5

    @pytest.mark.asyncio
    async def test_submit_inference_validates_input(self):
        """Missing model_id should return 422."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            app.state.orchestrator = MagicMock()

            response = await client.post(
                "/api/v1/inference",
                json={"prompt": "Hello"}  # Missing model_id
            )

        assert response.status_code == 422


class TestStatus:
    """Test GET /api/v1/status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_system_state(self):
        """Status endpoint should return queue and GPU info."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.get_status.return_value = {
            "queue_size": 5,
            "gpus_available": 3,
            "gpus_busy": 1
        }

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            app.state.orchestrator = mock_orchestrator

            response = await client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["queue_size"] == 5
        assert data["gpus_available"] == 3
        assert data["gpus_busy"] == 1


class TestErrorHandling:
    """Test error handling for API endpoints."""

    @pytest.mark.asyncio
    async def test_queue_full_returns_503(self):
        """When queue is at capacity, should return 503."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.submit.side_effect = QueueFullError("Queue full")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            app.state.orchestrator = mock_orchestrator

            response = await client.post(
                "/api/v1/inference",
                json={
                    "model_id": "test-model",
                    "prompt": "Hello",
                    "priority": 2
                }
            )

        assert response.status_code == 503
        assert "queue" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_504(self):
        """When request times out, should return 504."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.submit.side_effect = asyncio.TimeoutError()

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            app.state.orchestrator = mock_orchestrator

            response = await client.post(
                "/api/v1/inference",
                json={
                    "model_id": "test-model",
                    "prompt": "Hello",
                    "priority": 2
                }
            )

        assert response.status_code == 504
        assert "timeout" in response.json()["detail"].lower()
