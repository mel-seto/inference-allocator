# Inference Allocator

A distributed model inference system prototype demonstrating GPU resource management, priority-based request queuing, and concurrent processing patterns.

> **Note:** This is an architectural prototype with mock inference. It demonstrates backend systems design patterns, not actual GPU/ML integration.

## Features

- **Priority Queue** - Requests processed by priority (high → medium → low) with FIFO ordering within priority levels
- **GPU Resource Pool** - Simulated GPU allocation with blocking acquire/release semantics
- **Concurrent Processing** - Async request handling with parallel GPU utilization
- **Graceful Degradation** - Returns 503 when queue is full, 504 on timeout

## Architecture

```
POST /inference → Orchestrator → Priority Queue → Background Processor
       ↑                                              ↓
       └──── await Future ←── GPU Manager ←── Worker (mock inference)
```

### Components

| Component | Purpose |
|-----------|---------|
| `api/routes.py` | FastAPI endpoints (POST /inference, GET /status) |
| `services/orchestrator.py` | Coordinates request lifecycle |
| `services/priority_queue.py` | Async heap-based priority queue |
| `services/gpu_manager.py` | GPU slot allocation/release |
| `services/inference_worker.py` | Mock inference execution |

## Quick Start

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server
python -m uvicorn inference_allocator.main:app --port 8000

# Test request
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3-70b", "prompt": "Hello", "priority": "high"}'
```

## API

### POST /api/v1/inference

Submit an inference request.

**Request:**
```json
{
  "model_id": "llama-3-70b",
  "prompt": "user input text",
  "priority": "high|medium|low"
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "model_id": "llama-3-70b",
  "output": "generated text",
  "gpu_id": 0,
  "inference_time_ms": 342.5
}
```

### GET /api/v1/status

Get system status.

**Response:**
```json
{
  "queue_size": 5,
  "gpus_available": 3,
  "gpus_busy": 1
}
```

## Configuration

Environment variables (prefix: `INFERENCE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_GPU_COUNT` | 4 | Number of simulated GPUs |
| `INFERENCE_QUEUE_MAX_SIZE` | 100 | Max pending requests |
| `INFERENCE_REQUEST_TIMEOUT_SECONDS` | 30 | Request timeout |

## Testing

```bash
# Run all tests (37 total)
pytest inference_allocator/tests/ -v

# Unit tests only
pytest inference_allocator/tests/test_api.py -v

# Integration tests only
pytest inference_allocator/tests/test_api_integration.py -v
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Queue | heapq + async wrapper | Full control over priority + FIFO ordering |
| GPU allocation | Condition-based blocking | Fair, no busy-polling |
| Request handling | Future-based | Client waits synchronously, backend processes in parallel |
| Error handling | HTTP status codes | 503 (queue full), 504 (timeout), 422 (validation) |

## Project Structure

```
inference_allocator/
├── main.py              # FastAPI app entry point
├── config.py            # Settings
├── models/
│   ├── request.py       # Pydantic models
│   └── gpu.py           # GPU state enum
├── services/
│   ├── priority_queue.py
│   ├── gpu_manager.py
│   ├── inference_worker.py
│   └── orchestrator.py
├── api/
│   └── routes.py
└── tests/
    ├── test_api.py
    ├── test_api_integration.py
    ├── test_priority_queue.py
    ├── test_gpu_manager.py
    ├── test_inference_worker.py
    └── test_orchestrator.py
```

## License

MIT
