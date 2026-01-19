# Progress

## Completed
- [x] Step 1: Foundation (project structure, config.py, Pydantic models)
- [x] Step 2: Priority Queue (TDD - 8 tests passing)
- [x] Step 3: GPU Manager (TDD - 12 tests passing)
- [x] Step 4: Inference Worker (TDD - 4 tests passing)
- [x] Step 5: Orchestrator (TDD - 4 tests passing)
- [x] Step 6: API Layer (routes.py, main.py - 5 unit tests)
- [x] Step 7: Integration tests (4 tests with real orchestrator)

## Remaining
None - implementation complete!

## Approach
Write failing tests first, verify they fail, then implement.

## Test Status
```
37 tests passing
pytest inference_allocator/tests/ -v
```

---

# Overview
Design and build a distributed model inference system that handles multiple AI models, manages GPU resources efficiently, implements request queuing with priority levels, and auto-scales based on load. Include monitoring, rate limiting, and graceful degradation when resources are constrained.

# Functional requirements
A working prototype that demonstrates the core architecture - request routing, basic GPU allocation logic, and a simple queue implementation. You can mock the actual model inference.
Focus on code quality and system design over full features. I want to see clean, production-ready code with proper error handling.

sample HTTP POST:
```json
{
  "model_id": "llama-3-70b",
  "prompt": "user input text",
  "priority": "high|medium|low"
}
```

expected output:
```json
{
  "generated_text": "model output here",
  "inference_time_ms": 342,
  "gpu_id": "gpu-0",
  "model_id": "llama-3-70b"
}
```

# Architecture

```
HTTP Server -> Queue -> GPU Manager assigns free GPU -> Worker executes -> Response
```

1. HTTP Server - receives requests, returns responses
2. Priority Queue - holds pending requests sorted by priority
3. GPU Manager - tracks which GPUs are free/busy, assigns work
4. Worker Pool - processes requests on assigned GPUs

# Technology Stack
- **HTTP Server**: FastAPI (async)
- **Queue**: `heapq` with async wrapper
- **Concurrency**: `asyncio`
- **GPUs**: Simulated slots with configurable count

# Project Structure
```
inference_allocator/
├── main.py                    # FastAPI entry point
├── config.py                  # Settings (GPU count, timeouts)
├── models/
│   ├── request.py             # InferenceRequest, InferenceResponse
│   └── gpu.py                 # GPU state model
├── services/
│   ├── priority_queue.py      # Async priority queue
│   ├── gpu_manager.py         # GPU allocation/release
│   ├── inference_worker.py    # Mock inference execution
│   └── orchestrator.py        # Coordinates full request lifecycle
├── api/
│   └── routes.py              # POST /inference, GET /status
└── tests/
    └── test_*.py              # Unit and integration tests
```

# Implementation Steps

## Step 1: Foundation
- Create project structure and `__init__.py` files
- Implement `config.py` with settings (gpu_count, timeouts, queue_size)
- Implement Pydantic models in `models/request.py` and `models/gpu.py`

## Step 2: Priority Queue
- Implement `services/priority_queue.py` using `heapq`
- Priority ordering: HIGH=1, MEDIUM=2, LOW=3 (lower wins)
- FIFO within same priority using counter
- Async put/get with condition-based waiting

## Step 3: GPU Manager
- Implement `services/gpu_manager.py`
- Track N simulated GPU slots (AVAILABLE/BUSY)
- `acquire_gpu()` - blocks until GPU free, returns GPU
- `release_gpu()` - marks available, notifies waiters

## Step 4: Inference Worker
- Implement `services/inference_worker.py`
- Acquires GPU, runs mock delay (100-500ms), releases GPU
- Returns response with gpu_id, inference_time_ms

## Step 5: Orchestrator
- Implement `services/orchestrator.py`
- Creates `asyncio.Future` per request
- Background loop: dequeue → spawn task → execute → resolve future
- Enables parallel GPU utilization

## Step 6: API Layer
- Implement `api/routes.py` with FastAPI router
- `POST /api/v1/inference` - submit and wait for response
- `GET /api/v1/status` - queue size, GPU states
- Implement `main.py` with lifespan management

## Step 7: Testing & Polish
- Unit tests for queue, GPU manager, worker
- Integration tests with httpx
- Add requirements.txt

# Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Queue | heapq + async wrapper | Full control over priority + FIFO ordering |
| GPU allocation | First-available + Condition | Simple, fair, no busy-polling |
| Request handling | Future-based sync wait | Client waits, internally parallel |
| Error handling | Custom exceptions → HTTP codes | 503 for overload, 504 for timeout |

# Request/Response Flow
```
POST /inference → Orchestrator → Queue → Background processor
      ↑                                         ↓
      └──── await Future ←── GPU Manager ←── Worker (mock inference)
```

# Verification
1. Start server: `uvicorn main:app --reload --port 8000`
2. Test request:
   ```bash
   curl -X POST http://localhost:8000/api/v1/inference \
     -H "Content-Type: application/json" \
     -d '{"model_id": "llama-3-70b", "prompt": "Hello", "priority": "high"}'
   ```
3. Check status: `curl http://localhost:8000/api/v1/status`
4. Run tests: `pytest tests/`

# Files to Create
1. `inference_allocator/config.py`
2. `inference_allocator/models/request.py`
3. `inference_allocator/models/gpu.py`
4. `inference_allocator/services/priority_queue.py`
5. `inference_allocator/services/gpu_manager.py`
6. `inference_allocator/services/inference_worker.py`
7. `inference_allocator/services/orchestrator.py`
8. `inference_allocator/api/routes.py`
9. `inference_allocator/main.py`
10. `inference_allocator/requirements.txt`
