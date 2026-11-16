"""FastAPI REST application for AgentOps platform."""

from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agentops.config import get_settings
from agentops.orchestrator import ADKOrchestrator
from agentops.schemas import WorkflowExecutionState
from agentops.storage import ExecutionStore

logger = logging.getLogger("agentops.api")

store = ExecutionStore()
orchestrator = ADKOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing AgentOps application and SQLite persistence...")
    await store.init_db()
    yield
    logger.info("Shutting down AgentOps application.")


app = FastAPI(
    title="AgentOps Intelligence Platform API",
    description="Multi-Agent Research, Evaluation and Observability Platform powered by Google ADK and Gemini",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Middleware
settings = get_settings()
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrchestrateRequest(BaseModel):
    """Task submission payload."""
    task: str = Field(description="The research or codebase investigation task.")
    repo_url: Optional[str] = Field(default=None, description="Optional repository URL or local path.")
    max_iterations: Optional[int] = Field(default=3, ge=1, le=10, description="Max bounded iterations.")


@app.get("/api/health")
async def health_check():
    """Health status endpoint."""
    return {
        "status": "healthy",
        "gemini_live": orchestrator.provider.is_live,
        "default_model": settings.gemini_model,
        "otel_service": settings.otel_service_name,
        "storage": "sqlite_active",
    }


@app.post("/api/orchestrate", response_model=WorkflowExecutionState)
async def orchestrate_workflow(req: OrchestrateRequest):
    """Execute the multi-agent research, write, judge, and bounded refinement loop."""
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="Task string cannot be empty.")

    try:
        state = await orchestrator.run(
            task=req.task,
            repo_url=req.repo_url,
            max_iterations=req.max_iterations,
        )
        # Persist execution run
        await store.save_execution(state)
        return state
    except Exception as e:
        logger.error(f"Error during orchestration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_execution_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Retrieve paginated list of past workflow executions."""
    return await store.list_executions(limit=limit, offset=offset)


@app.get("/api/history/{request_id}", response_model=WorkflowExecutionState)
async def get_execution_details(request_id: str):
    """Retrieve complete execution trajectory and state by request ID."""
    state = await store.get_execution(request_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Execution '{request_id}' not found.")
    return state


@app.get("/api/metrics")
async def get_system_metrics():
    """Get aggregated token consumption, latency, costs, and request counts."""
    return await store.get_summary_metrics()


# Mount frontend build directory if present
dist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(dist_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_dir, "index.html"))

