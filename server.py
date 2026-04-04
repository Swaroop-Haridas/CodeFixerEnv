"""
server.py — FastAPI REST server for CodeFixerEnv.

Endpoints
---------
GET  /health        — liveness check
POST /reset         — start a new episode
POST /step          — submit an action
GET  /state         — inspect internal env state
GET  /tasks         — list all available tasks

All responses use Pydantic models for type-safe serialisation.
Hugging Face Spaces expects the server on port 7860.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.environment import CodeFixerEnv
from env.models import Action, Observation, Reward, StepResult
from tasks.tasks import TASKS

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CodeFixerEnv",
    description=(
        "OpenEnv environment for code-debugging RL tasks. "
        "An agent receives buggy Python code and must submit corrected solutions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (one session at a time)
env = CodeFixerEnv()


# ── Request/response schemas ──────────────────────────────────────────────────
class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"


class StepRequest(BaseModel):
    action_type: str
    action_content: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class TaskInfo(BaseModel):
    id: str
    difficulty: str
    context: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness check — returns 200 if the server is up."""
    return {"status": "ok", "environment": "CodeFixerEnv", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    """
    Start a new episode.

    Body: { "difficulty": "easy" | "medium" | "hard" }
    Returns the initial Observation.
    """
    try:
        obs = env.reset(difficulty=request.difficulty or "easy")
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit one action and advance the episode.

    Body: { "action_type": "fix"|"explain"|"give_up", "action_content": "..." }
    Returns observation, structured reward, done flag, and diagnostics.
    """
    try:
        action = Action(type=request.action_type, content=request.action_content)
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Return the current internal state of the environment (for debugging)."""
    return env.state()


@app.get("/tasks", response_model=list[TaskInfo])
def list_tasks():
    """List all available tasks with their IDs and descriptions."""
    return [
        TaskInfo(id=t["id"], difficulty=t["difficulty"], context=t["context"])
        for _, (t, _) in TASKS.items()
    ]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
