"""
ICDE OpenEnv Playground
Serves the FastAPI backend + an interactive Playground UI at /app
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import ICDEEnvironment
from env.models import ICDEAction
from graders import GRADERS

app = FastAPI(
    title="Incident Command Decision Environment (ICDE)",
    description="OpenEnv-compliant environment for AI incident command benchmarking.",
    version="1.0.0",
)

_environments: Dict[str, ICDEEnvironment] = {}


def _get_env(task_id: str) -> ICDEEnvironment:
    if task_id not in _environments:
        _environments[task_id] = ICDEEnvironment(task_id=task_id)
    return _environments[task_id]


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1_resource"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    task_id: Optional[str] = "task1_resource"
    action: ICDEAction


@app.get("/", response_class=HTMLResponse)
async def playground():
    return HTMLResponse(content="<h2>ICDE OpenEnv Running ✅</h2>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest):
    task_id = request.task_id or "task1_resource"
    seed = request.seed or 42
    env = ICDEEnvironment(task_id=task_id, seed=seed)
    _environments[task_id] = env
    obs = env.reset()
    return obs.dict()


@app.post("/step")
async def step(request: StepRequest):
    task_id = request.task_id or "task1_resource"
    env = _get_env(task_id)

    if env._sim is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    try:
        result = env.step(request.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
async def state(task_id: str = Query(default="task1_resource")):
    env = _get_env(task_id)

    if env._sim is None:
        raise HTTPException(status_code=400, detail="Not initialized")

    return env.state().dict()


@app.get("/grade")
async def grade(task_id: str = Query(default="task1_resource")):
    env = _get_env(task_id)

    if env._sim is None:
        raise HTTPException(status_code=400, detail="Not initialized")

    sim = env._sim
    grader_fn = GRADERS.get(task_id)

    if grader_fn is None:
        raise HTTPException(status_code=404, detail="No grader")

    resource_assignments = {
        rid: r.assigned_zone or "" for rid, r in sim.resources.items()
    }

    result = grader_fn(
        action_history=sim.action_history,
        flagged_reports=sim.flagged_reports,
        resource_assignments=resource_assignments,
        civilian_casualties=sim.civilian_casualties,
        cascade_triggered=sim.cascade_triggered,
        grader_subscores=sim.grader_subscores,
    )

    return result


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "task1_resource", "difficulty": "easy"},
            {"id": "task2_multiagency", "difficulty": "medium"},
            {"id": "task3_cascade", "difficulty": "hard"},
        ]
    }


# ✅ REQUIRED FOR OPENENV
def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
