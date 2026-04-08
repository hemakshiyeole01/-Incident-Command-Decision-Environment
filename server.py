"""
ICDE FastAPI Server
Exposes /reset, /step, /state endpoints compatible with OpenEnv spec.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import ICDEEnvironment
from env.models import ICDEAction, ICDEObservation, ICDEState, StepResult
from graders import GRADERS

app = FastAPI(
    title="Incident Command Decision Environment (ICDE)",
    description=(
        "OpenEnv-compliant environment where an AI agent plays Incident Commander "
        "managing real emergencies using the ICS framework."
    ),
    version="1.0.0",
)

# ── Session management ────────────────────────────────────────────────────────
# Simple in-memory session store (one env per task for demo/evaluation)
_environments: Dict[str, ICDEEnvironment] = {}


def _get_env(task_id: str) -> ICDEEnvironment:
    if task_id not in _environments:
        _environments[task_id] = ICDEEnvironment(task_id=task_id)
    return _environments[task_id]


# ── Models ────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1_resource"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    task_id: Optional[str] = "task1_resource"
    action: ICDEAction


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Incident Command Decision Environment (ICDE)",
        "version": "1.0.0",
        "tasks": ["task1_resource", "task2_multiagency", "task3_cascade"],
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── OpenEnv Core API ──────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and return initial observation.
    Compatible with OpenEnv reset() spec.
    """
    task_id = request.task_id or "task1_resource"
    seed = request.seed or 42

    env = ICDEEnvironment(task_id=task_id, seed=seed)
    _environments[task_id] = env

    obs = env.reset()
    return obs.dict()


@app.post("/step")
async def step(request: StepRequest):
    """
    Apply an action and return (observation, reward, done, info).
    Compatible with OpenEnv step() spec.
    """
    task_id = request.task_id or "task1_resource"
    env = _get_env(task_id)

    if env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        result = env.step(request.action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
async def state(task_id: str = Query(default="task1_resource")):
    """
    Return current internal state.
    Compatible with OpenEnv state() spec.
    """
    env = _get_env(task_id)
    if env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    s = env.state()
    return s.dict()


# ── Grading ───────────────────────────────────────────────────────────────────

@app.get("/grade")
async def grade(task_id: str = Query(default="task1_resource")):
    """
    Run the deterministic grader for the current episode.
    Returns score 0.0 – 1.0 with breakdown.
    """
    env = _get_env(task_id)
    if env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    sim = env._sim
    state_obj = env.state()

    grader_fn = GRADERS.get(task_id)
    if grader_fn is None:
        raise HTTPException(status_code=404, detail=f"No grader for task: {task_id}")

    resource_assignments = {
        rid: r.assigned_zone or ""
        for rid, r in sim.resources.items()
    }

    common_kwargs = dict(
        action_history=sim.action_history,
        flagged_reports=sim.flagged_reports,
        resource_assignments=resource_assignments,
        civilian_casualties=sim.civilian_casualties,
        cascade_triggered=sim.cascade_triggered,
        grader_subscores=sim.grader_subscores,
    )

    if task_id == "task1_resource":
        result = grader_fn(**common_kwargs)
    elif task_id == "task2_multiagency":
        command_established = any(
            "establish_command" in a for a in sim.action_history
        )
        steps_to_cmd = next(
            (i + 1 for i, a in enumerate(sim.action_history) if "establish_command" in a),
            99
        )
        result = grader_fn(
            **common_kwargs,
            command_established=command_established,
            steps_to_first_command=steps_to_cmd,
        )
    elif task_id == "task3_cascade":
        command_established = any(
            "establish_command" in a for a in sim.action_history
        )
        triggered_ids = [
            cid for cid, cond in sim.cascade_conditions.items()
            if cond.get("triggered", False)
        ]
        result = grader_fn(
            **common_kwargs,
            command_established=command_established,
            total_steps=sim.step_num,
            cascades_triggered_ids=triggered_ids,
        )
    else:
        result = {"score": 0.0, "error": "Unknown task"}

    return result


# ── Available tasks ───────────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1_resource",
                "name": "Resource Allocation Under Scarcity",
                "difficulty": "easy",
                "max_steps": 15,
                "description": "Warehouse fire with 3 active zones, limited resources, and contradictory field reports.",
            },
            {
                "id": "task2_multiagency",
                "name": "Multi-Agency Conflict Resolution",
                "difficulty": "medium",
                "max_steps": 20,
                "description": "Hospital mass casualty event with 4 agencies giving contradictory reports.",
            },
            {
                "id": "task3_cascade",
                "name": "Cascading Multi-Site Crisis",
                "difficulty": "hard",
                "max_steps": 30,
                "description": "Simultaneous earthquake, gas rupture, and hospital power failure with cascade logic.",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
