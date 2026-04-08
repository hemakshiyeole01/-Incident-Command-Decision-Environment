"""
Incident Command Decision Environment (ICDE) — Core Environment
Implements the OpenEnv interface: reset() / step() / state()
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from env.models import (
    ICDEAction,
    ICDEObservation,
    ICDEState,
    StepResult,
)
from env.simulator import IncidentSimulator


# Load scenarios from data directory
_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_scenario(task_id: str) -> Dict[str, Any]:
    path = _DATA_DIR / "scenarios.json"
    with open(path) as f:
        all_scenarios = json.load(f)
    for sc in all_scenarios:
        if sc["task_id"] == task_id:
            return sc
    raise ValueError(f"Unknown task_id: {task_id}. Available: {[s['task_id'] for s in all_scenarios]}")


class ICDEEnvironment:
    """
    OpenEnv-compliant environment for Incident Command Decision making.

    Supported tasks
    ---------------
    task1_resource      Easy   — Resource allocation under scarcity
    task2_multiagency   Medium — Multi-agency conflict resolution
    task3_cascade       Hard   — Cascading multi-site crisis management

    Usage
    -----
    env = ICDEEnvironment(task_id="task1_resource")
    obs = env.reset()
    while True:
        action = ICDEAction(command="dispatch", ...)
        result = env.step(action)
        if result.done:
            break
    """

    def __init__(self, task_id: str = "task1_resource", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self.scenario = _load_scenario(task_id)
        self._sim: Optional[IncidentSimulator] = None
        self._episode_id: str = str(uuid.uuid4())[:8]
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._last_observation: Optional[ICDEObservation] = None

    # ─── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> ICDEObservation:
        """
        Reset the environment to the initial state.
        Returns the initial observation.
        """
        self._sim = IncidentSimulator(self.scenario, seed=self.seed)
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._total_reward = 0.0
        obs = self._sim.reset()
        self._last_observation = obs
        return obs

    def step(self, action: ICDEAction) -> StepResult:
        """
        Apply an action and return (observation, reward, done, info).
        """
        if self._sim is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        obs, reward, done, info = self._sim.step(action)
        self._step_count += 1
        self._total_reward += reward
        self._last_observation = obs

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> ICDEState:
        """
        Return the current internal state (includes ground-truth values).
        """
        if self._sim is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        sim = self._sim
        return ICDEState(
            step=sim.step_num,
            task_id=self.task_id,
            episode_id=self._episode_id,
            done=sim.done,
            total_reward=self._total_reward,
            civilian_safe=sim.civilian_safe,
            civilian_at_risk=sim.civilian_at_risk,
            civilian_casualties=sim.civilian_casualties,
            optimal_casualties=sim.optimal_casualties,
            resource_double_assigned=any(
                sum(1 for r in sim.resources.values() if r.assigned_zone == z) > 1
                for z in sim.active_zones
            ),
            cascade_triggered=sim.cascade_triggered,
            last_action=sim.last_action,
            warnings_ignored=list(sim.warnings_ignored),
            action_history=list(sim.action_history),
            grader_subscores=dict(sim.grader_subscores),
        )

    # ─── Convenience helpers ─────────────────────────────────────────────────

    def available_tasks(self):
        return ["task1_resource", "task2_multiagency", "task3_cascade"]

    def close(self):
        """No persistent resources to close, but provided for API completeness."""
        pass
