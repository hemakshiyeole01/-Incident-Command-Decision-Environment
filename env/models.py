"""
Typed models for the Incident Command Decision Environment (ICDE).
Uses Pydantic for strict validation across the OpenEnv API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Enumerations ────────────────────────────────────────────────────────────

class ResourceType(str, Enum):
    ENGINE = "engine"
    HAZMAT = "hazmat"
    MEDICAL = "medical"
    POLICE = "police"
    RESCUE = "rescue"
    POWER = "power"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CommandAction(str, Enum):
    DISPATCH = "dispatch"
    RECALL = "recall"
    ESTABLISH_COMMAND = "establish_command"
    FLAG_CONFLICT = "flag_conflict"
    ESCALATE = "escalate"
    STAND_DOWN = "stand_down"
    REQUEST_MUTUAL_AID = "request_mutual_aid"
    ISSUE_DIRECTIVE = "issue_directive"


# ─── Action ──────────────────────────────────────────────────────────────────

class ICDEAction(BaseModel):
    """
    Action taken by the Incident Commander agent.

    Fields
    ------
    command : CommandAction
        High-level command type (DISPATCH, RECALL, FLAG_CONFLICT, …)
    resource_type : Optional[ResourceType]
        Which resource class to act on (required for DISPATCH / RECALL)
    target_zone : Optional[str]
        Zone identifier where action applies (e.g. "zone_a", "site_hospital")
    priority : Optional[Priority]
        Priority level assigned to this action
    directive : Optional[str]
        Free-text order / directive issued to field units (max 500 chars)
    flags : Optional[List[str]]
        List of report IDs / agency names flagged as unreliable
    """

    command: CommandAction
    resource_type: Optional[ResourceType] = None
    target_zone: Optional[str] = None
    priority: Optional[Priority] = Priority.HIGH
    directive: Optional[str] = Field(None, max_length=500)
    flags: Optional[List[str]] = Field(default_factory=list)

    class Config:
        use_enum_values = True


# ─── Observation ─────────────────────────────────────────────────────────────

class FieldReport(BaseModel):
    """A single field report from one agency / unit."""
    report_id: str
    agency: str
    zone: str
    casualty_count: int
    severity: str
    hazards: List[str]
    resources_needed: List[str]
    timestamp: int  # step number when received
    reliability_score: float = Field(1.0, ge=0.0, le=1.0)  # ground-truth, hidden from agent
    content: str    # Human-readable summary sent to agent


class ResourceStatus(BaseModel):
    resource_id: str
    resource_type: ResourceType
    assigned_zone: Optional[str]
    available: bool
    eta_steps: int = 0


class ICDEObservation(BaseModel):
    """
    Full observation returned to the agent each step.
    """
    step: int
    task_id: str
    incident_type: str
    active_zones: List[str]
    field_reports: List[FieldReport]
    available_resources: List[ResourceStatus]
    assigned_resources: List[ResourceStatus]
    recent_events: List[str]          # Plain-English event log (last 5 events)
    civilian_status: Dict[str, int]   # {"safe": N, "at_risk": N, "casualties": N}
    time_remaining: int               # Steps left before episode timeout
    warnings: List[str]               # Active unresolved warnings
    last_action_feedback: Optional[str] = None


# ─── State (internal, returned by state()) ───────────────────────────────────

class ICDEState(BaseModel):
    """Full internal state — includes ground-truth values hidden from agent."""
    step: int
    task_id: str
    episode_id: str
    done: bool
    total_reward: float
    civilian_safe: int
    civilian_at_risk: int
    civilian_casualties: int
    optimal_casualties: int           # What perfect play achieves
    resource_double_assigned: bool
    cascade_triggered: bool
    last_action: Optional[ICDEAction]
    warnings_ignored: List[str]
    action_history: List[str]
    grader_subscores: Dict[str, float]


# ─── Reward ──────────────────────────────────────────────────────────────────

class ICDEReward(BaseModel):
    """Decomposed reward signal for interpretability."""
    total: float = Field(..., ge=-1.0, le=1.0)
    life_safety: float = 0.0
    resource_efficiency: float = 0.0
    conflict_resolution: float = 0.0
    protocol_compliance: float = 0.0
    penalty_loop: float = 0.0
    penalty_cascade: float = 0.0
    explanation: str = ""


# ─── Step Result ─────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: ICDEObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
