"""
Reward shaping helpers for ICDE.
All reward components are deterministic given the same state.
"""

from __future__ import annotations

from env.models import ICDEAction, ICDEReward, ICDEState


def shape_reward(state: ICDEState, action: ICDEAction) -> ICDEReward:
    """
    Compute the dense reward from state + action.
    Called externally by tests and the grader.
    """
    life_safety = _life_safety(state)
    resource_efficiency = _resource_efficiency(state)
    conflict_resolution = _conflict_resolution(state)
    protocol_compliance = _protocol_compliance(state)
    penalty_loop = _penalty_loop(state)
    penalty_cascade = _penalty_cascade(state)

    total = (
        life_safety
        + resource_efficiency
        + conflict_resolution
        + protocol_compliance
        + penalty_loop
        + penalty_cascade
    )
    total = max(-1.0, min(1.0, total))

    return ICDEReward(
        total=total,
        life_safety=life_safety,
        resource_efficiency=resource_efficiency,
        conflict_resolution=conflict_resolution,
        protocol_compliance=protocol_compliance,
        penalty_loop=penalty_loop,
        penalty_cascade=penalty_cascade,
        explanation=(
            f"life_safety={life_safety:.2f} resource={resource_efficiency:.2f} "
            f"conflict={conflict_resolution:.2f} protocol={protocol_compliance:.2f} "
            f"loop={penalty_loop:.2f} cascade={penalty_cascade:.2f}"
        ),
    )


def _life_safety(state: ICDEState) -> float:
    """
    Higher when fewer casualties relative to optimal.
    Max +0.4, min -0.4.
    """
    excess = state.civilian_casualties - state.optimal_casualties
    if excess <= 0:
        return 0.4
    elif excess <= 2:
        return 0.2
    elif excess <= 5:
        return 0.0
    else:
        return max(-0.4, -0.05 * excess)


def _resource_efficiency(state: ICDEState) -> float:
    """
    Penalizes double-assignment, rewards clean allocation.
    """
    if state.resource_double_assigned:
        return -0.15
    return 0.15


def _conflict_resolution(state: ICDEState) -> float:
    """
    Rewards flagging unreliable reports vs. ignoring discrepancies.
    """
    score = state.grader_subscores.get("conflict_resolution", 0.0)
    return min(0.2, score)


def _protocol_compliance(state: ICDEState) -> float:
    """
    ICS protocol: unified command, proper escalation paths.
    """
    score = state.grader_subscores.get("protocol_compliance", 0.0)
    return min(0.1, score)


def _penalty_loop(state: ICDEState) -> float:
    """
    Penalizes repeating the same action 3x in a row.
    """
    if len(state.action_history) >= 3:
        last3 = state.action_history[-3:]
        if len(set(last3)) == 1:
            return -0.2
    return 0.0


def _penalty_cascade(state: ICDEState) -> float:
    """
    Penalizes cascade events that could have been prevented.
    """
    if state.cascade_triggered:
        return -0.3
    return 0.0
