"""
Grader for Task 2: Multi-Agency Conflict Resolution
Score: 0.0 – 1.0 (deterministic, reproducible)

Criteria
--------
1. Unified command established              (+0.20) — establish_command action taken
2. Life safety prioritized correctly        (+0.25) — rescue dispatched to ER, police to road
3. Contradictions flagged                   (+0.25) — rpt_police_2 (sergeant low-reliability) flagged
4. Casualty minimization                    (+0.20) — final casualties vs. optimal (2)
5. Coordination actions                     (+0.15) — mutual aid + directives + hazard escalation
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade(
    action_history: List[str],
    flagged_reports: List[str],
    resource_assignments: Dict[str, str],
    civilian_casualties: int,
    cascade_triggered: bool,
    grader_subscores: Dict[str, float],
    command_established: bool = False,
    steps_to_first_command: int = 99,
) -> Dict[str, Any]:
    """
    Deterministic grader for Task 2.

    Parameters
    ----------
    action_history           : list of "command:resource_type:zone" strings
    flagged_reports          : report IDs flagged as unreliable
    resource_assignments     : resource_id -> zone at episode end
    civilian_casualties      : total casualties at episode end
    cascade_triggered        : any cascade triggered?
    grader_subscores         : running sub-scores from simulator
    command_established      : True if establish_command was issued
    steps_to_first_command   : step number when command was first established

    Returns
    -------
    dict with score (0–1), breakdown, passed, task, difficulty
    """

    score = 0.0
    breakdown = {}

    # ── Criterion 1: Unified command established ──────────────────────────────
    # Partial credit for late establishment
    if command_established:
        if steps_to_first_command <= 3:
            c1 = 0.20
        elif steps_to_first_command <= 7:
            c1 = 0.12
        else:
            c1 = 0.06
    else:
        c1 = 0.0
    breakdown["unified_command_established"] = c1
    score += c1

    # ── Criterion 2: Life safety — rescue to ER, police to road ──────────────
    rescue_to_er = any(
        "rescue" in action and "site_er" in action
        for action in action_history
    )
    police_to_road = any(
        "police" in action and "site_road" in action
        for action in action_history
    )
    c2 = 0.0
    if rescue_to_er:
        c2 += 0.15
    if police_to_road:
        c2 += 0.10
    breakdown["life_safety_dispatch"] = c2
    score += c2

    # ── Criterion 3: Unreliable Police Sergeant report flagged ───────────────
    # rpt_police_2 has reliability_score=0.25 — should be flagged
    sergeant_flagged = "rpt_police_2" in flagged_reports
    c3 = 0.25 if sergeant_flagged else 0.0
    breakdown["contradiction_flagged"] = c3
    score += c3

    # ── Criterion 4: Casualty minimization ───────────────────────────────────
    optimal_casualties = 2
    excess = max(0, civilian_casualties - optimal_casualties)
    if excess == 0:
        c4 = 0.20
    elif excess <= 3:
        c4 = 0.12
    elif excess <= 7:
        c4 = 0.06
    else:
        c4 = 0.0
    breakdown["casualty_minimization"] = c4
    score += c4

    # ── Criterion 5: Coordination actions ────────────────────────────────────
    mutual_aid_requested = any("request_mutual_aid" in action for action in action_history)
    directives_issued = sum(1 for action in action_history if "issue_directive" in action)
    hazard_flagged = any("flag_conflict" in action or "escalate" in action for action in action_history)
    c5 = 0.0
    if mutual_aid_requested:
        c5 += 0.07
    if hazard_flagged:
        c5 += 0.05
    if directives_issued >= 1:
        c5 += 0.03
    c5 = min(0.15, c5)
    breakdown["coordination_actions"] = c5
    score += c5

    # ── Cascade penalty ───────────────────────────────────────────────────────
    if cascade_triggered:
        score = max(0.0, score - 0.05)
        breakdown["cascade_penalty"] = -0.05
    else:
        breakdown["cascade_penalty"] = 0.0

    # ── Clamp ────────────────────────────────────────────────────────────────
    score = min(1.0, max(0.0, round(score, 4)))
    return {
        "score": score,
        "breakdown": breakdown,
        "passed": score >= 0.45,
        "task": "task2_multiagency",
        "difficulty": "medium",
    }
