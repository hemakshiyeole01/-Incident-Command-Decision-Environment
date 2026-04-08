"""
Grader for Task 1: Resource Allocation Under Scarcity
Score: 0.0 – 1.0 (deterministic, reproducible)

Criteria
--------
1. Life safety prioritized  (+0.35) — hazmat sent to Zone B (chemical zone)
2. No double assignment     (+0.25) — no resource type assigned to multiple zones simultaneously
3. Contradiction flagged    (+0.20) — rpt_003 (low-reliability guard report) flagged
4. Medical to zone_c        (+0.10) — medical sent where smoke inhalation reported
5. No looping behavior      (+0.10) — no 3x repeated action
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade(
    action_history: List[str],
    flagged_reports: List[str],
    resource_assignments: Dict[str, str],  # resource_id -> zone
    civilian_casualties: int,
    cascade_triggered: bool,
    grader_subscores: Dict[str, float],
) -> Dict[str, Any]:
    """
    Deterministic grader for Task 1.

    Parameters
    ----------
    action_history       : list of "command:resource_type:zone" strings
    flagged_reports      : list of report_ids flagged as unreliable by agent
    resource_assignments : current final state of resource → zone assignments
    civilian_casualties  : total casualties at episode end
    cascade_triggered    : whether the chemical explosion cascade fired
    grader_subscores     : running scores tracked by simulator

    Returns
    -------
    dict with keys: score (float 0–1), breakdown (dict), passed (bool)
    """

    score = 0.0
    breakdown = {}

    # ── Criterion 1: Hazmat dispatched to Zone B (chemical zone) ─────────────
    hazmat_in_zone_b = any(
        "hazmat" in action and "zone_b" in action
        for action in action_history
    )
    if hazmat_in_zone_b and not cascade_triggered:
        c1 = 0.35
    elif hazmat_in_zone_b:
        # Sent hazmat but too late (cascade still fired)
        c1 = 0.15
    else:
        c1 = 0.0
    breakdown["hazmat_to_chemical_zone"] = c1
    score += c1

    # ── Criterion 2: No double assignment ────────────────────────────────────
    zone_type_pairs = set()
    double_assigned = False
    for action in action_history:
        parts = action.split(":")
        if len(parts) >= 3 and parts[0] == "dispatch":
            key = (parts[1], parts[2])  # (resource_type, zone)
            if key in zone_type_pairs:
                double_assigned = True
                break
            zone_type_pairs.add(key)

    c2 = 0.0 if double_assigned else 0.25
    breakdown["no_double_assignment"] = c2
    score += c2

    # ── Criterion 3: Unreliable report flagged ───────────────────────────────
    # rpt_003 is the low-reliability security guard report
    unreliable_flagged = "rpt_003" in flagged_reports
    c3 = 0.20 if unreliable_flagged else 0.0
    breakdown["contradiction_flagged"] = c3
    score += c3

    # ── Criterion 4: Medical sent to Zone C ──────────────────────────────────
    medical_to_c = any(
        "medical" in action and "zone_c" in action
        for action in action_history
    )
    c4 = 0.10 if medical_to_c else 0.0
    breakdown["medical_to_smoke_zone"] = c4
    score += c4

    # ── Criterion 5: No looping behavior ─────────────────────────────────────
    loop_detected = False
    if len(action_history) >= 3:
        for i in range(len(action_history) - 2):
            if action_history[i] == action_history[i+1] == action_history[i+2]:
                loop_detected = True
                break
    c5 = 0.0 if loop_detected else 0.10
    breakdown["no_loop_behavior"] = c5
    score += c5

    # ── Clamp and return ─────────────────────────────────────────────────────
    score = min(1.0, max(0.0, round(score, 4)))
    return {
        "score": score,
        "breakdown": breakdown,
        "passed": score >= 0.5,
        "task": "task1_resource",
        "difficulty": "easy",
    }
