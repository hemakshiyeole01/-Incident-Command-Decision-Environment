"""
Incident simulation engine for ICDE.
Manages scenario progression, cascades, resource tracking, and civilian outcomes.
"""

from __future__ import annotations

import random
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    CommandAction,
    FieldReport,
    ICDEAction,
    ICDEObservation,
    Priority,
    ResourceStatus,
    ResourceType,
)


class IncidentSimulator:
    """
    Simulates an evolving incident over multiple steps.
    Handles: resource assignment, cascade triggers, report injection,
    civilian fate tracking, and contradiction generation.
    """

    def __init__(self, scenario: Dict[str, Any], seed: int = 42):
        self.scenario = deepcopy(scenario)
        self.seed = seed
        self._rng = random.Random(seed)
        self.reset()

    # ─── Reset ───────────────────────────────────────────────────────────────

    def reset(self) -> ICDEObservation:
        self._rng = random.Random(self.seed)
        self.step_num = 0
        self.done = False
        self.episode_id = str(uuid.uuid4())[:8]

        sc = self.scenario
        self.task_id: str = sc["task_id"]
        self.incident_type: str = sc["incident_type"]
        self.max_steps: int = sc.get("max_steps", 20)
        self.active_zones: List[str] = list(sc["zones"].keys())

        # Civilian tracking
        self.civilian_safe: int = sc["civilians"]["safe"]
        self.civilian_at_risk: int = sc["civilians"]["at_risk"]
        self.civilian_casualties: int = sc["civilians"]["casualties"]
        self.optimal_casualties: int = sc.get("optimal_casualties", 0)

        # Resource pool
        self.resources: Dict[str, ResourceStatus] = {}
        for r in sc["resources"]:
            rs = ResourceStatus(
                resource_id=r["id"],
                resource_type=r["type"],
                assigned_zone=None,
                available=True,
                eta_steps=0,
            )
            self.resources[r["id"]] = rs

        # Zone state
        self.zone_state: Dict[str, Dict[str, Any]] = deepcopy(sc["zones"])

        # Event / warning tracking
        self.events: List[str] = []
        self.warnings: List[str] = list(sc.get("initial_warnings", []))
        self.warnings_ignored: List[str] = []

        # Cascade tracking
        self.cascade_triggered: bool = False
        self.cascade_conditions: Dict[str, Dict] = deepcopy(sc.get("cascades", {}))

        # Action history for loop detection
        self.action_history: List[str] = []
        self.last_action: Optional[ICDEAction] = None

        # Report injection schedule: {step: [report_dict, ...]}
        self.report_schedule: Dict[int, List[Dict]] = deepcopy(sc.get("report_schedule", {}))
        self.active_reports: List[FieldReport] = self._make_initial_reports()

        # Grader sub-scores accumulated
        self.grader_subscores: Dict[str, float] = {
            "life_safety": 0.0,
            "resource_efficiency": 0.0,
            "conflict_resolution": 0.0,
            "protocol_compliance": 0.0,
        }

        # Flags raised by agent
        self.flagged_reports: List[str] = []

        return self._build_observation()

    # ─── Step ────────────────────────────────────────────────────────────────

    def step(self, action: ICDEAction) -> Tuple[ICDEObservation, float, bool, Dict]:
        if self.done:
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "Episode already done"}

        self.step_num += 1
        self.last_action = action

        # 1. Process action
        action_result = self._process_action(action)

        # 2. Inject new reports (per schedule)
        self._inject_reports()

        # 3. Advance simulation (civilian outcomes, cascade checks)
        self._advance_simulation(action)

        # 4. Compute reward
        reward_obj = self._compute_reward(action, action_result)
        reward = reward_obj.total

        # 5. Check terminal condition
        self.done = self._check_done()

        # 6. Build observation
        obs = self._build_observation()
        info = {
            "action_result": action_result,
            "reward_breakdown": reward_obj.dict(),
            "cascade_triggered": self.cascade_triggered,
            "grader_subscores": dict(self.grader_subscores),
        }

        return obs, reward, self.done, info

    # ─── Action Processing ───────────────────────────────────────────────────

    def _process_action(self, action: ICDEAction) -> str:
        cmd = action.command

        if cmd == CommandAction.DISPATCH:
            return self._dispatch(action)
        elif cmd == CommandAction.RECALL:
            return self._recall(action)
        elif cmd == CommandAction.ESTABLISH_COMMAND:
            return self._establish_command(action)
        elif cmd == CommandAction.FLAG_CONFLICT:
            return self._flag_conflict(action)
        elif cmd == CommandAction.ESCALATE:
            self.events.append(f"Step {self.step_num}: Escalation issued for {action.target_zone}")
            return "escalated"
        elif cmd == CommandAction.STAND_DOWN:
            return self._stand_down(action)
        elif cmd == CommandAction.REQUEST_MUTUAL_AID:
            self.events.append(f"Step {self.step_num}: Mutual aid requested")
            return "mutual_aid_requested"
        elif cmd == CommandAction.ISSUE_DIRECTIVE:
            self.events.append(f"Step {self.step_num}: Directive issued: {action.directive[:80] if action.directive else 'N/A'}")
            return "directive_issued"
        return "no_op"

    def _dispatch(self, action: ICDEAction) -> str:
        if not action.resource_type or not action.target_zone:
            return "error:missing_resource_or_zone"

        rtype = action.resource_type
        zone = action.target_zone

        if zone not in self.active_zones:
            return f"error:unknown_zone:{zone}"

        # Find available resource of requested type
        available = [
            r for r in self.resources.values()
            if r.resource_type == rtype and r.available
        ]

        if not available:
            self.events.append(f"Step {self.step_num}: DISPATCH FAILED — no {rtype} available")
            return f"error:no_{rtype}_available"

        # Check double-assignment (same type already in zone)
        already_assigned = [
            r for r in self.resources.values()
            if r.resource_type == rtype and r.assigned_zone == zone
        ]
        if already_assigned:
            self.grader_subscores["resource_efficiency"] -= 0.1
            self.events.append(f"Step {self.step_num}: WARNING — {rtype} already in {zone}")

        unit = available[0]
        unit.available = False
        unit.assigned_zone = zone
        unit.eta_steps = 1

        # Mark zone as being handled
        if zone in self.zone_state:
            self.zone_state[zone]["units_assigned"] = (
                self.zone_state[zone].get("units_assigned", 0) + 1
            )

        self.events.append(f"Step {self.step_num}: Dispatched {rtype} → {zone}")
        return f"dispatched:{rtype}:{zone}"

    def _recall(self, action: ICDEAction) -> str:
        if not action.resource_type or not action.target_zone:
            return "error:missing_resource_or_zone"

        rtype = action.resource_type
        zone = action.target_zone

        recalled = [
            r for r in self.resources.values()
            if r.resource_type == rtype and r.assigned_zone == zone
        ]
        if not recalled:
            return f"error:no_{rtype}_in_{zone}"

        unit = recalled[0]
        unit.available = True
        unit.assigned_zone = None
        self.events.append(f"Step {self.step_num}: Recalled {rtype} from {zone}")
        return f"recalled:{rtype}:{zone}"

    def _establish_command(self, action: ICDEAction) -> str:
        zone = action.target_zone or "unified"
        self.events.append(f"Step {self.step_num}: Unified command established at {zone}")
        self.grader_subscores["protocol_compliance"] = min(
            self.grader_subscores["protocol_compliance"] + 0.15, 0.3
        )
        return "command_established"

    def _flag_conflict(self, action: ICDEAction) -> str:
        flags = action.flags or []
        if not flags:
            return "error:no_flags_provided"

        for flag in flags:
            if flag not in self.flagged_reports:
                self.flagged_reports.append(flag)

        # Check if flagged reports are actually unreliable
        unreliable_ids = {
            r.report_id for r in self.active_reports if r.reliability_score < 0.6
        }
        correct_flags = [f for f in flags if f in unreliable_ids]
        wrong_flags = [f for f in flags if f not in unreliable_ids]

        score_delta = len(correct_flags) * 0.1 - len(wrong_flags) * 0.05
        self.grader_subscores["conflict_resolution"] = min(
            self.grader_subscores["conflict_resolution"] + score_delta, 0.4
        )
        self.events.append(f"Step {self.step_num}: Flagged conflicts: {flags} (correct: {correct_flags})")
        return f"flagged:{len(correct_flags)}_correct:{len(wrong_flags)}_wrong"

    def _stand_down(self, action: ICDEAction) -> str:
        zone = action.target_zone
        if zone and zone in self.zone_state:
            self.zone_state[zone]["status"] = "resolved"
            self.events.append(f"Step {self.step_num}: Stand-down ordered for {zone}")
            return f"stand_down:{zone}"
        return "error:invalid_zone_for_stand_down"

    # ─── Simulation Advancement ──────────────────────────────────────────────

    def _advance_simulation(self, action: ICDEAction):
        # Reduce ETA for in-transit units
        for r in self.resources.values():
            if r.eta_steps > 0:
                r.eta_steps -= 1

        # Advance zone deterioration for unhandled zones
        for zone_id, zs in self.zone_state.items():
            if zs.get("status") == "resolved":
                continue
            units = zs.get("units_assigned", 0)
            if units == 0 and zs.get("deteriorates", False):
                zs["severity"] = min(zs.get("severity", 1) + 1, 5)
                if zs["severity"] >= 4 and zone_id in self.warnings:
                    self.warnings_ignored.append(zone_id)
                    # Increase at-risk civilians
                    transfer = min(self.civilian_at_risk, 2)
                    self.civilian_casualties += transfer
                    self.civilian_at_risk = max(0, self.civilian_at_risk - transfer)

        # Check cascade triggers
        self._check_cascades(action)

        # Move at-risk civilians to safe when units on scene
        for zone_id, zs in self.zone_state.items():
            if zs.get("units_assigned", 0) >= 1:
                rescue = min(self.civilian_at_risk, 3)
                self.civilian_safe += rescue
                self.civilian_at_risk = max(0, self.civilian_at_risk - rescue)

    def _check_cascades(self, action: ICDEAction):
        for cascade_id, cond in self.cascade_conditions.items():
            if cond.get("triggered"):
                continue
            trigger_step = cond.get("trigger_step", 999)
            # Check if prevention action was taken
            prevention_cmd = cond.get("prevention_command")
            prevention_zone = cond.get("prevention_zone")
            prevented = False
            if prevention_cmd and prevention_zone:
                # Check action history
                for hist_action in self.action_history:
                    if prevention_cmd in hist_action and prevention_zone in hist_action:
                        prevented = True
                        break

            if not prevented and self.step_num >= trigger_step:
                # Cascade fires
                cond["triggered"] = True
                self.cascade_triggered = True
                effect = cond.get("effect", {})
                casualties = effect.get("casualties", 5)
                self.civilian_casualties += casualties
                self.civilian_at_risk = max(0, self.civilian_at_risk - casualties)
                warning_msg = cond.get("warning", "CASCADE EVENT TRIGGERED")
                self.warnings.append(f"CASCADE:{cascade_id}:{warning_msg}")
                self.events.append(f"Step {self.step_num}: ⚠️ CASCADE — {warning_msg}")

        # Track action history for cascade prevention
        # Normalize to string values (handles both real enums and str-enums)
        cmd_val = action.command.value if hasattr(action.command, 'value') else str(action.command)
        rtype_val = (
            action.resource_type.value if hasattr(action.resource_type, 'value')
            else str(action.resource_type)
        ) if action.resource_type else "none"
        zone_val = action.target_zone or "none"
        action_str = f"{cmd_val}:{rtype_val}:{zone_val}"
        self.action_history.append(action_str)

    # ─── Report Injection ────────────────────────────────────────────────────

    def _inject_reports(self):
        step_reports = self.report_schedule.get(self.step_num, [])
        for rdata in step_reports:
            fr = FieldReport(
                report_id=rdata["report_id"],
                agency=rdata["agency"],
                zone=rdata["zone"],
                casualty_count=rdata["casualty_count"],
                severity=rdata["severity"],
                hazards=rdata.get("hazards", []),
                resources_needed=rdata.get("resources_needed", []),
                timestamp=self.step_num,
                reliability_score=rdata.get("reliability_score", 1.0),
                content=rdata["content"],
            )
            self.active_reports.append(fr)
            if fr.reliability_score < 0.6:
                # Add to warnings as a discrepancy
                self.warnings.append(f"DISCREPANCY:report:{fr.report_id}:from:{fr.agency}")

    def _make_initial_reports(self) -> List[FieldReport]:
        reports = []
        for rdata in self.scenario.get("initial_reports", []):
            fr = FieldReport(
                report_id=rdata["report_id"],
                agency=rdata["agency"],
                zone=rdata["zone"],
                casualty_count=rdata["casualty_count"],
                severity=rdata["severity"],
                hazards=rdata.get("hazards", []),
                resources_needed=rdata.get("resources_needed", []),
                timestamp=0,
                reliability_score=rdata.get("reliability_score", 1.0),
                content=rdata["content"],
            )
            reports.append(fr)
        return reports

    # ─── Reward ──────────────────────────────────────────────────────────────

    def _compute_reward(self, action: ICDEAction, action_result: str) -> Any:
        from env.models import ICDEReward

        life_safety = 0.0
        resource_efficiency = 0.0
        conflict_resolution = 0.0
        protocol_compliance = 0.0
        penalty_loop = 0.0
        penalty_cascade = 0.0

        # Life safety: rescues relative to optimal
        if self.civilian_safe > 0 or self.civilian_casualties == 0:
            life_safety = 0.2
        if self.civilian_casualties < self.optimal_casualties + 2:
            life_safety += 0.2
        else:
            life_safety -= 0.1 * max(0, self.civilian_casualties - self.optimal_casualties - 2)

        # Resource efficiency: penalize double assignment
        assigned_zones = [r.assigned_zone for r in self.resources.values() if r.assigned_zone]
        if len(assigned_zones) == len(set(assigned_zones)):
            resource_efficiency = 0.15
        else:
            resource_efficiency = -0.1

        # Conflict resolution: flags raised vs. discrepancies
        discrepancy_count = len([w for w in self.warnings if w.startswith("DISCREPANCY")])
        if discrepancy_count > 0 and len(self.flagged_reports) > 0:
            conflict_resolution = min(len(self.flagged_reports) / discrepancy_count * 0.2, 0.2)

        # Protocol compliance from grader subscores
        protocol_compliance = min(self.grader_subscores.get("protocol_compliance", 0.0), 0.1)

        # Loop detection
        if len(self.action_history) >= 3:
            last_3 = self.action_history[-3:]
            if len(set(last_3)) == 1:
                penalty_loop = -0.2

        # Cascade penalty
        if self.cascade_triggered:
            penalty_cascade = -0.3

        # Clamp life_safety
        life_safety = max(-0.4, min(0.4, life_safety))

        total = (
            life_safety
            + resource_efficiency
            + conflict_resolution
            + protocol_compliance
            + penalty_loop
            + penalty_cascade
        )
        total = max(-1.0, min(1.0, total))

        explanation = (
            f"life_safety={life_safety:.2f} resource={resource_efficiency:.2f} "
            f"conflict={conflict_resolution:.2f} protocol={protocol_compliance:.2f} "
            f"loop={penalty_loop:.2f} cascade={penalty_cascade:.2f}"
        )

        return ICDEReward(
            total=total,
            life_safety=life_safety,
            resource_efficiency=resource_efficiency,
            conflict_resolution=conflict_resolution,
            protocol_compliance=protocol_compliance,
            penalty_loop=penalty_loop,
            penalty_cascade=penalty_cascade,
            explanation=explanation,
        )

    # ─── Terminal Check ──────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        if self.step_num >= self.max_steps:
            return True
        if self.civilian_at_risk == 0 and self.civilian_casualties >= 0:
            all_resolved = all(
                zs.get("status") == "resolved"
                for zs in self.zone_state.values()
            )
            if all_resolved:
                return True
        return False

    # ─── Build Observation ───────────────────────────────────────────────────

    def _build_observation(self) -> ICDEObservation:
        available = [r for r in self.resources.values() if r.available]
        assigned = [r for r in self.resources.values() if not r.available]
        recent = self.events[-5:] if self.events else []

        return ICDEObservation(
            step=self.step_num,
            task_id=self.task_id,
            incident_type=self.incident_type,
            active_zones=self.active_zones,
            field_reports=list(self.active_reports),
            available_resources=available,
            assigned_resources=assigned,
            recent_events=recent,
            civilian_status={
                "safe": self.civilian_safe,
                "at_risk": self.civilian_at_risk,
                "casualties": self.civilian_casualties,
            },
            time_remaining=self.max_steps - self.step_num,
            warnings=list(self.warnings),
        )
