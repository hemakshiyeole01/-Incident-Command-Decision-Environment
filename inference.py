"""
Inference Script — ICDE (Incident Command Decision Environment)
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    ICDE_SERVER_URL     URL of the running ICDE server (default: http://localhost:7860)

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=task1_resource env=icde model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action={"command":"establish_command"} reward=0.65 done=false error=null
    [STEP] step=2 action={"command":"dispatch","resource_type":"hazmat","target_zone":"zone_b"} reward=0.85 done=false error=null
    [STEP] step=3 action={"command":"issue_directive","directive":"stand by"} reward=0.85 done=true error=null
    [END] success=true steps=3 score=0.85 rewards=0.65,0.85,0.85
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required")

# ── Runtime config ────────────────────────────────────────────────────────────
SERVER_URL = os.getenv("ICDE_SERVER_URL", "http://localhost:7860")
BENCHMARK  = "icde"

TASK_CONFIGS: Dict[str, Dict] = {
    "task1_resource":    {"max_steps": 15, "success_threshold": 0.50},
    "task2_multiagency": {"max_steps": 20, "success_threshold": 0.45},
    "task3_cascade":     {"max_steps": 30, "success_threshold": 0.35},
}

TEMPERATURE = 0.2
MAX_TOKENS  = 400

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Logging helpers (exact stdout format required by OpenEnv spec) ─────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Flatten any internal newlines so the line stays single-line
    action_safe = " ".join(action.split())
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── ICDE server HTTP client ───────────────────────────────────────────────────

def env_reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    resp = requests.post(
        f"{SERVER_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{SERVER_URL}/step",
        json={"task_id": task_id, "action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_grade(task_id: str) -> float:
    try:
        resp = requests.get(
            f"{SERVER_URL}/grade",
            params={"task_id": task_id},
            timeout=15,
        )
        resp.raise_for_status()
        return float(resp.json().get("score", 0.0))
    except Exception as e:
        print(f"[DEBUG] Grading failed: {e}", flush=True)
        return 0.0


def env_close(task_id: str) -> None:
    """Stateless server — no persistent connection to close. Spec-compliant placeholder."""
    pass


# ── System prompt (Incident Commander role) ───────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI Incident Commander using the ICS (Incident Command System).
Your goal: minimize civilian casualties, allocate resources efficiently, and resolve conflicting field reports.

PRIORITY ORDER (never deviate):
1. Life safety — always highest
2. Incident stabilization
3. Property preservation

KEY RULES:
- Dispatch HAZMAT before ENGINE units to any zone with confirmed chemical hazard
- Never send units into a structurally unsafe zone without engineering clearance
- Use establish_command in the first 1-2 steps
- Flag unreliable report IDs with flag_conflict (use the flags list)
- Do NOT repeat the exact same action 3+ times — it triggers a loop penalty
- Use issue_directive to communicate orders when no dispatch is needed

AVAILABLE COMMANDS:
  dispatch            — send a resource to a zone (requires resource_type + target_zone)
  recall              — pull a resource back from a zone
  establish_command   — set up unified incident command
  flag_conflict       — mark unreliable report IDs (requires flags list of report_ids)
  escalate            — escalate severity for a zone
  stand_down          — mark a zone as resolved
  request_mutual_aid  — request additional resources
  issue_directive     — issue a free-text order to field units

RESOURCE TYPES: engine | hazmat | medical | police | rescue | power
PRIORITY LEVELS: critical | high | medium | low

Respond with EXACTLY one JSON object — no markdown fences, no explanation, no extra text:
{
  "command": "<command>",
  "resource_type": "<type or null>",
  "target_zone": "<zone_id or null>",
  "priority": "<critical|high|medium|low>",
  "directive": "<text or null>",
  "flags": ["<report_id>", ...]
}
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int, last_error: Optional[str]) -> str:
    """Convert an observation dict into the LLM user prompt."""
    field_reports = obs.get("field_reports", [])
    reports_text  = "\n".join(
        f"  [{r['report_id']}] {r['agency']} @ {r['zone']}: {r['content']}"
        for r in field_reports[-6:]          # Cap at 6 to control context length
    ) or "  (none)"

    available_res = ", ".join(
        f"{r['resource_type']}({r['resource_id']})"
        for r in obs.get("available_resources", [])
    ) or "none"

    assigned_res = ", ".join(
        f"{r['resource_type']}→{r['assigned_zone']}"
        for r in obs.get("assigned_resources", [])
    ) or "none"

    civ    = obs.get("civilian_status", {})
    warns  = obs.get("warnings", [])[:6]
    recent = obs.get("recent_events", [])[-3:]

    warning_block = "\n".join(f"  ⚠️  {w}" for w in warns) or "  (none)"
    event_block   = "\n".join(f"  {e}" for e in recent)     or "  (none)"
    error_note    = (
        f"\n⚠️  PREVIOUS ACTION ERROR: {last_error} — choose a different valid action."
        if last_error else ""
    )

    return textwrap.dedent(f"""
STEP {step} | Task: {obs.get('task_id')} | Time remaining: {obs.get('time_remaining')} steps
Incident: {obs.get('incident_type')}
Active zones: {', '.join(obs.get('active_zones', []))}

CIVILIAN STATUS:
  safe={civ.get('safe', 0)}  at_risk={civ.get('at_risk', 0)}  casualties={civ.get('casualties', 0)}

FIELD REPORTS:
{reports_text}

AVAILABLE RESOURCES: {available_res}
ASSIGNED RESOURCES:  {assigned_res}

ACTIVE WARNINGS:
{warning_block}

RECENT EVENTS:
{event_block}
{error_note}
Your command (one JSON object only):
""").strip()


# ── LLM action selection ──────────────────────────────────────────────────────

def get_llm_action(obs: Dict[str, Any], step: int, last_error: Optional[str]) -> Dict[str, Any]:
    """
    Call the LLM with current observation and return a parsed action dict.
    Falls back to a safe default action on any parse or API error.
    """
    user_prompt = build_user_prompt(obs, step, last_error)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if the model added them
        if content.startswith("```"):
            content = "\n".join(
                line for line in content.split("\n")
                if not line.startswith("```")
            ).strip()

        action = json.loads(content)

        if "command" not in action:
            raise ValueError("Missing 'command' field in LLM response")

        # Normalise "null" strings → Python None
        for field in ("resource_type", "target_zone", "directive"):
            if action.get(field) in ("null", "none", "", "None", None):
                action[field] = None
        if not isinstance(action.get("flags"), list):
            action["flags"] = []

        return action

    except (json.JSONDecodeError, ValueError) as parse_err:
        print(f"[DEBUG] JSON parse error at step {step}: {parse_err}", flush=True)
        return {
            "command":       "issue_directive",
            "resource_type": None,
            "target_zone":   None,
            "priority":      "high",
            "directive":     "Assessing situation — all units hold position.",
            "flags":         [],
        }
    except Exception as api_err:
        print(f"[DEBUG] LLM call failed at step {step}: {api_err}", flush=True)
        return {
            "command":       "establish_command",
            "resource_type": None,
            "target_zone":   None,
            "priority":      "high",
            "directive":     None,
            "flags":         [],
        }


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = 42) -> Dict[str, Any]:
    """
    Run one full episode for task_id.

    Emits to stdout:
      [START] once
      [STEP]  once per step (immediately after env.step returns)
      [END]   always, even on exception (after env.close)

    Returns a summary dict.
    """
    cfg       = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    threshold = cfg["success_threshold"]

    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False
    obs: Dict[str, Any] = {}

    # ── [START] ────────────────────────────────────────────────────────────
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs        = env_reset(task_id, seed=seed)
        last_error: Optional[str] = None

        for step in range(1, max_steps + 1):

            # Get LLM action
            action     = get_llm_action(obs, step, last_error)
            action_str = json.dumps(action, separators=(",", ":"))

            # Call environment
            try:
                result        = env_step(task_id, action)
                reward        = float(result.get("reward", 0.0))
                done          = bool(result.get("done", False))
                info          = result.get("info", {})
                obs           = result.get("observation", obs)

                action_result = info.get("action_result", "")
                last_error    = action_result if str(action_result).startswith("error:") else None

            except requests.HTTPError as http_err:
                last_error = str(http_err)
                reward     = 0.0
                done       = False

            # ── [STEP] ─────────────────────────────────────────────────────
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

            if done:
                break

        # Grade the completed episode
        score   = env_grade(task_id)
        score   = min(max(score, 0.0), 1.0)   # clamp [0,1]
        success = score >= threshold

    except Exception as exc:
        print(f"[DEBUG] Episode exception: {exc}", flush=True)

    finally:
        # ── [END] always emitted, even on exception ─────────────────────
        try:
            env_close(task_id)
        except Exception as close_err:
            print(f"[DEBUG] env.close() error: {close_err}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score":   score,
        "success": success,
        "steps":   steps_taken,
        "rewards": rewards,
    }


# ── Server readiness check ────────────────────────────────────────────────────

def wait_for_server(max_retries: int = 30, delay: float = 2.0) -> None:
    """Poll /health until the ICDE FastAPI server responds 200."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                print(f"[INFO] ICDE server ready at {SERVER_URL}", flush=True)
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"[INFO] Waiting for ICDE server... ({attempt + 1}/{max_retries})", flush=True)
        time.sleep(delay)
    raise RuntimeError(
        f"ICDE server not available at {SERVER_URL} after {max_retries} retries"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ICDE Baseline Inference — evaluates an LLM agent across all three tasks"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_CONFIGS.keys()),
        choices=list(TASK_CONFIGS.keys()),
        help="Task IDs to evaluate (default: all three)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Skip server readiness polling (use if server is already confirmed up)",
    )
    args = parser.parse_args()

    if not args.no_wait:
        wait_for_server()

    results = []
    for task_id in args.tasks:
        summary = run_episode(task_id, seed=args.seed)
        results.append(summary)

    # Summary goes to stderr so it doesn't pollute the required stdout format
    print("\n[SUMMARY]", file=sys.stderr)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task_id']:25s}  score={r['score']:.3f}  steps={r['steps']:3d}  [{status}]",
            file=sys.stderr,
        )
    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"  {'AVERAGE':25s}  score={avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
