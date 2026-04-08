---
title: ICDE OpenEnv
emoji: 🚨
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - emergency-management
  - reinforcement-learning
  - agent-benchmark
  - ics
pinned: false
---

# 🚨 Incident Command Decision Environment (ICDE)

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://docker.com)

> **An OpenEnv-compliant benchmark where an AI agent plays Incident Commander managing real-world emergencies using the ICS (Incident Command System) — the exact framework used by FEMA, hospitals, fire departments, and militaries worldwide.**

---

## 🎯 Overview & Motivation

Real emergency response requires:
- **Conflicting information** — 4 agencies reporting simultaneously with contradictory data
- **Causal reasoning** — wrong step 3 decisions show up as casualties at step 18
- **Resource scarcity math** — cannot send 3 units to 5 locations simultaneously  
- **Protocol knowledge** — ICS structure matters, not just intent
- **Adaptive response** — new complications inject mid-episode

This makes ICDE a rigorous benchmark for evaluating whether LLMs can reason under uncertainty, reconcile conflicting information, and make defensible decisions with real consequences — skills that go far beyond trivia or code generation.

---

## 🏗️ Project Structure

```
icde/
├── env/
│   ├── __init__.py       # Package exports
│   ├── models.py         # Pydantic: Action, Observation, State, Reward
│   ├── environment.py    # Core: step() / reset() / state()
│   ├── reward.py         # Dense reward shaping logic
│   └── simulator.py      # Incident simulation engine
├── tasks/                # Task definitions (see graders)
├── graders/
│   ├── __init__.py
│   ├── grader1.py        # Task 1: Resource allocation
│   ├── grader2.py        # Task 2: Multi-agency conflict
│   └── grader3.py        # Task 3: Cascading crisis
├── data/
│   └── scenarios.json    # 3 fully-defined incident scenarios
├── server.py             # FastAPI server (OpenEnv HTTP API)
├── inference.py          # ✅ OpenAI client baseline script (ROOT)
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧪 Tasks

### Task 1 — Resource Allocation Under Scarcity (Easy)
**Scenario:** Industrial warehouse fire with 3 active zones and 4 limited resources.

| Zone | Threat | Resources Needed |
|------|--------|-----------------|
| Zone A (Loading Dock) | Active flames, 4 trapped | Engine |
| Zone B (Chemical Storage) | Acetylene cylinders, explosion risk | **Hazmat FIRST**, then Engine |
| Zone C (Office Wing) | Smoke inhalation | Medical |

**Challenge:** Conflicting reports — Security Guard says "zone A looks clear" (reliability: 0.3), Engine 7 confirms 4 workers trapped. Agent must flag unreliable report.

**Grader criteria (→ 1.0):**
- Hazmat dispatched to Zone B: +0.35
- No double assignment: +0.25  
- Unreliable report flagged: +0.20
- Medical to Zone C: +0.10
- No looping behavior: +0.10

**Baseline score:** ~0.60

---

### Task 2 — Multi-Agency Conflict Resolution (Medium)
**Scenario:** Hospital mass casualty event. Fire, EMS, Police, and Hospital all reporting simultaneously with contradictory casualty counts and safety assessments.

**Key conflicts:**
- Fire IC: "15 casualties inside" vs EMS: "8 transported, 20 still on scene"
- Hospital Director: "ER structurally unsafe — do NOT enter"  
- Police Sergeant: "ER looks fine, safe to enter" ← **reliability: 0.25, should be flagged**

**Cascade triggers if not addressed:**
- No rescue to ER by step 10 → 3 critical patients die
- No police to access road by step 6 → ambulance blocked, 2 deaths

**Baseline score:** ~0.45

---

### Task 3 — Cascading Multi-Site Crisis (Hard)
**Scenario:** Simultaneous earthquake + gas main rupture + hospital power failure across a city grid. 30 steps. Early decisions have irreversible downstream consequences.

**5 active sites:**
| Site | Threat | Critical Window |
|------|--------|----------------|
| Gas District | 16-inch rupture, explosion risk | Cascade at step 12 without hazmat |
| Hospital | 45 ICU patients, 2hr battery | Cascade at step 15 without power unit |
| Residential | Partial collapse, 15 trapped | Cascade at step 20 without rescue |
| School | 200 students, structural damage | Low severity, can wait |
| Bridge | Cracked — no heavy vehicles | Routing constraint |

**Injected complications:** aftershock at step 5, hospital update at step 10, road closure at step 18.

**Cascade chains:** Wrong priority decision at step 1-3 → hospital loses life support at step 15 → 8 ICU patients die.

**Baseline score:** ~0.30

---

## 🏆 Action & Observation Spaces

### Action Space (`ICDEAction`)
```python
class ICDEAction(BaseModel):
    command: CommandAction          # dispatch | recall | establish_command | 
                                    # flag_conflict | escalate | stand_down |
                                    # request_mutual_aid | issue_directive
    resource_type: Optional[str]    # engine | hazmat | medical | police | rescue | power
    target_zone: Optional[str]      # zone identifier
    priority: Optional[str]         # critical | high | medium | low
    directive: Optional[str]        # free-text order (max 500 chars)
    flags: Optional[List[str]]      # report IDs to flag as unreliable
```

### Observation Space (`ICDEObservation`)
```python
class ICDEObservation(BaseModel):
    step: int
    task_id: str
    incident_type: str
    active_zones: List[str]
    field_reports: List[FieldReport]         # Per-agency reports (reliability hidden)
    available_resources: List[ResourceStatus]
    assigned_resources: List[ResourceStatus]
    recent_events: List[str]                 # Last 5 events
    civilian_status: Dict[str, int]          # safe / at_risk / casualties
    time_remaining: int
    warnings: List[str]                      # Active unresolved warnings
```

### Reward Function (Dense, Range: -1.0 to +1.0)
| Component | Range | Description |
|-----------|-------|-------------|
| `life_safety` | -0.4 to +0.4 | Civilian outcomes vs optimal |
| `resource_efficiency` | -0.15 to +0.15 | No double-assignment |
| `conflict_resolution` | 0 to +0.2 | Correct flags raised |
| `protocol_compliance` | 0 to +0.1 | ICS structure followed |
| `penalty_loop` | -0.2 | 3x repeated identical action |
| `penalty_cascade` | -0.3 | Preventable cascade triggered |

---

## 🚀 Setup & Usage

### Local Development

```bash
git clone <repo>
cd icde
pip install -r requirements.txt
python server.py
```

### Run Baseline Inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
docker build -t icde .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN icde
```

### API Usage

```bash
# Reset task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_resource"}'

# Take action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task1_resource",
    "action": {
      "command": "dispatch",
      "resource_type": "hazmat",
      "target_zone": "zone_b",
      "priority": "critical"
    }
  }'

# Get current state
curl http://localhost:7860/state?task_id=task1_resource

# Get episode grade
curl http://localhost:7860/grade?task_id=task1_resource
```

---

## 📊 Baseline Performance

| Task | Difficulty | Baseline Score | Threshold |
|------|-----------|---------------|-----------|
| task1_resource | Easy | 0.60 | 0.50 |
| task2_multiagency | Medium | 0.45 | 0.45 |
| task3_cascade | Hard | 0.30 | 0.35 |

*Baseline model: Qwen/Qwen2.5-72B-Instruct via Hugging Face Inference API*

---

## 🔒 Anti-Reward-Hacking Design

- **Life safety** verified by simulated outcomes — not keyword matching
- **Protocol compliance** checked structurally — not by buzzwords
- **Cascade penalties** only fire on verified causal chains
- **Reliability scores** are hidden from agent — must reason about contradictions
- **Loop detection** prevents degenerate policies

---

## 📋 OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

---

## 🔬 Real-World Basis

The ICS framework used in this environment is the actual system mandated by FEMA for all US incident management. The scenario designs are based on:
- NIMS (National Incident Management System) training scenarios
- Multi-agency coordination case studies from FEMA training materials
- Hospital emergency operations plan templates

This makes ICDE directly applicable to training AI systems for:
- Emergency dispatch assistance
- Incident Commander decision support
- Multi-agency coordination tools
- Crisis simulation and training

---

## 📄 License

MIT License — see LICENSE file.
