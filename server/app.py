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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from env.environment import ICDEEnvironment
from env.models import ICDEAction, ICDEObservation, ICDEState, StepResult
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
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ICDE OpenEnv Playground</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #ffffff;
    --bg2: #f7f7f5;
    --bg3: #f0efea;
    --border: rgba(0,0,0,0.12);
    --border2: rgba(0,0,0,0.2);
    --text: #1a1a1a;
    --text2: #555;
    --text3: #888;
    --green: #1D9E75;
    --green-bg: #E1F5EE;
    --green-text: #0F6E56;
    --red: #E24B4A;
    --red-bg: #FCEBEB;
    --red-text: #A32D2D;
    --blue: #378ADD;
    --blue-bg: #E6F1FB;
    --blue-text: #185FA5;
    --amber: #BA7517;
    --amber-bg: #FAEEDA;
    --amber-text: #854F0B;
    --purple: #7F77DD;
    --purple-bg: #EEEDFE;
    --purple-text: #3C3489;
    --radius: 8px;
    --radius-lg: 12px;
    --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1e1e1c;
      --bg2: #252523;
      --bg3: #2c2c2a;
      --border: rgba(255,255,255,0.1);
      --border2: rgba(255,255,255,0.18);
      --text: #f0eeea;
      --text2: #aaa;
      --text3: #666;
      --green-bg: #04342C;
      --green-text: #9FE1CB;
      --red-bg: #501313;
      --red-text: #F7C1C1;
      --blue-bg: #042C53;
      --blue-text: #B5D4F4;
      --amber-bg: #412402;
      --amber-text: #FAC775;
      --purple-bg: #26215C;
      --purple-text: #CECBF6;
    }
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }
  header {
    border-bottom: 0.5px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--bg);
  }
  header .logo {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.3px;
  }
  header .badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
    background: var(--green-bg);
    color: var(--green-text);
    border: 0.5px solid var(--green);
  }
  header .running {
    margin-left: auto;
    font-size: 12px;
    color: var(--green);
    display: flex;
    align-items: center;
    gap: 5px;
  }
  header .running::before {
    content: '';
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 0.5px solid var(--border);
    padding: 0 24px;
    background: var(--bg);
  }
  .tab {
    padding: 10px 16px;
    font-size: 13px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    color: var(--text2);
    transition: all 0.15s;
    user-select: none;
  }
  .tab.active {
    color: var(--text);
    border-bottom-color: var(--text);
    font-weight: 500;
  }
  .tab:hover:not(.active) { color: var(--text); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .layout {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 0;
    height: calc(100vh - 89px);
  }
  .sidebar {
    border-right: 0.5px solid var(--border);
    padding: 20px;
    overflow-y: auto;
    background: var(--bg2);
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .main {
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text3);
    margin-bottom: 8px;
  }
  select, input[type=text], textarea {
    width: 100%;
    padding: 8px 10px;
    border: 0.5px solid var(--border2);
    border-radius: var(--radius);
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s;
  }
  select:focus, input[type=text]:focus, textarea:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 2px rgba(55,138,221,0.15);
  }
  textarea { resize: vertical; min-height: 60px; font-family: var(--mono); font-size: 12px; }
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: var(--radius);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: 0.5px solid var(--border2);
    background: var(--bg);
    color: var(--text);
    transition: all 0.15s;
    white-space: nowrap;
  }
  .btn:hover { background: var(--bg3); border-color: var(--border2); }
  .btn:active { transform: scale(0.98); }
  .btn.primary {
    background: var(--green);
    color: white;
    border-color: var(--green);
  }
  .btn.primary:hover { opacity: 0.9; }
  .btn.danger {
    background: var(--red-bg);
    color: var(--red-text);
    border-color: var(--red);
  }
  .btn.secondary {
    background: var(--blue-bg);
    color: var(--blue-text);
    border-color: var(--blue);
  }
  .btn-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .card {
    background: var(--bg);
    border: 0.5px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px;
  }
  .card-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text2);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .json-out {
    background: var(--bg3);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text2);
    white-space: pre-wrap;
    word-break: break-all;
    min-height: 80px;
    max-height: 400px;
    overflow-y: auto;
    line-height: 1.5;
  }
  .json-out.empty { color: var(--text3); font-style: italic; font-family: inherit; font-size: 13px; }
  .metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }
  .metric {
    background: var(--bg2);
    border-radius: var(--radius);
    padding: 10px 12px;
    text-align: center;
  }
  .metric-val {
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
  }
  .metric-lbl {
    font-size: 11px;
    color: var(--text3);
    margin-top: 2px;
  }
  .event-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 160px;
    overflow-y: auto;
  }
  .event-item {
    font-size: 12px;
    color: var(--text2);
    padding: 4px 8px;
    border-left: 2px solid var(--border2);
    background: var(--bg2);
    border-radius: 0 4px 4px 0;
  }
  .tag {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
  }
  .tag-easy { background: var(--green-bg); color: var(--green-text); }
  .tag-medium { background: var(--amber-bg); color: var(--amber-text); }
  .tag-hard { background: var(--red-bg); color: var(--red-text); }
  .task-card {
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
    cursor: pointer;
    transition: all 0.15s;
    background: var(--bg);
  }
  .task-card:hover { border-color: var(--border2); background: var(--bg2); }
  .task-card.selected { border-color: var(--blue); background: var(--blue-bg); }
  .task-card-name { font-weight: 500; font-size: 13px; margin-bottom: 4px; }
  .task-card-desc { font-size: 12px; color: var(--text2); }
  .warn-item {
    font-size: 12px;
    color: var(--amber-text);
    background: var(--amber-bg);
    padding: 4px 10px;
    border-radius: 4px;
    border-left: 2px solid var(--amber);
  }
  .status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text3);
    padding: 6px 10px;
    background: var(--bg2);
    border-radius: var(--radius);
    border: 0.5px solid var(--border);
  }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text3); flex-shrink: 0; }
  .status-dot.ok { background: var(--green); }
  .status-dot.err { background: var(--red); }
  .status-dot.loading { background: var(--amber); animation: pulse 1s infinite; }
  .divider { height: 0.5px; background: var(--border); }
  .quick-start {
    background: var(--bg3);
    border-radius: var(--radius);
    padding: 14px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text2);
    white-space: pre;
    overflow-x: auto;
    line-height: 1.6;
  }
  .qs-section { margin-bottom: 16px; }
  .qs-section:last-child { margin-bottom: 0; }
  .qs-title { font-family: inherit; font-size: 12px; font-weight: 600; color: var(--text2); margin-bottom: 6px; }
  .copy-btn {
    float: right;
    font-family: inherit;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    border: 0.5px solid var(--border2);
    background: var(--bg);
    color: var(--text2);
    cursor: pointer;
  }
  .copy-btn:hover { background: var(--bg3); }
  .resource-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .resource-item {
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 6px;
    border: 0.5px solid var(--border);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .resource-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .r-available { background: var(--green); }
  .r-assigned { background: var(--amber); }
  @media (max-width: 700px) {
    .layout { grid-template-columns: 1fr; height: auto; }
    .sidebar { border-right: none; border-bottom: 0.5px solid var(--border); }
  }
</style>
</head>
<body>

<header>
  <span style="font-size:18px;">🚨</span>
  <span class="logo">ICDE OpenEnv</span>
  <span class="badge">OpenEnv compliant</span>
  <span class="running">Running</span>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab('playground')">Playground</div>
  <div class="tab" onclick="switchTab('quickstart')">Quick Start</div>
  <div class="tab" onclick="switchTab('readme')">README</div>
</div>

<div id="tab-playground" class="tab-content active">
  <div class="layout">
    <div class="sidebar">
      <div>
        <div class="section-label">Task</div>
        <div style="display:flex;flex-direction:column;gap:6px;" id="task-list">
          <div class="task-card selected" data-task="task1_resource" onclick="selectTask(this)">
            <div class="task-card-name">Resource Allocation <span class="tag tag-easy">easy</span></div>
            <div class="task-card-desc">Warehouse fire · 3 zones · 4 resources</div>
          </div>
          <div class="task-card" data-task="task2_multiagency" onclick="selectTask(this)">
            <div class="task-card-name">Multi-Agency Conflict <span class="tag tag-medium">medium</span></div>
            <div class="task-card-desc">Hospital MCI · 4 agencies · contradictory data</div>
          </div>
          <div class="task-card" data-task="task3_cascade" onclick="selectTask(this)">
            <div class="task-card-name">Cascading Crisis <span class="tag tag-hard">hard</span></div>
            <div class="task-card-desc">Earthquake + gas rupture + power failure</div>
          </div>
        </div>
      </div>

      <div class="divider"></div>

      <div>
        <div class="section-label">Action Builder</div>
        <div style="display:flex;flex-direction:column;gap:8px;">
          <div>
            <label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px;">Command</label>
            <select id="cmd">
              <option value="dispatch">dispatch</option>
              <option value="recall">recall</option>
              <option value="establish_command">establish_command</option>
              <option value="flag_conflict">flag_conflict</option>
              <option value="escalate">escalate</option>
              <option value="stand_down">stand_down</option>
              <option value="request_mutual_aid">request_mutual_aid</option>
              <option value="issue_directive">issue_directive</option>
            </select>
          </div>
          <div>
            <label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px;">Resource type</label>
            <select id="resource">
              <option value="">— none —</option>
              <option value="engine">engine</option>
              <option value="hazmat">hazmat</option>
              <option value="medical">medical</option>
              <option value="police">police</option>
              <option value="rescue">rescue</option>
              <option value="power">power</option>
            </select>
          </div>
          <div>
            <label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px;">Target zone</label>
            <input type="text" id="zone" placeholder="e.g. zone_a, site_hospital"/>
          </div>
          <div>
            <label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px;">Priority</label>
            <select id="priority">
              <option value="critical">critical</option>
              <option value="high" selected>high</option>
              <option value="medium">medium</option>
              <option value="low">low</option>
            </select>
          </div>
          <div>
            <label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px;">Directive (optional)</label>
            <textarea id="directive" placeholder="Free-text order to field units..."></textarea>
          </div>
          <div>
            <label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px;">Flag reports (comma-separated IDs)</label>
            <input type="text" id="flags" placeholder="e.g. report_1, report_3"/>
          </div>
        </div>
      </div>

      <div class="divider"></div>

      <div class="btn-row">
        <button class="btn primary" onclick="doStep()" style="flex:1;">▶ Step</button>
        <button class="btn" onclick="doReset()" style="flex:1;">↺ Reset</button>
        <button class="btn secondary" onclick="doState()">State</button>
        <button class="btn" onclick="doGrade()">Grade</button>
      </div>

      <div id="status-bar" class="status-bar">
        <div class="status-dot" id="status-dot"></div>
        <span id="status-msg">Click Reset to start a new episode.</span>
      </div>
    </div>

    <div class="main">
      <div class="metric-row" id="metrics" style="display:none;">
        <div class="metric">
          <div class="metric-val" id="m-step">—</div>
          <div class="metric-lbl">Step</div>
        </div>
        <div class="metric">
          <div class="metric-val" id="m-reward">—</div>
          <div class="metric-lbl">Last Reward</div>
        </div>
        <div class="metric">
          <div class="metric-val" id="m-done">—</div>
          <div class="metric-lbl">Done</div>
        </div>
      </div>

      <div id="civ-card" class="card" style="display:none;">
        <div class="card-title">Civilian Status</div>
        <div class="metric-row">
          <div class="metric">
            <div class="metric-val" style="color:var(--green);" id="c-safe">—</div>
            <div class="metric-lbl">Safe</div>
          </div>
          <div class="metric">
            <div class="metric-val" style="color:var(--amber);" id="c-risk">—</div>
            <div class="metric-lbl">At Risk</div>
          </div>
          <div class="metric">
            <div class="metric-val" style="color:var(--red);" id="c-cas">—</div>
            <div class="metric-lbl">Casualties</div>
          </div>
        </div>
      </div>

      <div id="warnings-card" class="card" style="display:none;">
        <div class="card-title">⚠ Active Warnings</div>
        <div id="warnings-list" style="display:flex;flex-direction:column;gap:4px;"></div>
      </div>

      <div id="events-card" class="card" style="display:none;">
        <div class="card-title">Recent Events</div>
        <div class="event-list" id="events-list"></div>
      </div>

      <div id="resources-card" class="card" style="display:none;">
        <div class="card-title">Resources</div>
        <div class="resource-grid" id="resources-list"></div>
      </div>

      <div class="card">
        <div class="card-title" style="display:flex;align-items:center;justify-content:space-between;">
          Raw JSON Response
          <button class="copy-btn" onclick="copyJson()">Copy</button>
        </div>
        <div class="json-out empty" id="json-out">Click Reset to start a new episode, then use Step to send actions.</div>
      </div>
    </div>
  </div>
</div>

<div id="tab-quickstart" class="tab-content" style="padding:24px;max-width:780px;">
  <h2 style="font-size:18px;font-weight:600;margin-bottom:4px;">Connect to this environment</h2>
  <p style="color:var(--text2);font-size:13px;margin-bottom:20px;">Use the OpenEnv HTTP API directly from Python or any HTTP client.</p>

  <div class="qs-section">
    <div class="qs-title">Connect from Python using OpenEnv SDK</div>
    <div style="position:relative;">
      <button class="copy-btn" onclick="copyCode('py-sdk')">Copy</button>
      <div class="quick-start" id="py-sdk">from openenv import OpenEnvClient

env = OpenEnvClient(base_url="https://Hemakshiy-icde-openenv.hf.space")

obs = env.reset(task_id="task1_resource")
result = env.step(action={
    "command": "dispatch",
    "resource_type": "hazmat",
    "target_zone": "zone_b",
    "priority": "critical"
})
print(result)</div>
    </div>
  </div>

  <div class="qs-section">
    <div class="qs-title">Or connect directly with requests</div>
    <div style="position:relative;">
      <button class="copy-btn" onclick="copyCode('py-raw')">Copy</button>
      <div class="quick-start" id="py-raw">import requests

BASE = "https://Hemakshiy-icde-openenv.hf.space"

# Reset
obs = requests.post(f"{BASE}/reset", json={"task_id": "task1_resource"}).json()

# Step
result = requests.post(f"{BASE}/step", json={
    "task_id": "task1_resource",
    "action": {
        "command": "dispatch",
        "resource_type": "hazmat",
        "target_zone": "zone_b",
        "priority": "critical"
    }
}).json()

# Grade
score = requests.get(f"{BASE}/grade", params={"task_id": "task1_resource"}).json()
print(score)</div>
    </div>
  </div>

  <div class="qs-section">
    <div class="qs-title">Or use curl</div>
    <div style="position:relative;">
      <button class="copy-btn" onclick="copyCode('curl-code')">Copy</button>
      <div class="quick-start" id="curl-code">curl -X POST https://Hemakshiy-icde-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_resource"}'

curl -X POST https://Hemakshiy-icde-openenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1_resource","action":{"command":"dispatch","resource_type":"hazmat","target_zone":"zone_b","priority":"critical"}}'</div>
    </div>
  </div>
</div>

<div id="tab-readme" class="tab-content" style="padding:24px;max-width:760px;">
  <h2 style="font-size:18px;font-weight:600;margin-bottom:8px;">🚨 Incident Command Decision Environment</h2>
  <p style="color:var(--text2);margin-bottom:16px;font-size:13px;">An OpenEnv-compliant benchmark where an AI agent plays Incident Commander managing real-world emergencies using the ICS framework used by FEMA, hospitals, fire departments, and militaries worldwide.</p>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:20px;">
    <div class="metric"><div class="metric-val">3</div><div class="metric-lbl">Tasks</div></div>
    <div class="metric"><div class="metric-val">[-1,1]</div><div class="metric-lbl">Reward Range</div></div>
    <div class="metric"><div class="metric-val">0.60</div><div class="metric-lbl">Best Baseline</div></div>
  </div>

  <div style="display:flex;flex-direction:column;gap:10px;">
    <div class="card">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-weight:500;font-size:13px;">Task 1 — Resource Allocation</span>
        <span class="tag tag-easy">easy</span>
        <span style="margin-left:auto;font-size:12px;color:var(--text3);">15 steps · baseline 0.60</span>
      </div>
      <div style="font-size:12px;color:var(--text2);">Warehouse fire with 3 active zones, 4 limited resources, and deliberately contradictory field reports. Agent must prioritize life safety and flag unreliable sources.</div>
    </div>
    <div class="card">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-weight:500;font-size:13px;">Task 2 — Multi-Agency Conflict</span>
        <span class="tag tag-medium">medium</span>
        <span style="margin-left:auto;font-size:12px;color:var(--text3);">20 steps · baseline 0.45</span>
      </div>
      <div style="font-size:12px;color:var(--text2);">Hospital mass casualty event. Fire, EMS, Police, and Hospital give contradictory casualty counts and safety assessments. Wrong calls trigger cascade deaths.</div>
    </div>
    <div class="card">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-weight:500;font-size:13px;">Task 3 — Cascading Crisis</span>
        <span class="tag tag-hard">hard</span>
        <span style="margin-left:auto;font-size:12px;color:var(--text3);">30 steps · baseline 0.30</span>
      </div>
      <div style="font-size:12px;color:var(--text2);">Simultaneous earthquake, gas rupture, and hospital power failure. Early decisions cause irreversible downstream consequences. New complications inject every 5 steps.</div>
    </div>
  </div>
</div>

<script>
const BASE = window.location.origin;
let currentTask = "task1_resource";

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  const tabs = ['playground','quickstart','readme'];
  document.querySelectorAll('.tab')[tabs.indexOf(name)].classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

function selectTask(el) {
  document.querySelectorAll('.task-card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  currentTask = el.dataset.task;
  setStatus('idle', 'Task changed. Click Reset to start.');
}

function setStatus(state, msg) {
  const dot = document.getElementById('status-dot');
  dot.className = 'status-dot' + (state==='ok'?' ok':state==='err'?' err':state==='loading'?' loading':'');
  document.getElementById('status-msg').textContent = msg;
}

function showJson(data) {
  const el = document.getElementById('json-out');
  el.classList.remove('empty');
  el.textContent = JSON.stringify(data, null, 2);
}

function updateUI(obs, reward, done) {
  if (!obs) return;
  document.getElementById('metrics').style.display = 'grid';
  document.getElementById('m-step').textContent = obs.step ?? '—';
  document.getElementById('m-reward').textContent = reward !== undefined ? reward.toFixed(3) : '—';
  document.getElementById('m-done').textContent = done ? 'Yes' : 'No';

  if (obs.civilian_status) {
    document.getElementById('civ-card').style.display = 'block';
    document.getElementById('c-safe').textContent = obs.civilian_status.safe ?? 0;
    document.getElementById('c-risk').textContent = obs.civilian_status.at_risk ?? 0;
    document.getElementById('c-cas').textContent = obs.civilian_status.casualties ?? 0;
  }

  if (obs.warnings && obs.warnings.length > 0) {
    document.getElementById('warnings-card').style.display = 'block';
    document.getElementById('warnings-list').innerHTML = obs.warnings.map(w =>
      `<div class="warn-item">${w}</div>`).join('');
  } else {
    document.getElementById('warnings-card').style.display = 'none';
  }

  if (obs.recent_events && obs.recent_events.length > 0) {
    document.getElementById('events-card').style.display = 'block';
    document.getElementById('events-list').innerHTML = obs.recent_events.map(e =>
      `<div class="event-item">${e}</div>`).join('');
  }

  const allRes = [...(obs.available_resources||[]), ...(obs.assigned_resources||[])];
  if (allRes.length > 0) {
    document.getElementById('resources-card').style.display = 'block';
    document.getElementById('resources-list').innerHTML = allRes.map(r =>
      `<div class="resource-item">
        <div class="resource-dot ${r.available ? 'r-available' : 'r-assigned'}"></div>
        <span style="font-size:12px;color:var(--text);">${r.resource_type}</span>
        <span style="font-size:11px;color:var(--text3);margin-left:auto;">${r.assigned_zone || 'available'}</span>
      </div>`).join('');
  }
}

async function doReset() {
  setStatus('loading', 'Resetting environment…');
  try {
    const r = await fetch(`${BASE}/reset`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({task_id: currentTask, seed: 42})
    });
    const data = await r.json();
    showJson(data);
    updateUI(data, undefined, false);
    setStatus('ok', `Episode started — step ${data.step || 0} of ${(data.time_remaining||0)+(data.step||0)}`);
  } catch(e) {
    setStatus('err', 'Reset failed: ' + e.message);
  }
}

async function doStep() {
  const flags = document.getElementById('flags').value
    .split(',').map(s=>s.trim()).filter(Boolean);
  const resource = document.getElementById('resource').value || undefined;
  const zone = document.getElementById('zone').value.trim() || undefined;
  const directive = document.getElementById('directive').value.trim() || undefined;

  const action = {
    command: document.getElementById('cmd').value,
    resource_type: resource,
    target_zone: zone,
    priority: document.getElementById('priority').value,
    directive: directive,
    flags: flags
  };

  setStatus('loading', 'Sending action…');
  try {
    const r = await fetch(`${BASE}/step`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({task_id: currentTask, action})
    });
    const data = await r.json();
    showJson(data);
    if (data.observation) {
      updateUI(data.observation, data.reward, data.done);
      const msg = data.done
        ? `Episode done — reward ${data.reward?.toFixed(3)}`
        : `Step ${data.observation.step} — reward ${data.reward?.toFixed(3)}`;
      setStatus(data.done ? 'err' : 'ok', msg);
    } else {
      setStatus('err', data.detail || 'Error in step');
    }
  } catch(e) {
    setStatus('err', 'Step failed: ' + e.message);
  }
}

async function doState() {
  setStatus('loading', 'Fetching state…');
  try {
    const r = await fetch(`${BASE}/state?task_id=${currentTask}`);
    const data = await r.json();
    showJson(data);
    setStatus('ok', 'State fetched.');
  } catch(e) {
    setStatus('err', 'State failed: ' + e.message);
  }
}

async function doGrade() {
  setStatus('loading', 'Grading episode…');
  try {
    const r = await fetch(`${BASE}/grade?task_id=${currentTask}`);
    const data = await r.json();
    showJson(data);
    const score = data.score !== undefined ? ` — score: ${data.score.toFixed(3)}` : '';
    setStatus('ok', 'Grade computed' + score);
  } catch(e) {
    setStatus('err', 'Grade failed: ' + e.message);
  }
}

function copyJson() {
  const text = document.getElementById('json-out').textContent;
  navigator.clipboard.writeText(text).catch(()=>{});
}

function copyCode(id) {
  const text = document.getElementById(id).textContent;
  navigator.clipboard.writeText(text).catch(()=>{});
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
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
    env = _get_env(task_id)
    if env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return env.state().dict()


@app.get("/grade")
async def grade(task_id: str = Query(default="task1_resource")):
    env = _get_env(task_id)
    if env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    sim = env._sim
    grader_fn = GRADERS.get(task_id)
    if grader_fn is None:
        raise HTTPException(status_code=404, detail=f"No grader for task: {task_id}")
    resource_assignments = {
        rid: r.assigned_zone or "" for rid, r in sim.resources.items()
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
        return grader_fn(**common_kwargs)
    elif task_id == "task2_multiagency":
        command_established = any("establish_command" in a for a in sim.action_history)
        steps_to_cmd = next((i+1 for i,a in enumerate(sim.action_history) if "establish_command" in a), 99)
        return grader_fn(**common_kwargs, command_established=command_established, steps_to_first_command=steps_to_cmd)
    elif task_id == "task3_cascade":
        command_established = any("establish_command" in a for a in sim.action_history)
        triggered_ids = [cid for cid, cond in sim.cascade_conditions.items() if cond.get("triggered", False)]
        return grader_fn(**common_kwargs, command_established=command_established, total_steps=sim.step_num, cascades_triggered_ids=triggered_ids)
    return {"score": 0.0, "error": "Unknown task"}


@app.get("/tasks")
async def list_tasks():
    return {"tasks": [
        {"id": "task1_resource", "name": "Resource Allocation Under Scarcity", "difficulty": "easy", "max_steps": 15},
        {"id": "task2_multiagency", "name": "Multi-Agency Conflict Resolution", "difficulty": "medium", "max_steps": 20},
        {"id": "task3_cascade", "name": "Cascading Multi-Site Crisis", "difficulty": "hard", "max_steps": 30},
    ]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
