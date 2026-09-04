# Agent Computer 🖥️🤖

**Agent Computer** is an open-source, local-first Agent Computer Operating System & Fleet Runtime (inspired by the xAI Grok Bot Workstation model).

It provides persistent, multi-screen virtual workstations for autonomous AI agents (such as Hermes or custom orchestrators), coupled with an instant MicroVM hypervisor layer, sub-second multimodal visual driving (Gemini 3.8 Flash), out-of-band secret management, and human-in-the-loop takeover.

---

## Architecture Overview

```
                                  HOST CONTROL PLANE
                     ┌─────────────────────────────────────────┐
                     │   Main Orchestrator / Hermes Fleet      │
                     │   Host Vault (~/.reach/vault/secrets)   │
                     └────────────────────┬────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     │  Agent Computer Hypervisor & Supervisor │
                     └────────────────────┬────────────────────┘
                                          │
       ┌──────────────────────────────────┴──────────────────────────────────┐
       ▼                                                                     ▼
[ MicroVM 1: Research Agent ]                                         [ MicroVM 2: Ops Agent ]
 • Display :99 (Xvfb)                                                  • Display :100 (Xvfb)
 • noVNC :6080 (Live View)                                             • noVNC :6081 (Live View)
 • Browser: Chromium Headed                                            • Browser: Chromium Headed
 • Memory: 53 MB base stack                                            • Memory: 53 MB base stack
 • Reset: Instant CoW fork (<250ms)                                    • Reset: Instant CoW fork (<250ms)
       │                                                                     │
       └────────────────► /workspace/.reach/state.json ◄─────────────────────┘
                         (Shared Cookie & Session State)
```

---

## Core Capabilities

### 1. Instant MicroVM Hypervisor (`scripts/microvm.sh`)
* **APFS Copy-on-Write Snapshots**: Golden template (`reach-golden`) boots and snapshots once; instances fork in **~160–250 ms** taking only ~170 kB of host storage delta.
* **Instant Nuke & Clean Slate**: Obliterate dirty or corrupted environments in **~200 ms**.
* **3-Second Full Reset**: Completely wipes the container/VM, clones a pristine template, boots, and completes health checks in **< 3.0 seconds**.

### 2. Multi-Screen Virtual Display & Live View
* **Multiplexed Virtual Screens**: Each agent leases a dedicated virtual screen (`DISPLAY=:99+i`) with independent windowing (Openbox), virtual framebuffers (Xvfb), and RFB sockets (x11vnc).
* **Live noVNC Viewport**: Open `http://localhost:6080/vnc.html` to watch the agent think, type, and click live with responsive remote scaling.
* **noVNC Canvas Fix**: Automatically paints `#1e1e2e` (Catppuccin Mocha) and serves an interactive dashboard (`assets/home.html`) so the viewport is never pitch-black on idle.

### 3. Sub-Second CUA Vision Loop (`scripts/reach_drive.py`)
* **Powered by Gemini 3.8 Flash via `agy`**: Multimodal visual decision loop running in **~1.0s turns** with state-of-the-art coordinate resolution and zero API token cost via local Antigravity sessions.
* **Gauntlet Safety Fences**: Untrusted DOM text and screenshots are strictly delimited as unprivileged data, preventing prompt injection attacks from adversarial web pages.

### 4. Out-of-Band Secret Broker & TOTP (`scripts/reach_vault.py`)
* **Zero Secrets in the Sandbox**: Passwords, API tokens, and 2FA secrets live on the host in `~/.reach/vault/secrets.json` with strict `0600` permissions and optional PBKDF2 encryption.
* **Built-in RFC 6238 TOTP**: Pure standard library implementation generating 6-digit codes on the fly.
* **Direct Synthetic Typing**: Injects credentials directly into input fields via synthetic key events—secrets are never written to the guest filesystem.

### 5. Dangerous Mutation Approval Gates
* **Policy Interceptor**: Catches high-stakes actions matching destructive patterns (`delete`, `remove`, `pay`, `purchase`, `order`, `drop table`, `transfer`).
* **Non-Interactive Circuit Breaker**: Halts execution, emits `status: approval_required`, and pauses until confirmed by the user or an authorized callback.

### 6. Visual Diff Audit Reel & HTML Mission Reports
* **Automatic Timeline Diffing**: Captures side-by-side Before/After screenshots for every action.
* **Target Crosshairs**: Renders animated crosshairs directly on the clicked coordinates in the audit viewer.
* **Interactive HTML Dashboard**: Self-contained `report.html` detailing timestamps, action badges, mutation warnings, and metrics.

### 7. Multi-Bot Kanban Coordination
* **Task Chains**: Allows specialized agents (e.g. `piper` collects $\to$ `otto` files) to pass structured tasks via Hermes Kanban.
* **Stable Identity Anchors**: Retains browser device fingerprints and persists session cookies via Playwright `storage_state.json` so microVM resets do not trigger anti-bot session churn.

---

## Quick Start

### 1. Build and Test
```bash
# Build Rust binaries
cargo build --workspace

# Run full Rust test suite
cargo test --workspace

# Run Python test suite
pytest tests/
```

### 2. Spawn a MicroVM
```bash
# Spawn a fresh microVM in ~250ms
./scripts/microvm.sh spawn bot-01

# Watch live in your browser
open $(./scripts/microvm.sh vnc bot-01)

# Check health
./scripts/microvm.sh healthcheck bot-01

# Nuke when done (zero residue)
./scripts/microvm.sh nuke bot-01
```

### 3. Run an Autonomous CUA Mission
```bash
python3 scripts/reach_drive.py \
  --goal "Open https://news.ycombinator.com and report the top story title" \
  --screen 0 \
  -v
```

### 4. Manage Credentials & 2FA
```bash
# Store a credential on host
python3 scripts/reach_vault.py set github.com --user myuser --pass mypass --totp JBSWY3DPEHPK3PXP

# Generate current 2FA code
python3 scripts/reach_vault.py totp github.com

# Inject into active screen without writing to disk
python3 scripts/reach_vault.py inject 0 github.com
```

---

## Resource Footprint per Screen

| Component | Resident Memory (RSS) |
| :--- | :--- |
| `reach-supervisor` (Rust) | **1.2 MB** |
| `Xvfb` (Virtual Display) | **15.0 MB** |
| `openbox` (Window Manager) | **3.5 MB** |
| `x11vnc` (VNC Server) | **14.5 MB** |
| `websockify` (noVNC Bridge) | **19.0 MB** |
| **Total Base Stack** | **~53.2 MB per screen** |
| **Active Chromium** | **120 – 250 MB** |

---

## License

MIT License.
