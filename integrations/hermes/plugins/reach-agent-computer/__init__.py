"""Hermes thin Python shim for Reach Agent Computer."""
import json, os, shutil, subprocess, urllib.error, urllib.request
API_DEFAULT = "http://127.0.0.1:4200"
_state = {"screen": None}; get_state = lambda: dict(_state); reset_state = lambda: _state.update(screen=None)
def _req(path, method="GET", body=None, api_url=None):
    base = (api_url or os.environ.get("REACH_AGENT_URL", API_DEFAULT)).rstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{base}{path}", data=data, headers={"content-type": "application/json"} if data else {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r: return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e: return {"status": "error", "code": e.code, "message": e.read().decode(errors="replace")}
    except Exception as e: return {"status": "error", "message": str(e)}
def reach_lease_screen(screen=None, owner=None, api_url=None):
    owner = owner or os.environ.get("HERMES_PROFILE", "default")
    if screen is None:
        screens = _req("/agent/screens", api_url=api_url)
        if not isinstance(screens, list): return {"status": "error", "message": str(screens)}
        free = next((s for s in screens if s.get("owner") in (None, owner)), None)
        if not free: return {"status": "exhausted", "message": "No free screens"}
        screen = free.get("id", 0)
    res = _req(f"/agent/screens/{screen}/lease", "POST", {"owner": owner}, api_url=api_url)
    if isinstance(res, dict) and res.get("status") == "error":
        return {"status": "conflict" if res.get("code") == 409 else "error", "message": res.get("message")}
    _state["screen"] = screen
    return {"status": "ok", "screen": screen, "owner": owner}
def reach_release_screen(screen=None, owner=None, api_url=None):
    target = screen if screen is not None else _state.get("screen", 0)
    _req(f"/agent/screens/{target}/lease", "DELETE", {"owner": owner or os.environ.get("HERMES_PROFILE", "default")}, api_url=api_url)
    if _state.get("screen") == target: _state["screen"] = None
    return {"status": "ok", "screen": target, "released": True}
def reach_status(screen=None, api_url=None):
    screens = _req("/agent/screens", api_url=api_url)
    if not isinstance(screens, list): return {"status": "error", "message": str(screens)}
    if screen is not None:
        found = next((s for s in screens if s.get("id") == screen), None)
        return {"status": "ok", "screen": found} if found else {"status": "not_found", "message": f"Screen {screen} not found."}
    return {"status": "ok", "screens": screens}
def reach_drive(goal, screen=None, max_steps=15, target="agent-computer", model=None, **kw):
    s = screen if screen is not None else (_state.get("screen") or 0)
    bin_name = os.environ.get("REACH_BIN") or shutil.which("agent-computer") or shutil.which("reach") or "agent-computer"
    cmd = [bin_name, "drive", "--goal", goal, "--screen", str(s), "--max-steps", str(max_steps), "--target", target]
    if model: cmd.extend(["--model", model])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try: return json.loads(proc.stdout)
    except Exception: return {"status": "completed" if proc.returncode == 0 else "failed", "stdout": proc.stdout, "stderr": proc.stderr}
def register(ctx):
    for name, fn in [("reach_lease_screen", reach_lease_screen), ("reach_release_screen", reach_release_screen), ("reach_status", reach_status), ("reach_drive", reach_drive)]:
        if hasattr(ctx, "register_tool"): ctx.register_tool(name, fn)
        elif hasattr(ctx, "tools") and isinstance(ctx.tools, dict): ctx.tools[name] = fn
