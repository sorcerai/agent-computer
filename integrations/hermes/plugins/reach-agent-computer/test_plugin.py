"""Unit tests for reach-agent-computer thin Hermes plugin."""

import http.server
import json
import os
import subprocess
import threading
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

from pathlib import Path
import sys

PLUGIN_DIR = Path(__file__).parent.resolve()
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from __init__ import (  # noqa: E402
    get_state,
    reach_drive,
    reach_lease_screen,
    reach_release_screen,
    reach_status,
    register,
    reset_state,
)


class FakeReachServer(http.server.HTTPServer):
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        super().__init__((host, port), FakeReachHandler)
        self.screens: List[Dict[str, Any]] = [
            {
                "id": 0,
                "owner": None,
                "takeover_pending": False,
                "takeover_url": None,
                "leased_at": None,
                "novnc_url": "http://127.0.0.1:6080/vnc.html",
            },
            {
                "id": 1,
                "owner": None,
                "takeover_pending": False,
                "takeover_url": None,
                "leased_at": None,
                "novnc_url": "http://127.0.0.1:6081/vnc.html",
            },
        ]
        self.requests_log: List[Dict[str, Any]] = []


class FakeReachHandler(http.server.BaseHTTPRequestHandler):
    server: FakeReachServer

    def do_GET(self) -> None:
        self.server.requests_log.append({"method": "GET", "path": self.path})
        if self.path == "/agent/screens":
            self._respond_json(200, self.server.screens)
        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        self.server.requests_log.append(
            {"method": "POST", "path": self.path, "body": body}
        )

        if self.path.startswith("/agent/screens/") and self.path.endswith("/lease"):
            screen_id = int(self.path.split("/")[3])
            owner = body.get("owner", "default")
            screen = next(
                (s for s in self.server.screens if s["id"] == screen_id), None
            )
            if not screen:
                self._respond_json(404, {"error": "screen not found"})
                return
            if screen["owner"] is not None and screen["owner"] != owner:
                self._respond_json(409, {"error": "occupied"})
                return
            screen["owner"] = owner
            self._respond_json(200, {"status": "ok", "id": screen_id, "owner": owner})
        else:
            self._respond_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        length = int(self.headers.get("content-length", 0))
        body = (
            json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if length > 0
            else {}
        )
        self.server.requests_log.append(
            {"method": "DELETE", "path": self.path, "body": body}
        )

        if self.path.startswith("/agent/screens/") and self.path.endswith("/lease"):
            screen_id = int(self.path.split("/")[3])
            owner = body.get("owner")
            screen = next(
                (s for s in self.server.screens if s["id"] == screen_id), None
            )
            if not screen:
                self._respond_json(404, {"error": "screen not found"})
                return
            if owner and screen["owner"] != owner:
                self._respond_json(400, {"error": "not owner"})
                return
            screen["owner"] = None
            self._respond_json(200, {"status": "ok", "id": screen_id, "released": True})
        else:
            self._respond_json(404, {"error": "not found"})

    def _respond_json(self, status: int, data: Any) -> None:
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        pass


class FakeHermesContext:
    def __init__(self) -> None:
        self.tools: Dict[str, Any] = {}

    def register_tool(self, name: str, fn: Any, description: str = "") -> None:
        self.tools[name] = fn


class ThinHermesShimTests(unittest.TestCase):
    server: FakeReachServer
    server_thread: threading.Thread
    api_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = FakeReachServer()
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()
        port = cls.server.server_address[1]
        cls.api_url = f"http://127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        reset_state()
        os.environ["REACH_AGENT_URL"] = self.api_url
        os.environ["HERMES_PROFILE"] = "tester"
        for s in self.server.screens:
            s["owner"] = None
        self.server.requests_log.clear()

    def tearDown(self) -> None:
        reset_state()

    def test_reach_status_queries_screens(self) -> None:
        res = reach_status()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["screens"]), 2)

    def test_reach_status_single_screen(self) -> None:
        res0 = reach_status(screen=0)
        self.assertEqual(res0["status"], "ok")
        self.assertEqual(res0["screen"]["id"], 0)

        res_none = reach_status(screen=99)
        self.assertEqual(res_none["status"], "not_found")

    def test_reach_lease_screen_auto_and_release(self) -> None:
        # Auto-lease screen
        lease = reach_lease_screen()
        self.assertEqual(lease["status"], "ok")
        self.assertEqual(lease["screen"], 0)
        self.assertEqual(get_state()["screen"], 0)
        self.assertEqual(self.server.screens[0]["owner"], "tester")

        # Release screen
        rel = reach_release_screen()
        self.assertEqual(rel["status"], "ok")
        self.assertEqual(rel["screen"], 0)
        self.assertIsNone(get_state()["screen"])
        self.assertIsNone(self.server.screens[0]["owner"])

    def test_reach_lease_explicit_screen(self) -> None:
        lease = reach_lease_screen(screen=1, owner="custom_user")
        self.assertEqual(lease["status"], "ok")
        self.assertEqual(lease["screen"], 1)
        self.assertEqual(lease["owner"], "custom_user")
        self.assertEqual(self.server.screens[1]["owner"], "custom_user")

    def test_reach_lease_screen_exhausted(self) -> None:
        for s in self.server.screens:
            s["owner"] = "someone_else"

        res = reach_lease_screen()
        self.assertEqual(res["status"], "exhausted")

    def test_reach_lease_screen_conflict(self) -> None:
        self.server.screens[0]["owner"] = "other_user"
        res = reach_lease_screen(screen=0, owner="me")
        self.assertEqual(res["status"], "conflict")

    @patch("subprocess.run")
    def test_reach_drive_delegates_to_cli(self, mock_run: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "success": True,
            "status": "completed",
            "goal": "Test goal",
            "final_description": "Goal achieved",
            "steps": [],
        })
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        result = reach_drive(goal="Test goal", screen=0, max_steps=10)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["success"])

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("drive", cmd)
        self.assertIn("--goal", cmd)
        self.assertIn("Test goal", cmd)
        self.assertIn("--screen", cmd)
        self.assertIn("0", cmd)
        self.assertIn("--max-steps", cmd)
        self.assertIn("10", cmd)

    def test_register_adds_tools(self) -> None:
        ctx = FakeHermesContext()
        register(ctx)
        self.assertIn("reach_lease_screen", ctx.tools)
        self.assertIn("reach_release_screen", ctx.tools)
        self.assertIn("reach_status", ctx.tools)
        self.assertIn("reach_drive", ctx.tools)


if __name__ == "__main__":
    unittest.main()
