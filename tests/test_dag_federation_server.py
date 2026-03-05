from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _wait_ready(url: str, timeout: float = 5.0, headers: dict[str, str] | None = None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET", headers=headers or {})
            with urllib.request.urlopen(req, timeout=0.5):
                return
        except urllib.error.HTTPError:
            # Server responded (e.g. 401), so it is already up.
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("server not ready")


def test_snapshot_server_get_post(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "scripts/dag_federation_server.py",
            "--host",
            "127.0.0.1",
            "--port",
            "8788",
            "--snapshot-file",
            str(snapshot),
            "--token",
            "abc",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_ready("http://127.0.0.1:8788/snapshot", headers={"Authorization": "Bearer abc"})

        req = urllib.request.Request(
            "http://127.0.0.1:8788/snapshot",
            method="GET",
            headers={"Authorization": "Bearer abc"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert "capability_dag" in data

        payload = {"version": 1, "capability_dag": {"nodes": {}, "edges": []}, "theories": {}}
        post = urllib.request.Request(
            "http://127.0.0.1:8788/snapshot",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer abc",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(post, timeout=2) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        assert result["ok"] is True

        stored = json.loads(snapshot.read_text(encoding="utf-8"))
        assert stored["version"] == 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_snapshot_server_rejects_bad_token(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "scripts/dag_federation_server.py",
            "--host",
            "127.0.0.1",
            "--port",
            "8789",
            "--snapshot-file",
            str(snapshot),
            "--token",
            "secret",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_ready("http://127.0.0.1:8789/snapshot")
        req = urllib.request.Request("http://127.0.0.1:8789/snapshot", method="GET")
        try:
            urllib.request.urlopen(req, timeout=2)
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        proc.terminate()
        proc.wait(timeout=5)
