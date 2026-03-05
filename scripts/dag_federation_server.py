#!/usr/bin/env python3
"""Minimal DAG federation snapshot server.

Provides a tiny HTTP service compatible with AutoForge DAG federation:
- GET  /snapshot  -> returns JSON snapshot
- POST /snapshot  -> replaces JSON snapshot

Auth:
- If DAG_FEDERATION_TOKEN is set, requires header:
  Authorization: Bearer <token>

Storage:
- JSON file on disk (default: .autoforge/federation/snapshot.json)

This is intentionally dependency-free (stdlib only) so teams can deploy fast.
"""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _default_snapshot() -> dict:
    return {
        "version": 1,
        "capability_dag": {
            "version": 1,
            "growth_round": 0,
            "stats": {
                "nodes_added": 0,
                "nodes_pruned": 0,
                "queries": 0,
                "growth_rounds": 0,
                "merges": 0,
            },
            "nodes": {},
            "edges": [],
        },
        "theories": {},
    }


class SnapshotStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self.write(_default_snapshot())

    def read(self) -> dict:
        with self._lock:
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return _default_snapshot()

    def write(self, payload: dict) -> None:
        with self._lock:
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


class SnapshotHandler(BaseHTTPRequestHandler):
    store: SnapshotStore
    required_token: str = ""

    def _check_auth(self) -> bool:
        if not self.required_token:
            return True
        auth_header = self.headers.get("Authorization", "")
        return auth_header == f"Bearer {self.required_token}"

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path != "/snapshot":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._check_auth():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        self._send_json(HTTPStatus.OK, self.store.read())

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path != "/snapshot":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._check_auth():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be object")
        except Exception as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json", "detail": str(e)})
            return

        self.store.write(payload)
        self._send_json(HTTPStatus.OK, {"ok": True})

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a DAG federation snapshot server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--snapshot-file",
        type=Path,
        default=Path(".autoforge/federation/snapshot.json"),
    )
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    SnapshotHandler.store = SnapshotStore(args.snapshot_file)
    SnapshotHandler.required_token = args.token

    server = ThreadingHTTPServer((args.host, args.port), SnapshotHandler)
    print(f"[dag-federation] listening on http://{args.host}:{args.port}/snapshot")
    if args.token:
        print("[dag-federation] bearer auth enabled")
    print(f"[dag-federation] snapshot file: {args.snapshot_file}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
