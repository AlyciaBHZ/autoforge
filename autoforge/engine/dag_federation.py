"""Federated synchronization for CapabilityDAG and theory graphs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from autoforge.engine.capability_dag import CapabilityDAG
    from autoforge.engine.theoretical_reasoning import TheoreticalReasoningEngine

logger = logging.getLogger(__name__)


@dataclass
class DAGFederationConfig:
    """Runtime settings for community DAG federation."""

    enabled: bool = False
    endpoint: str = ""
    api_key: str = ""
    timeout_seconds: float = 10.0


class DAGFederationClient:
    """HTTP client for two-way sync with a shared knowledge service."""

    def __init__(self, config: DAGFederationConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._config.endpoint)

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._config.api_key:
            headers["authorization"] = f"Bearer {self._config.api_key}"
        return headers

    async def pull_snapshot(self) -> dict[str, Any] | None:
        """Fetch latest federated snapshot from remote service."""
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await client.get(self._config.endpoint, headers=self._headers())
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    logger.warning("[DAG Federation] Invalid snapshot payload type: %s", type(payload))
                    return None
                return payload
        except Exception as e:
            logger.warning("[DAG Federation] Pull failed: %s", e)
            return None

    async def push_snapshot(self, payload: dict[str, Any]) -> bool:
        """Push local merged snapshot to remote service."""
        if not self.enabled:
            return False

        try:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await client.post(
                    self._config.endpoint,
                    headers=self._headers(),
                    content=json.dumps(payload, ensure_ascii=False),
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.warning("[DAG Federation] Push failed: %s", e)
            return False


async def pull_into_local_knowledge(
    *,
    federation: DAGFederationClient,
    capability_dag: CapabilityDAG | None,
    theoretical_reasoning: TheoreticalReasoningEngine | None,
    global_theory_dir: Path,
) -> dict[str, int]:
    """Pull remote snapshot and merge into local DAG + theory graphs."""
    stats = {"dag_nodes": 0, "theories": 0}
    payload = await federation.pull_snapshot()
    if not payload:
        return stats

    dag_data = payload.get("capability_dag")
    if capability_dag is not None and isinstance(dag_data, dict):
        before = capability_dag.size
        capability_dag.load_dict(dag_data)
        stats["dag_nodes"] = max(capability_dag.size - before, 0)

    theories_data = payload.get("theories")
    if theoretical_reasoning is not None and isinstance(theories_data, dict):
        global_theory_dir.mkdir(parents=True, exist_ok=True)
        for title, theory_data in theories_data.items():
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(title))[:50] or "theory"
            theory_path = global_theory_dir / safe_name
            theory_path.mkdir(parents=True, exist_ok=True)
            (theory_path / "theory_graph.json").write_text(
                json.dumps(theory_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            stats["theories"] += 1
        theoretical_reasoning.load_all(global_theory_dir)

    return stats


def build_snapshot_payload(
    *,
    capability_dag: CapabilityDAG | None,
    theoretical_reasoning: TheoreticalReasoningEngine | None,
) -> dict[str, Any]:
    """Create a federated payload with DAG + theory graph knowledge."""
    payload: dict[str, Any] = {"version": 1}
    if capability_dag is not None:
        payload["capability_dag"] = capability_dag.to_dict()

    if theoretical_reasoning is not None:
        theories: dict[str, dict[str, Any]] = {}
        for title, theory in theoretical_reasoning._theories.items():
            theories[title] = {
                "title": theory.title,
                "source": theory.source,
                "nodes": {nid: node.to_dict() for nid, node in theory._nodes.items()},
                "relations": [rel.to_dict() for rel in theory._relations],
            }
        payload["theories"] = theories

    return payload
