from __future__ import annotations

import asyncio
from pathlib import Path

from autoforge.engine.capability_dag import CapabilityDAG, Domain
from autoforge.engine.config import ForgeConfig
from autoforge.engine.dag_federation import (
    DAGFederationClient,
    DAGFederationConfig,
    build_snapshot_payload,
    pull_into_local_knowledge,
)
from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptType,
    ScientificDomain,
    TheoryGraph,
    TheoreticalReasoningEngine,
)


class _StubFederation(DAGFederationClient):
    def __init__(self, payload: dict):
        super().__init__(DAGFederationConfig(enabled=True, endpoint="https://example.test"))
        self._payload = payload

    async def pull_snapshot(self):
        return self._payload


def test_capability_dag_roundtrip_dict() -> None:
    dag = CapabilityDAG()
    dag.add("use lockfile", Domain.WORKFLOW, summary="lock strategy", confidence=0.7)

    cloned = CapabilityDAG()
    ok = cloned.load_dict(dag.to_dict())

    assert ok is True
    assert cloned.size == 1


def test_build_snapshot_payload_contains_theory_graphs() -> None:
    dag = CapabilityDAG()
    dag.add("error -> fix", Domain.DEBUGGING, summary="pattern", confidence=0.6)

    theory = TheoryGraph(title="T1", source="unit-test")
    theory.add_concept(
        ConceptNode(
            id="c1",
            concept_type=ConceptType.THEOREM,
            domain=ScientificDomain.COMPUTER_SCIENCE,
            formal_statement="A -> B",
            informal_statement="implication",
            overall_confidence=0.8,
        )
    )

    engine = TheoreticalReasoningEngine()
    engine._theories[theory.title] = theory

    payload = build_snapshot_payload(capability_dag=dag, theoretical_reasoning=engine)

    assert "capability_dag" in payload
    assert "theories" in payload
    assert "T1" in payload["theories"]


def test_pull_into_local_knowledge_merges_both_graphs(tmp_path: Path) -> None:
    remote = {
        "capability_dag": CapabilityDAG().to_dict(),
        "theories": {
            "Shared Theory": {
                "title": "Shared Theory",
                "source": "remote",
                "nodes": {
                    "n1": {
                        "id": "n1",
                        "concept_type": "theorem",
                        "domain": "computer_science",
                        "formal_statement": "X",
                        "informal_statement": "Y",
                        "intuition": "",
                        "proof_sketch": "",
                        "formal_proof": "",
                        "tags": [],
                        "sub_domain": "",
                        "verification_status": {},
                        "overall_confidence": 0.9,
                        "source_article": "",
                        "source_section": "",
                        "generation_strategy": "",
                        "parent_ids": [],
                        "created_at": 0.0,
                        "metadata": {},
                    }
                },
                "relations": [],
            }
        },
    }
    remote["capability_dag"]["nodes"] = {
        "abcd": {
            "id": "abcd",
            "domain": "workflow",
            "content": "do ci first",
            "summary": "remote node",
            "tags": ["ci"],
            "verification_type": "self_report",
            "confidence": 0.4,
            "verification_details": "",
            "source_project": "",
            "source_user": "",
            "generation_strategy": "",
            "parent_ids": [],
            "usage_count": 0,
            "success_count": 0,
            "last_used": 0.0,
            "created_at": 0.0,
            "updated_at": 0.0,
            "metadata": {},
        }
    }

    dag = CapabilityDAG()
    theory_engine = TheoreticalReasoningEngine()
    stats = asyncio.run(
        pull_into_local_knowledge(
            federation=_StubFederation(remote),
            capability_dag=dag,
            theoretical_reasoning=theory_engine,
            global_theory_dir=tmp_path / "theories",
        )
    )

    assert dag.size == 1
    assert stats["theories"] == 1
    assert "Shared Theory" in theory_engine._theories


def test_federation_env_config(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_DAG_FEDERATION_ENABLED", "true")
    monkeypatch.setenv("FORGE_DAG_FEDERATION_ENDPOINT", "https://dag.example/api/snapshot")
    monkeypatch.setenv("FORGE_DAG_FEDERATION_TIMEOUT_SECONDS", "15")

    cfg = ForgeConfig.from_env()

    assert cfg.dag_federation_enabled is True
    assert cfg.dag_federation_endpoint == "https://dag.example/api/snapshot"
    assert cfg.dag_federation_timeout_seconds == 15.0
