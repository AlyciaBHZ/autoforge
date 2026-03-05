"""
Structured world model backed by TheoryGraph.

Provides:
- temporal event logging
- filtered concept queries
- cross-session persistence
- lightweight contextual retrieval
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptRelation,
    ConceptType,
    RelationType,
    ScientificDomain,
    TheoryGraph,
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalEvent:
    """Record of a discovery/verification event."""

    event_type: str
    concept_id: str
    round_number: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalEvent:
        return cls(**data)


@dataclass
class QueryFilter:
    """Filter criteria for WorldModel queries."""

    domains: Optional[list[ScientificDomain]] = None
    concept_types: Optional[list[ConceptType]] = None
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    after_round: int = 0
    before_round: int = 999999
    tags: Optional[list[str]] = None
    limit: int = 50


class WorldModel:
    """Persistent, queryable wrapper around TheoryGraph."""

    def __init__(self, graph: Optional[TheoryGraph] = None) -> None:
        self.graph = graph or TheoryGraph()
        self.temporal_log: list[TemporalEvent] = []
        logger.info("WorldModel initialized")

    def record_event(self, event: TemporalEvent) -> None:
        self.temporal_log.append(event)
        logger.debug(
            "Recorded event %s for %s in round %d",
            event.event_type,
            event.concept_id,
            event.round_number,
        )

    async def add_discovery(
        self, node: ConceptNode, round_number: int, strategy: str
    ) -> None:
        """Add concept + timeline event."""
        self.graph.add_concept(node)
        self.record_event(
            TemporalEvent(
                event_type="discovery",
                concept_id=node.id,
                round_number=round_number,
                timestamp=time.time(),
                metadata={
                    "strategy": strategy,
                    "formal_statement": node.formal_statement,
                    "domain": node.domain.value,
                    "confidence": node.overall_confidence,
                },
            )
        )

    def query(self, filter_: QueryFilter) -> list[ConceptNode]:
        """Return concepts matching filter criteria."""
        results: list[ConceptNode] = []

        for node in self.graph.nodes.values():
            if filter_.domains and node.domain not in filter_.domains:
                continue
            if filter_.concept_types and node.concept_type not in filter_.concept_types:
                continue
            if not (filter_.min_confidence <= node.overall_confidence <= filter_.max_confidence):
                continue
            if filter_.tags:
                node_tags = set(node.tags or [])
                if not any(tag in node_tags for tag in filter_.tags):
                    continue
            results.append(node)

        if filter_.after_round > 0 or filter_.before_round < 999999:
            allowed_ids = {
                e.concept_id
                for e in self.temporal_log
                if filter_.after_round <= e.round_number <= filter_.before_round
            }
            results = [node for node in results if node.id in allowed_ids]

        results.sort(key=lambda n: n.overall_confidence, reverse=True)
        return results[: filter_.limit]

    def get_recent_discoveries(self, n: int = 10) -> list[ConceptNode]:
        """Most recent discovered concepts, newest first."""
        recent_events = sorted(
            [event for event in self.temporal_log if event.event_type == "discovery"],
            key=lambda event: event.timestamp,
            reverse=True,
        )[:n]

        concepts: list[ConceptNode] = []
        seen: set[str] = set()
        for event in recent_events:
            if event.concept_id in seen:
                continue
            node = self.graph.get_concept(event.concept_id)
            if node is not None:
                seen.add(event.concept_id)
                concepts.append(node)
        return concepts

    def get_round_summary(self, round_number: int) -> dict[str, Any]:
        """Round-level summary stats."""
        round_events = [e for e in self.temporal_log if e.round_number == round_number]
        discoveries = [e for e in round_events if e.event_type == "discovery"]
        discovery_nodes = [
            self.graph.get_concept(e.concept_id)
            for e in discoveries
        ]
        discovery_nodes = [node for node in discovery_nodes if node is not None]

        avg_confidence = (
            sum(node.overall_confidence for node in discovery_nodes) / len(discovery_nodes)
            if discovery_nodes
            else 0.0
        )
        return {
            "round_number": round_number,
            "event_count": len(round_events),
            "discovery_count": len(discoveries),
            "avg_confidence": avg_confidence,
            "events": [event.to_dict() for event in round_events],
            "discovery_domains": sorted({node.domain.value for node in discovery_nodes}),
        }

    def get_relevant_context(self, query_text: str, max_tokens: int = 2000) -> str:
        """Retrieve compact context using keyword overlap + confidence + recency."""
        query_keywords = set(query_text.lower().split())
        recent_nodes = self.get_recent_discoveries(n=len(self.graph.nodes))
        recent_ids = {node.id for node in recent_nodes[:5]}
        very_recent_ids = {node.id for node in recent_nodes[:20]}

        scored_nodes: list[tuple[ConceptNode, float]] = []
        for node in self.graph.nodes.values():
            formal_keywords = set(node.formal_statement.lower().split())
            overlap = len(query_keywords & formal_keywords)
            keyword_score = min(overlap / (len(query_keywords) + 1), 1.0)
            confidence_score = node.overall_confidence

            if node.id in recent_ids:
                recency = 1.0
            elif node.id in very_recent_ids:
                recency = 0.5
            else:
                recency = 0.2

            total_score = keyword_score * 0.4 + confidence_score * 0.3 + recency * 0.3
            scored_nodes.append((node, total_score))

        scored_nodes.sort(key=lambda pair: pair[1], reverse=True)

        lines = ["# Relevant World Model Context\n"]
        token_count = 10
        for node, score in scored_nodes:
            if token_count >= max_tokens:
                break
            line = (
                f"- [{node.domain.value}] {node.id} "
                f"(confidence: {node.overall_confidence:.2f}, score: {score:.2f})\n"
                f"  {node.formal_statement}\n"
            )
            line_tokens = len(line) // 4
            if token_count + line_tokens > max_tokens:
                break
            lines.append(line)
            token_count += line_tokens

        return "".join(lines)

    def get_session_stats(self) -> dict[str, Any]:
        discovery_events = [e for e in self.temporal_log if e.event_type == "discovery"]
        discovery_nodes = [
            self.graph.get_concept(e.concept_id)
            for e in discovery_events
        ]
        discovery_nodes = [node for node in discovery_nodes if node is not None]

        avg_confidence = (
            sum(node.overall_confidence for node in discovery_nodes) / len(discovery_nodes)
            if discovery_nodes
            else 0.0
        )
        total_rounds = max((e.round_number for e in discovery_events), default=0)
        uptime = 0.0
        if self.temporal_log:
            uptime = max(0.0, time.time() - min(e.timestamp for e in self.temporal_log))

        return {
            "total_concepts": len(self.graph.nodes),
            "total_relations": len(self.graph.relations),
            "total_events": len(self.temporal_log),
            "discovery_count": len(discovery_events),
            "total_rounds": total_rounds,
            "avg_confidence": avg_confidence,
            "domains_discovered": sorted({n.domain.value for n in discovery_nodes}),
            "uptime_seconds": uptime,
        }

    async def save(self, path: Path) -> None:
        """Persist model as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "graph": {
                "title": self.graph.title,
                "source": self.graph.source,
                "nodes": {node_id: node.to_dict() for node_id, node in self.graph.nodes.items()},
                "relations": [rel.to_dict() for rel in self.graph.relations],
            },
            "temporal_log": [event.to_dict() for event in self.temporal_log],
            "timestamp": time.time(),
        }
        payload = json.dumps(state, indent=2, ensure_ascii=False)
        await asyncio.to_thread(path.write_text, payload, encoding="utf-8")
        logger.info("WorldModel saved to %s", path)

    async def load(self, path: Path) -> None:
        """Load model from JSON (supports legacy and current formats)."""
        if not path.exists():
            logger.warning("WorldModel file not found: %s", path)
            return

        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        state = json.loads(raw)
        graph_state = state.get("graph")
        if graph_state is None:
            graph_state = {
                "title": "",
                "source": "",
                "nodes": state.get("concepts", {}),
                "relations": state.get("relations", []),
            }

        rebuilt = TheoryGraph(
            title=graph_state.get("title", ""),
            source=graph_state.get("source", ""),
        )

        nodes_raw = graph_state.get("nodes", {})
        if isinstance(nodes_raw, dict):
            node_items = nodes_raw.values()
        else:
            node_items = nodes_raw

        for node_data in node_items:
            try:
                normalized = self._normalize_legacy_node(node_data)
                rebuilt.add_concept(ConceptNode.from_dict(normalized))
            except Exception as exc:
                logger.debug("Skipping malformed node during load: %s", exc)

        relations_raw = graph_state.get("relations", [])
        if isinstance(relations_raw, dict):
            relation_items = relations_raw.values()
        else:
            relation_items = relations_raw

        for rel_data in relation_items:
            try:
                normalized_rel = self._normalize_legacy_relation(rel_data)
                rebuilt.add_relation(ConceptRelation.from_dict(normalized_rel))
            except Exception as exc:
                logger.debug("Skipping malformed relation during load: %s", exc)

        self.graph = rebuilt
        self.temporal_log = [
            TemporalEvent.from_dict(event)
            for event in state.get("temporal_log", [])
            if isinstance(event, dict)
        ]
        logger.info("WorldModel loaded from %s", path)

    def merge(self, other: WorldModel) -> None:
        """Merge another world model in-place."""
        for node_id, other_node in other.graph.nodes.items():
            existing = self.graph.get_concept(node_id)
            if existing is None or other_node.overall_confidence > existing.overall_confidence:
                self.graph.add_concept(other_node)

        existing_rel_keys = {
            (rel.source_id, rel.target_id, rel.relation_type.value)
            for rel in self.graph.relations
        }
        for rel in other.graph.relations:
            rel_key = (rel.source_id, rel.target_id, rel.relation_type.value)
            if rel_key in existing_rel_keys:
                continue
            self.graph.add_relation(rel)
            existing_rel_keys.add(rel_key)

        existing_event_keys = {
            (e.event_type, e.concept_id, e.round_number, e.timestamp)
            for e in self.temporal_log
        }
        for event in other.temporal_log:
            key = (event.event_type, event.concept_id, event.round_number, event.timestamp)
            if key not in existing_event_keys:
                self.temporal_log.append(event)
                existing_event_keys.add(key)

        self.temporal_log.sort(key=lambda event: event.timestamp)
        logger.info("Merged WorldModel with %d events", len(other.temporal_log))

    @staticmethod
    def _normalize_legacy_node(node_data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(node_data, dict):
            raise ValueError("node_data must be dict")

        concept_id = str(node_data.get("id", "")).strip()
        if not concept_id:
            raise ValueError("missing concept id")

        concept_type = node_data.get("concept_type") or node_data.get("type") or "theorem"
        domain = node_data.get("domain", ScientificDomain.GENERAL.value)
        overall_conf = node_data.get("overall_confidence", node_data.get("confidence", 0.0))

        normalized = dict(node_data)
        normalized["id"] = concept_id
        normalized["concept_type"] = concept_type
        normalized["domain"] = domain
        normalized.setdefault("formal_statement", node_data.get("description", ""))
        normalized.setdefault("informal_statement", "")
        normalized["overall_confidence"] = float(overall_conf)
        normalized.setdefault("verification_status", {})
        normalized.setdefault("tags", node_data.get("tags", []))
        normalized.setdefault("metadata", node_data.get("metadata", {}))
        return normalized

    @staticmethod
    def _normalize_legacy_relation(rel_data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(rel_data, dict):
            raise ValueError("rel_data must be dict")

        relation_type = rel_data.get("relation_type") or rel_data.get("type") or "depends_on"
        normalized = {
            "source_id": rel_data["source_id"],
            "target_id": rel_data["target_id"],
            "relation_type": relation_type,
            "description": rel_data.get("description", ""),
            "strength": float(rel_data.get("strength", 1.0)),
            "bridging_insight": rel_data.get("bridging_insight", ""),
        }
        return normalized

