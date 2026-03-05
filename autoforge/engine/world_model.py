"""
Structured World Model — persistent, queryable knowledge store for autonomous discovery.

Wraps TheoryGraph with temporal queries, cross-session persistence, and intelligent
context retrieval inspired by Kosmos (2025) world models as long-term memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

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
    """Record of a discovery or verification event in the world model timeline."""

    event_type: str  # 'discovery', 'verification', 'relation_added', etc.
    concept_id: str
    round_number: int
    timestamp: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TemporalEvent:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class QueryFilter:
    """Filter criteria for querying the world model."""

    domains: Optional[list[ScientificDomain]] = None
    concept_types: Optional[list[ConceptType]] = None
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    after_round: int = 0
    before_round: int = 999999
    tags: Optional[list[str]] = None
    limit: int = 50


class WorldModel:
    """
    Structured persistent knowledge store wrapping TheoryGraph.

    Provides temporal querying, cross-session persistence, and intelligent
    context retrieval for autonomous discovery agents.
    """

    def __init__(self, graph: Optional[TheoryGraph] = None) -> None:
        """
        Initialize world model.

        Args:
            graph: Optional existing TheoryGraph to wrap. If None, creates new empty graph.
        """
        self.graph = graph or TheoryGraph()
        self.temporal_log: list[TemporalEvent] = []
        self._last_temporal_index = 0
        logger.info("WorldModel initialized")

    def record_event(self, event: TemporalEvent) -> None:
        """
        Record a temporal event in the world model timeline.

        Args:
            event: TemporalEvent to record.
        """
        self.temporal_log.append(event)
        logger.debug(
            f"Event recorded: {event.event_type} for concept {event.concept_id} "
            f"at round {event.round_number}"
        )

    async def add_discovery(
        self, node: ConceptNode, round_number: int, strategy: str
    ) -> None:
        """
        Add a discovered concept to the graph and record temporal event.

        Args:
            node: ConceptNode representing the discovery.
            round_number: Current discovery round number.
            strategy: Strategy used to discover (e.g., 'domain_templated', 'cross_domain').
        """
        # Add to graph
        self.graph.add_concept(node)

        # Record event
        event = TemporalEvent(
            event_type="discovery",
            concept_id=node.id,
            round_number=round_number,
            timestamp=time.time(),
            metadata={
                "strategy": strategy,
                "formal_statement": node.formal_statement,
                "domain": node.domain.value,
                "confidence": node.confidence,
            },
        )
        self.record_event(event)

    def query(self, filter_: QueryFilter) -> list[ConceptNode]:
        """
        Retrieve concepts from the graph filtered by criteria.

        Args:
            filter_: QueryFilter with desired criteria.

        Returns:
            List of ConceptNode objects matching the filter.
        """
        results = []

        for node_id, node in self.graph.concepts.items():
            # Domain filter
            if filter_.domains and node.domain not in filter_.domains:
                continue

            # Type filter
            if filter_.concept_types and node.type not in filter_.concept_types:
                continue

            # Confidence filter
            if not (filter_.min_confidence <= node.confidence <= filter_.max_confidence):
                continue

            # Tags filter
            if filter_.tags:
                node_tags = set(getattr(node, "tags", []))
                if not any(tag in node_tags for tag in filter_.tags):
                    continue

            results.append(node)

        # Filter by temporal bounds using temporal log
        if filter_.after_round > 0 or filter_.before_round < 999999:
            node_ids_in_round = set()
            for event in self.temporal_log:
                if (
                    filter_.after_round <= event.round_number <= filter_.before_round
                ):
                    node_ids_in_round.add(event.concept_id)

            results = [n for n in results if n.id in node_ids_in_round]

        # Sort by confidence (descending) and limit
        results.sort(key=lambda n: n.confidence, reverse=True)
        return results[: filter_.limit]

    def get_recent_discoveries(self, n: int = 10) -> list[ConceptNode]:
        """
        Get the N most recent discoveries by timestamp.

        Args:
            n: Number of recent discoveries to return.

        Returns:
            List of ConceptNode objects from recent discoveries.
        """
        recent_events = sorted(
            [e for e in self.temporal_log if e.event_type == "discovery"],
            key=lambda e: e.timestamp,
            reverse=True,
        )[:n]

        recent_node_ids = {e.concept_id for e in recent_events}
        recent_nodes = [
            self.graph.concepts[nid]
            for nid in recent_node_ids
            if nid in self.graph.concepts
        ]

        return sorted(
            recent_nodes,
            key=lambda n: next(
                (e.timestamp for e in recent_events if e.concept_id == n.id), 0
            ),
            reverse=True,
        )

    def get_round_summary(self, round_number: int) -> dict:
        """
        Get statistics and summary for a specific round.

        Args:
            round_number: The round to summarize.

        Returns:
            Dictionary with round statistics and events.
        """
        round_events = [e for e in self.temporal_log if e.round_number == round_number]

        if not round_events:
            return {
                "round_number": round_number,
                "event_count": 0,
                "discoveries": [],
            }

        discoveries = [e for e in round_events if e.event_type == "discovery"]
        discovery_nodes = [
            self.graph.concepts.get(e.concept_id) for e in discoveries
        ]
        discovery_nodes = [n for n in discovery_nodes if n is not None]

        avg_confidence = (
            sum(n.confidence for n in discovery_nodes) / len(discovery_nodes)
            if discovery_nodes
            else 0.0
        )

        return {
            "round_number": round_number,
            "event_count": len(round_events),
            "discovery_count": len(discoveries),
            "avg_confidence": avg_confidence,
            "events": [e.to_dict() for e in round_events],
            "discovery_domains": list(
                set(n.domain.value for n in discovery_nodes)
            ),
        }

    def get_relevant_context(
        self, query_text: str, max_tokens: int = 2000
    ) -> str:
        """
        Retrieve relevant concepts for a query using TF-IDF-like scoring.

        Scores concepts based on:
        - Keyword overlap with query_text (40%)
        - Confidence level (30%)
        - Recency bonus (30%)

        Args:
            query_text: Natural language query or prompt fragment.
            max_tokens: Maximum tokens in returned context string.

        Returns:
            Compact formatted string of most relevant concepts.
        """
        # Normalize query for keyword matching
        query_keywords = set(query_text.lower().split())

        scored_nodes = []

        # Get all recent discoveries for recency calculation
        all_recent = self.get_recent_discoveries(n=len(self.graph.concepts))
        recent_ids = {n.id for n in all_recent[:5]}
        very_recent_ids = {n.id for n in all_recent[:20]}

        for node in self.graph.concepts.values():
            # Keyword overlap score (TF-IDF-like)
            formal_keywords = set(node.formal_statement.lower().split())
            overlap = len(query_keywords & formal_keywords)
            keyword_score = min(overlap / (len(query_keywords) + 1), 1.0)

            # Confidence score (normalized 0-1)
            confidence_score = node.confidence

            # Recency bonus
            if node.id in recent_ids:
                recency_bonus = 1.0
            elif node.id in very_recent_ids:
                recency_bonus = 0.5
            else:
                recency_bonus = 0.2

            # Combined score
            total_score = (
                keyword_score * 0.4 + confidence_score * 0.3 + recency_bonus * 0.3
            )

            scored_nodes.append((node, total_score))

        # Sort by score and format
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # Build context string up to token limit
        context_lines = ["# Relevant World Model Context\n"]
        token_count = 10  # Header tokens

        for node, score in scored_nodes:
            if token_count >= max_tokens:
                break

            line = (
                f"- [{node.domain.value}] {node.id} (confidence: {node.confidence:.2f})\n"
                f"  {node.formal_statement}\n"
            )

            # Rough token estimate: 1 token ≈ 4 chars
            line_tokens = len(line) // 4
            if token_count + line_tokens <= max_tokens:
                context_lines.append(line)
                token_count += line_tokens

        return "".join(context_lines)

    def get_session_stats(self) -> dict:
        """
        Get overall statistics for the current session.

        Returns:
            Dictionary with session-level statistics.
        """
        total_concepts = len(self.graph.concepts)
        total_relations = len(self.graph.relations)

        discovery_events = [e for e in self.temporal_log if e.event_type == "discovery"]
        total_rounds = max([e.round_number for e in discovery_events], default=0)

        discovery_nodes = [
            self.graph.concepts.get(e.concept_id)
            for e in discovery_events
        ]
        discovery_nodes = [n for n in discovery_nodes if n is not None]

        avg_confidence = (
            sum(n.confidence for n in discovery_nodes) / len(discovery_nodes)
            if discovery_nodes
            else 0.0
        )

        domains_discovered = set()
        for node in discovery_nodes:
            domains_discovered.add(node.domain.value)

        return {
            "total_concepts": total_concepts,
            "total_relations": total_relations,
            "total_events": len(self.temporal_log),
            "discovery_count": len(discovery_events),
            "total_rounds": total_rounds,
            "avg_confidence": avg_confidence,
            "domains_discovered": list(domains_discovered),
            "uptime_seconds": (
                time.time() - self.temporal_log[0].timestamp
                if self.temporal_log
                else 0
            ),
        }

    async def save(self, path: Path) -> None:
        """
        Persist world model to JSON file.

        Args:
            path: Path to save JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize graph concepts
        concepts_data = {}
        for node_id, node in self.graph.concepts.items():
            concepts_data[node_id] = {
                "id": node.id,
                "formal_statement": node.formal_statement,
                "domain": node.domain.value,
                "type": node.type.value,
                "confidence": node.confidence,
                "tags": getattr(node, "tags", []),
            }

        # Serialize relations
        relations_data = []
        for rel_id, rel in self.graph.relations.items():
            relations_data.append(
                {
                    "id": rel_id,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "type": rel.type.value,
                    "strength": rel.strength,
                }
            )

        # Serialize temporal log
        temporal_data = [e.to_dict() for e in self.temporal_log]

        # Write to file
        state = {
            "concepts": concepts_data,
            "relations": relations_data,
            "temporal_log": temporal_data,
            "timestamp": time.time(),
        }

        async def _write() -> None:
            path.write_text(json.dumps(state, indent=2))

        await _write()
        logger.info(f"WorldModel saved to {path}")

    async def load(self, path: Path) -> None:
        """
        Restore world model from JSON file.

        Args:
            path: Path to load JSON file from.
        """
        if not path.exists():
            logger.warning(f"WorldModel file not found: {path}")
            return

        async def _read() -> dict:
            return json.loads(path.read_text())

        state = await _read()

        # Restore concepts
        self.graph.concepts.clear()
        for concept_id, concept_data in state.get("concepts", {}).items():
            node = ConceptNode(
                id=concept_data["id"],
                formal_statement=concept_data["formal_statement"],
                domain=ScientificDomain(concept_data["domain"]),
                type=ConceptType(concept_data["type"]),
                confidence=concept_data["confidence"],
            )
            node.tags = concept_data.get("tags", [])
            self.graph.concepts[concept_id] = node

        # Restore relations
        self.graph.relations.clear()
        for rel_data in state.get("relations", []):
            rel = ConceptRelation(
                id=rel_data["id"],
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                type=RelationType(rel_data["type"]),
                strength=rel_data["strength"],
            )
            self.graph.relations[rel_data["id"]] = rel

        # Restore temporal log
        self.temporal_log.clear()
        for event_data in state.get("temporal_log", []):
            event = TemporalEvent.from_dict(event_data)
            self.temporal_log.append(event)

        logger.info(f"WorldModel loaded from {path}")

    def merge(self, other: WorldModel) -> None:
        """
        Merge another world model into this one.

        Useful for cross-session accumulation of knowledge.
        Prioritizes higher-confidence concepts and recent discoveries.

        Args:
            other: Another WorldModel to merge.
        """
        # Merge concepts (keep highest confidence version)
        for node_id, other_node in other.graph.concepts.items():
            if node_id in self.graph.concepts:
                existing = self.graph.concepts[node_id]
                if other_node.confidence > existing.confidence:
                    self.graph.concepts[node_id] = other_node
                    logger.debug(f"Updated concept {node_id} with higher confidence")
            else:
                self.graph.concepts[node_id] = other_node
                logger.debug(f"Added concept {node_id} from merge")

        # Merge relations
        for rel_id, rel in other.graph.relations.items():
            if rel_id not in self.graph.relations:
                self.graph.relations[rel_id] = rel
                logger.debug(f"Added relation {rel_id} from merge")

        # Merge temporal log
        self.temporal_log.extend(other.temporal_log)
        self.temporal_log.sort(key=lambda e: e.timestamp)

        logger.info(f"Merged WorldModel with {len(other.temporal_log)} events")
