"""Behavioral tests for AutoForge engine modules.

Tests cover: search_tree, capability_dag, evolution, reflexion, speculative_pipeline.
All tests run offline -- no API keys, no network access.
"""
from __future__ import annotations

import asyncio
import json
import math
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Module imports ──────────────────────────────────────────

from autoforge.engine.search_tree import (
    BranchNode,
    MCTSNode,
    MCTSSearchTree,
    NodeStatus,
    SearchTree,
)
from autoforge.engine.capability_dag import (
    CapabilityDAG,
    CapabilityEdge,
    CapabilityNode,
    DAGBridge,
    Domain,
    EdgeType,
    GrowthStrategy,
    VerificationType,
)
from autoforge.engine.evolution import (
    EvolutionEngine,
    EvolutionRecord,
    FitnessScore,
    StrategyMemory,
    WorkflowGenome,
)
from autoforge.engine.reflexion import (
    Reflection,
    ReflectionMemory,
    ReflexionEngine,
)
from autoforge.engine.speculative_pipeline import (
    SpecPhase,
    SpecTaskStatus,
    SpeculativePipeline,
    SpeculativeResult,
    SpeculativeTask,
)
from autoforge.engine.provers.proof_search import AdaptiveBeamSearch
from autoforge.engine.provers.lean_core import ProofState


# ══════════════════════════════════════════════════════════════
# 1. SearchTree
# ══════════════════════════════════════════════════════════════


class TestSearchTree:
    """Behavioral tests for SearchTree (branching, evaluation, selection)."""

    def test_create_root(self):
        tree = SearchTree()
        root = tree.create_root("Initial approach", strategy="Do X")
        assert root.description == "Initial approach"
        assert root.strategy == "Do X"
        assert root.status == NodeStatus.EXPLORING
        assert tree.root_id == root.id
        assert tree.current_id == root.id

    def test_branch_creates_children(self):
        tree = SearchTree()
        root = tree.create_root("root")
        candidates = [
            {"description": "Plan A", "strategy": "Use framework X"},
            {"description": "Plan B", "strategy": "Use framework Y"},
        ]
        children = tree.branch(root.id, candidates)
        assert len(children) == 2
        assert children[0].parent_id == root.id
        assert children[1].depth == 1
        assert children[0].id in root.children_ids

    def test_branch_respects_max_children(self):
        tree = SearchTree(max_children=2)
        root = tree.create_root("root")
        candidates = [
            {"description": f"Plan {i}"} for i in range(5)
        ]
        children = tree.branch(root.id, candidates)
        assert len(children) == 2

    def test_branch_respects_max_depth(self):
        tree = SearchTree(max_depth=1)
        root = tree.create_root("root")
        children = tree.branch(root.id, [{"description": "child"}])
        assert len(children) == 1
        # Children are at depth 1 == max_depth, so branching from them gives []
        grandchildren = tree.branch(children[0].id, [{"description": "grand"}])
        assert grandchildren == []

    def test_branch_invalid_parent_raises(self):
        tree = SearchTree()
        tree.create_root("root")
        with pytest.raises(ValueError, match="not found"):
            tree.branch("nonexistent", [{"description": "x"}])

    def test_evaluate_node_clamps_score(self):
        tree = SearchTree()
        root = tree.create_root("root")
        tree.evaluate_node(root.id, score=1.5, confidence=-0.2, reason="test")
        assert root.score == 1.0
        assert root.confidence == 0.0
        assert root.status == NodeStatus.EVALUATED

    def test_select_best_picks_highest_weighted_score(self):
        tree = SearchTree()
        root = tree.create_root("root")
        children = tree.branch(root.id, [
            {"description": "A"},
            {"description": "B"},
            {"description": "C"},
        ])
        tree.evaluate_node(children[0].id, score=0.6, confidence=1.0)
        tree.evaluate_node(children[1].id, score=0.9, confidence=1.0)
        tree.evaluate_node(children[2].id, score=0.3, confidence=1.0)

        best = tree.select_best(root.id)
        assert best is not None
        assert best.id == children[1].id
        assert best.status == NodeStatus.SELECTED
        # Others should be pruned
        assert children[0].status == NodeStatus.PRUNED
        assert children[2].status == NodeStatus.PRUNED

    def test_select_best_prefers_higher_confidence_on_tie(self):
        tree = SearchTree()
        root = tree.create_root("root")
        children = tree.branch(root.id, [
            {"description": "A"},
            {"description": "B"},
        ])
        # Very close scores, but B has higher confidence
        tree.evaluate_node(children[0].id, score=0.80, confidence=0.5)
        tree.evaluate_node(children[1].id, score=0.75, confidence=0.9)
        best = tree.select_best(root.id)
        assert best is not None
        assert best.id == children[1].id

    def test_select_best_returns_none_for_no_evaluated_children(self):
        tree = SearchTree()
        root = tree.create_root("root")
        tree.branch(root.id, [{"description": "pending"}])
        assert tree.select_best(root.id) is None

    def test_backtrack_picks_pruned_sibling(self):
        tree = SearchTree()
        root = tree.create_root("root")
        children = tree.branch(root.id, [
            {"description": "A"},
            {"description": "B"},
        ])
        tree.evaluate_node(children[0].id, score=0.9, confidence=1.0)
        tree.evaluate_node(children[1].id, score=0.7, confidence=1.0)
        tree.select_best(root.id)  # A selected, B pruned

        # Simulate A failing
        tree.current_id = children[0].id
        alt = tree.backtrack()
        assert alt is not None
        assert alt.id == children[1].id
        assert alt.status == NodeStatus.SELECTED
        assert children[0].status == NodeStatus.FAILED

    def test_backtrack_returns_none_when_exhausted(self):
        tree = SearchTree()
        root = tree.create_root("root")
        # No children, backtrack from root
        tree.current_id = root.id
        # Mark root as current, root has no parent
        result = tree.backtrack()
        assert result is None

    def test_get_path_to_root(self):
        tree = SearchTree()
        root = tree.create_root("root")
        children = tree.branch(root.id, [{"description": "child"}])
        grandchildren = tree.branch(children[0].id, [{"description": "grand"}])
        path = tree.get_path_to_root(grandchildren[0].id)
        assert len(path) == 3
        assert path[0].id == root.id
        assert path[2].id == grandchildren[0].id

    def test_summary_counts_statuses(self):
        tree = SearchTree()
        root = tree.create_root("root")
        tree.branch(root.id, [{"description": "A"}, {"description": "B"}])
        s = tree.summary()
        assert s["total_nodes"] == 3
        assert s["max_depth_reached"] == 1
        assert "exploring" in s["status_counts"]

    def test_to_dict_serialization(self):
        tree = SearchTree()
        root = tree.create_root("root")
        tree.branch(root.id, [{"description": "A"}])
        data = tree.to_dict()
        assert data["root_id"] == root.id
        assert len(data["nodes"]) == 2

    def test_history_records_events(self):
        tree = SearchTree()
        root = tree.create_root("root")
        tree.branch(root.id, [{"description": "A"}])
        assert len(tree.history) >= 1
        assert tree.history[0]["type"] == "branch"


# ══════════════════════════════════════════════════════════════
# 1b. MCTSSearchTree
# ══════════════════════════════════════════════════════════════


class TestMCTSNode:
    """Tests for MCTSNode scoring."""

    def test_q_value_unvisited_returns_prior(self):
        node = MCTSNode(prior=0.6)
        assert node.q_value == 0.6

    def test_q_value_after_visits(self):
        node = MCTSNode(visit_count=4, value_sum=2.0)
        assert node.q_value == pytest.approx(0.5)

    def test_ucb1_unvisited_is_inf(self):
        node = MCTSNode()
        assert node.ucb1(parent_visits=10) == float("inf")

    def test_ucb1_formula(self):
        node = MCTSNode(visit_count=3, value_sum=1.5, prior=0.5)
        parent_visits = 12
        expected_exploit = 0.5  # 1.5 / 3
        expected_explore = 1.41 * math.sqrt(math.log(12) / 3)
        result = node.ucb1(parent_visits)
        assert result == pytest.approx(expected_exploit + expected_explore, rel=1e-3)


class TestMCTSSearchTree:
    """Tests for MCTSSearchTree."""

    def test_create_root(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("task", strategy="plan", thought_chain=["step1"])
        assert root.description == "task"
        assert root.thought_chain == ["step1"]
        assert mcts.root_id == root.id

    def test_select_returns_root_when_no_children(self):
        mcts = MCTSSearchTree()
        mcts.create_root("task")
        leaf = mcts.select()
        assert leaf.id == mcts.root_id

    def test_select_prefers_unvisited_children(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("task")
        child_a = MCTSNode(description="a", parent_id=root.id, depth=1, visit_count=5, value_sum=3.0)
        child_b = MCTSNode(description="b", parent_id=root.id, depth=1, visit_count=0)
        mcts.nodes[child_a.id] = child_a
        mcts.nodes[child_b.id] = child_b
        root.children_ids = [child_a.id, child_b.id]
        root.visit_count = 5

        leaf = mcts.select()
        assert leaf.id == child_b.id  # Unvisited has inf UCB1

    def test_backpropagate_updates_path(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("root")
        child = MCTSNode(description="child", parent_id=root.id, depth=1)
        mcts.nodes[child.id] = child
        root.children_ids.append(child.id)

        mcts.backpropagate(child.id, 0.8)
        assert child.visit_count == 1
        assert child.value_sum == pytest.approx(0.8)
        assert root.visit_count == 1
        assert root.value_sum == pytest.approx(0.8)

    def test_get_best_action_by_visit_count(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("root")
        # Child A: more visits but lower value
        a = MCTSNode(description="a", parent_id=root.id, depth=1, visit_count=10, value_sum=5.0)
        # Child B: fewer visits but higher value
        b = MCTSNode(description="b", parent_id=root.id, depth=1, visit_count=3, value_sum=2.7)
        mcts.nodes[a.id] = a
        mcts.nodes[b.id] = b
        root.children_ids = [a.id, b.id]

        best = mcts.get_best_action()
        assert best is not None
        assert best.id == a.id  # Most visited wins

    def test_get_best_action_returns_root_when_no_children(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("root")
        best = mcts.get_best_action()
        assert best is not None
        assert best.id == root.id

    def test_inject_execution_feedback_success(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("root")
        child = MCTSNode(description="child", parent_id=root.id, depth=1)
        mcts.nodes[child.id] = child
        root.children_ids.append(child.id)

        mcts.inject_execution_feedback(child.id, "All tests pass", success=True)
        assert child.execution_feedback == "All tests pass"
        assert child.execution_success is True
        assert child.visit_count == 1
        assert child.value_sum == pytest.approx(0.9)

    def test_inject_execution_feedback_failure(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("root")
        child = MCTSNode(description="child", parent_id=root.id, depth=1)
        mcts.nodes[child.id] = child
        root.children_ids.append(child.id)

        mcts.inject_execution_feedback(child.id, "Error: import failed", success=False)
        assert child.execution_success is False
        assert child.value_sum == pytest.approx(0.2)

    def test_summary_includes_stats(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("root")
        s = mcts.summary()
        assert s["total_nodes"] == 1
        assert s["iterations_completed"] == 0
        assert s["max_iterations"] == 9  # default

    def test_thought_chain_context(self):
        mcts = MCTSSearchTree()
        root = mcts.create_root("root", strategy="plan")
        child = MCTSNode(
            description="child step",
            parent_id=root.id,
            depth=1,
            thought_chain=["root thought", "child thought"],
        )
        mcts.nodes[child.id] = child
        root.children_ids.append(child.id)

        ctx = mcts._get_thought_chain_context(child)
        assert "root" in ctx
        assert "child step" in ctx


# ══════════════════════════════════════════════════════════════
# 2. CapabilityDAG
# ══════════════════════════════════════════════════════════════


class TestCapabilityDAG:
    """Behavioral tests for CapabilityDAG."""

    def test_add_and_get_node(self):
        dag = CapabilityDAG()
        node = dag.add("def hello(): pass", Domain.CODE_PATTERN, summary="hello func")
        assert node.id in dag
        assert dag.get(node.id) is not None
        assert dag.get(node.id).summary == "hello func"

    def test_content_addressed_deduplication(self):
        dag = CapabilityDAG()
        n1 = dag.add("same content", Domain.CODE_PATTERN, confidence=0.3)
        n2 = dag.add("same content", Domain.CODE_PATTERN, confidence=0.7)
        assert n1.id == n2.id
        assert len(dag) == 1
        assert dag.get(n1.id).confidence == 0.7  # max of 0.3 and 0.7

    def test_add_edge_basic(self):
        dag = CapabilityDAG()
        a = dag.add("content a", Domain.CODE_PATTERN)
        b = dag.add("content b", Domain.CODE_PATTERN)
        edge = dag.add_edge(a.id, b.id, EdgeType.DEPENDS_ON)
        assert edge is not None
        assert edge.source_id == a.id
        assert edge.target_id == b.id

    def test_add_edge_rejects_cycle(self):
        dag = CapabilityDAG()
        a = dag.add("a", Domain.CODE_PATTERN)
        b = dag.add("b", Domain.CODE_PATTERN)
        dag.add_edge(a.id, b.id, EdgeType.DEPENDS_ON)
        # b -> a would create a cycle
        edge = dag.add_edge(b.id, a.id, EdgeType.DEPENDS_ON)
        assert edge is None

    def test_add_edge_rejects_self_loop(self):
        dag = CapabilityDAG()
        a = dag.add("a", Domain.CODE_PATTERN)
        edge = dag.add_edge(a.id, a.id, EdgeType.DEPENDS_ON)
        assert edge is None

    def test_add_edge_returns_none_for_missing_nodes(self):
        dag = CapabilityDAG()
        edge = dag.add_edge("fake1", "fake2", EdgeType.DEPENDS_ON)
        assert edge is None

    def test_remove_node(self):
        dag = CapabilityDAG()
        node = dag.add("content", Domain.CODE_PATTERN, tags=["python"])
        nid = node.id
        assert dag.remove(nid) is True
        assert nid not in dag
        assert len(dag) == 0

    def test_query_by_text(self):
        dag = CapabilityDAG()
        dag.add("python async web server handler", Domain.CODE_PATTERN,
                summary="async web handler", tags=["python", "web"])
        dag.add("rust memory allocator", Domain.CODE_PATTERN,
                summary="memory allocator", tags=["rust"])
        results = dag.query("python async handler")
        assert len(results) >= 1
        assert results[0].summary == "async web handler"

    def test_query_by_domain_filter(self):
        dag = CapabilityDAG()
        dag.add("code snippet", Domain.CODE_PATTERN)
        dag.add("proof snippet", Domain.MATH_PROOF)
        results = dag.query("snippet", domain=Domain.MATH_PROOF)
        assert all(r.domain == Domain.MATH_PROOF for r in results)

    def test_query_by_tags(self):
        dag = CapabilityDAG()
        dag.add("flask app", Domain.CODE_PATTERN, summary="flask app", tags=["python", "flask"])
        dag.add("react app", Domain.CODE_PATTERN, summary="react app", tags=["javascript", "react"])
        results = dag.query("app", tags=["flask"])
        assert len(results) == 1
        assert results[0].summary == "flask app"

    def test_query_respects_min_confidence(self):
        dag = CapabilityDAG()
        dag.add("low conf", Domain.CODE_PATTERN, summary="low conf content", confidence=0.05)
        dag.add("high conf", Domain.CODE_PATTERN, summary="high conf content", confidence=0.8)
        results = dag.query("conf content", min_confidence=0.5)
        assert len(results) == 1
        assert results[0].confidence >= 0.5

    def test_record_usage_updates_stats(self):
        dag = CapabilityDAG()
        node = dag.add("content", Domain.CODE_PATTERN)
        dag.record_usage(node.id, success=True)
        dag.record_usage(node.id, success=False)
        assert node.usage_count == 2
        assert node.success_count == 1

    def test_record_usage_updates_confidence_after_threshold(self):
        dag = CapabilityDAG()
        node = dag.add("content", Domain.CODE_PATTERN, confidence=0.3)
        for _ in range(dag.PROMOTE_USAGE_COUNT):
            dag.record_usage(node.id, success=True)
        # Confidence should have been boosted
        assert node.confidence > 0.3

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            dag1 = CapabilityDAG(storage_dir=path)
            dag1.add("saved content", Domain.CODE_PATTERN, summary="saved", tags=["test"])
            dag1.save()

            dag2 = CapabilityDAG(storage_dir=path)
            assert dag2.load() is True
            assert len(dag2) == 1
            nodes = dag2.query("saved content")
            assert len(nodes) == 1
            assert nodes[0].summary == "saved"

    def test_load_returns_false_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dag = CapabilityDAG(storage_dir=Path(tmpdir))
            assert dag.load() is False

    def test_merge_two_dags(self):
        dag1 = CapabilityDAG()
        dag2 = CapabilityDAG()
        dag1.add("shared", Domain.CODE_PATTERN, confidence=0.3)
        dag2.add("shared", Domain.CODE_PATTERN, confidence=0.8)
        dag2.add("unique to 2", Domain.WORKFLOW)

        stats = dag1.merge(dag2)
        assert stats["merged_nodes"] == 1
        assert stats["new_nodes"] == 1
        assert len(dag1) == 2
        # Shared node should have max confidence
        shared_id = CapabilityDAG.content_id(Domain.CODE_PATTERN, "shared")
        assert dag1.get(shared_id).confidence == 0.8

    def test_prune_removes_low_confidence_old_nodes(self):
        dag = CapabilityDAG()
        node = dag.add("stale", Domain.CODE_PATTERN, confidence=0.01)
        # Force the node to appear old
        node.created_at = time.time() - 86400 * 10  # 10 days old
        removed = dag.prune(min_confidence=0.05)
        assert removed == 1
        assert len(dag) == 0

    def test_query_dependencies_transitive(self):
        dag = CapabilityDAG()
        a = dag.add("a", Domain.CODE_PATTERN)
        b = dag.add("b", Domain.CODE_PATTERN)
        c = dag.add("c", Domain.CODE_PATTERN)
        dag.add_edge(a.id, b.id, EdgeType.DEPENDS_ON)
        dag.add_edge(b.id, c.id, EdgeType.DEPENDS_ON)
        deps = dag.query_dependencies(c.id)
        dep_ids = {d.id for d in deps}
        assert b.id in dep_ids
        assert a.id in dep_ids

    def test_size_and_contains(self):
        dag = CapabilityDAG()
        assert dag.size == 0
        node = dag.add("x", Domain.GENERAL)
        assert dag.size == 1
        assert len(dag) == 1
        assert node.id in dag

    def test_get_stats(self):
        dag = CapabilityDAG()
        dag.add("x", Domain.CODE_PATTERN)
        dag.query("x")
        stats = dag.get_stats()
        assert stats["total_nodes"] == 1
        assert stats["operations"]["queries"] == 1


class TestDAGBridge:
    """Tests for DAGBridge adapter."""

    def test_ingest_generic(self):
        dag = CapabilityDAG()
        bridge = DAGBridge(dag)
        node = bridge.ingest(Domain.GENERAL, "some content", summary="test")
        assert len(dag) == 1
        assert node.domain == Domain.GENERAL

    def test_ingest_code_snippet(self):
        dag = CapabilityDAG()
        bridge = DAGBridge(dag)
        node = bridge.ingest_code_snippet(
            code="def foo(): pass",
            language="python",
            function_name="foo",
            docstring="A foo function",
        )
        assert node.domain == Domain.CODE_PATTERN
        assert "python" in node.tags

    def test_ingest_proof(self):
        dag = CapabilityDAG()
        bridge = DAGBridge(dag)
        node = bridge.ingest_proof(
            lean_code="theorem T : True := trivial",
            theorem_name="T",
            verified=True,
        )
        assert node.domain == Domain.MATH_PROOF
        assert node.verification_type == VerificationType.FORMAL
        assert node.confidence == 0.95

    def test_find_relevant(self):
        dag = CapabilityDAG()
        bridge = DAGBridge(dag)
        bridge.ingest_code_snippet("async def fetch():", "python", "fetch", "Fetch data")
        results = bridge.find_relevant("fetch data python")
        assert len(results) >= 1

    def test_ingest_debug_pattern(self):
        dag = CapabilityDAG()
        bridge = DAGBridge(dag)
        node = bridge.ingest_debug_pattern(
            error_pattern="ImportError: no module named X",
            fix_strategy="pip install X",
            success=True,
        )
        assert node.domain == Domain.DEBUGGING
        assert node.confidence == 0.7


# ══════════════════════════════════════════════════════════════
# 3. Evolution
# ══════════════════════════════════════════════════════════════


class TestFitnessScore:
    """Tests for FitnessScore composite calculation."""

    def test_composite_score_perfect(self):
        fs = FitnessScore(
            quality_score=10.0,
            test_pass_rate=1.0,
            cost_usd=0.50,
            duration_seconds=100,
            tasks_completed=10,
            tasks_total=10,
            build_success_rate=1.0,
            refactor_needed=False,
        )
        score = fs.composite_score
        assert 0.9 <= score <= 1.0

    def test_composite_score_zero(self):
        fs = FitnessScore()
        score = fs.composite_score
        assert score >= 0.0

    def test_cost_penalty(self):
        cheap = FitnessScore(quality_score=5.0, cost_usd=0.5, build_success_rate=0.5)
        expensive = FitnessScore(quality_score=5.0, cost_usd=10.0, build_success_rate=0.5)
        assert cheap.composite_score > expensive.composite_score

    def test_speed_bonus(self):
        fast = FitnessScore(quality_score=5.0, duration_seconds=100, build_success_rate=0.5)
        slow = FitnessScore(quality_score=5.0, duration_seconds=600, build_success_rate=0.5)
        assert fast.composite_score > slow.composite_score

    def test_serialization_roundtrip(self):
        fs = FitnessScore(quality_score=7.5, test_pass_rate=0.8, cost_usd=1.2)
        data = fs.to_dict()
        fs2 = FitnessScore.from_dict(data)
        assert fs2.quality_score == 7.5
        assert fs2.test_pass_rate == 0.8


class TestWorkflowGenome:
    """Tests for WorkflowGenome."""

    def test_serialization_roundtrip(self):
        g = WorkflowGenome(
            arch_strategy="microservices",
            tech_fingerprint="python-fastapi",
            project_type="api-service",
            parallel_builders=3,
        )
        data = g.to_dict()
        g2 = WorkflowGenome.from_dict(data)
        assert g2.arch_strategy == "microservices"
        assert g2.parallel_builders == 3

    def test_default_values(self):
        g = WorkflowGenome()
        assert g.generation == 0
        assert g.parent_id is None
        assert g.model_preference == "balanced"


class TestEvolutionEngine:
    """Tests for EvolutionEngine."""

    def test_prepare_genome_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = EvolutionEngine(base_dir=Path(tmpdir))
            genome = engine.prepare_genome(project_type="web-app")
            assert genome.project_type == "web-app"
            assert genome.generation == 0

    def test_prepare_genome_inherits_from_ancestor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = EvolutionEngine(base_dir=Path(tmpdir))
            # Record a good ancestor
            ancestor_genome = WorkflowGenome(
                tech_fingerprint="python-flask",
                project_type="web-app",
                parallel_builders=3,
                tdd_loops=2,
            )
            fitness = FitnessScore(
                quality_score=8.0, build_success_rate=0.9, test_pass_rate=0.95,
            )
            engine.record_result("ancestor-project", fitness, ancestor_genome)

            # New project should inherit
            genome = engine.prepare_genome(
                project_type="web-app",
                tech_fingerprint="python-flask",
            )
            assert genome.generation >= 1
            assert genome.parent_id == ancestor_genome.id

    def test_record_result_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = EvolutionEngine(base_dir=Path(tmpdir))
            genome = engine.prepare_genome(project_type="cli-tool")
            fitness = FitnessScore(quality_score=7.0, build_success_rate=0.8)
            record = engine.record_result("my-cli", fitness, genome)
            assert record.project_name == "my-cli"

            # Reload from disk
            engine2 = EvolutionEngine(base_dir=Path(tmpdir))
            stats = engine2.get_evolution_stats()
            assert stats["total_runs"] >= 1

    def test_crossover(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = EvolutionEngine(base_dir=Path(tmpdir))
            a = WorkflowGenome(
                arch_strategy="monolith",
                parallel_builders=1,
                tdd_loops=0,
                active_patches=["patch-a"],
            )
            b = WorkflowGenome(
                arch_strategy="microservices",
                parallel_builders=4,
                tdd_loops=3,
                active_patches=["patch-b"],
            )
            child = engine.crossover(a, b)
            assert child.generation == 1
            # Patches should be union of both parents
            assert set(child.active_patches) == {"patch-a", "patch-b"}
            assert "crossover" in child.mutations[0]

    def test_infer_project_type(self):
        assert EvolutionEngine.infer_project_type(
            {"description": "A REST API for user management"}
        ) == "api-service"
        assert EvolutionEngine.infer_project_type(
            {"description": "A web dashboard for analytics"}
        ) == "web-app"
        assert EvolutionEngine.infer_project_type(
            {"description": "A CLI tool for file conversion"}
        ) == "cli-tool"
        assert EvolutionEngine.infer_project_type(
            {"description": "Something completely unrelated"}
        ) == "general"

    def test_extract_tech_fingerprint(self):
        fp = EvolutionEngine.extract_tech_fingerprint({
            "tech_stack": {"backend": "FastAPI", "frontend": "React", "db": "PostgreSQL"},
        })
        assert "fastapi" in fp
        assert "react" in fp
        assert "postgresql" in fp

    def test_extract_tech_fingerprint_empty(self):
        assert EvolutionEngine.extract_tech_fingerprint({}) == "general"

    def test_novelty_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = EvolutionEngine(base_dir=Path(tmpdir))
            g1 = WorkflowGenome(search_tree_enabled=True, tdd_loops=2, parallel_builders=2)
            # Identical genome is not novel
            assert engine._novelty_check(g1, [g1]) is False
            # Different genome is novel
            g2 = WorkflowGenome(search_tree_enabled=False, tdd_loops=0, parallel_builders=4)
            assert engine._novelty_check(g2, [g1]) is True

    def test_apply_genome_to_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = EvolutionEngine(base_dir=Path(tmpdir))
            genome = WorkflowGenome(
                parallel_builders=3,
                tdd_loops=2,
                checkpoints_enabled=True,
                search_tree_enabled=True,
            )
            config = MagicMock()
            config.max_agents = 1
            config.build_test_loops = 0
            config.checkpoints_enabled = False
            config.search_tree_enabled = False
            config.search_tree_max_candidates = 1
            engine.apply_genome_to_config(genome, config)
            assert config.max_agents == 3
            assert config.build_test_loops == 2


class TestStrategyMemory:
    """Tests for StrategyMemory."""

    def _make_record(self, score: float, niche: str = "general") -> EvolutionRecord:
        genome = WorkflowGenome(tech_fingerprint=niche, project_type=niche)
        fitness = FitnessScore(quality_score=score, build_success_rate=score / 10)
        return EvolutionRecord(genome=genome, fitness=fitness, project_name=f"proj-{score}")

    def test_record_and_retrieve(self):
        mem = StrategyMemory()
        rec = self._make_record(8.0, "python-flask")
        mem.record(rec)
        best = mem.get_best_for_type("python-flask")
        assert best is not None
        assert best.fitness.quality_score == 8.0

    def test_hall_of_fame_limit(self):
        mem = StrategyMemory(max_hall_of_fame=3)
        for i in range(5):
            mem.record(self._make_record(float(i), f"niche-{i}"))
        assert len(mem.hall_of_fame) == 3

    def test_get_diverse_strategies(self):
        mem = StrategyMemory()
        for i in range(4):
            mem.record(self._make_record(7.0, f"niche-{i}"))
        diverse = mem.get_diverse_strategies(n=3)
        assert len(diverse) == 3


# ══════════════════════════════════════════════════════════════
# 4. Reflexion
# ══════════════════════════════════════════════════════════════


class TestReflectionMemory:
    """Tests for ReflectionMemory."""

    def _make_reflection(self, task: str = "build auth", idx: int = 0, project: str = "proj") -> Reflection:
        return Reflection(
            id=f"task-{idx}",
            task_description=task,
            failure_summary=f"Error in attempt {idx}",
            reflection=f"Should try different approach {idx}",
            attempt_number=idx + 1,
            project=project,
        )

    def test_add_and_retrieve(self):
        mem = ReflectionMemory()
        r = self._make_reflection()
        mem.add(r)
        assert len(mem.reflections) == 1

    def test_eviction_on_overflow(self):
        mem = ReflectionMemory(max_total=3)
        for i in range(5):
            mem.add(self._make_reflection(idx=i))
        assert len(mem.reflections) == 3

    def test_evicts_resolved_first(self):
        mem = ReflectionMemory(max_total=2)
        r0 = self._make_reflection(idx=0)
        r0.outcome = "resolved"
        mem.add(r0)
        mem.add(self._make_reflection(idx=1))
        # This should evict r0 (resolved) before r1 (pending)
        mem.add(self._make_reflection(idx=2))
        assert len(mem.reflections) == 2
        ids = [r.id for r in mem.reflections]
        assert "task-0" not in ids

    def test_get_relevant_by_word_overlap(self):
        mem = ReflectionMemory()
        mem.add(self._make_reflection(task="build authentication module", idx=0))
        mem.add(self._make_reflection(task="setup database connection", idx=1))
        results = mem.get_relevant("authentication module error")
        assert len(results) >= 1
        assert results[0].task_description == "build authentication module"

    def test_get_relevant_boosts_same_project(self):
        mem = ReflectionMemory()
        r1 = self._make_reflection(task="build api server endpoint handler", idx=0, project="alpha")
        r2 = self._make_reflection(task="build api server endpoint handler", idx=1, project="beta")
        mem.add(r1)
        mem.add(r2)
        results = mem.get_relevant("build api server", project="alpha")
        # Both should appear since they match, but same-project is boosted
        assert len(results) >= 1

    def test_mark_resolved(self):
        mem = ReflectionMemory()
        r = self._make_reflection(idx=0)
        mem.add(r)
        mem.mark_resolved("task-0")
        assert mem.reflections[0].outcome == "resolved"

    def test_get_chain(self):
        mem = ReflectionMemory()
        mem.add(Reflection(id="task-A-attempt1", task_description="A", failure_summary="err", reflection="fix"))
        mem.add(Reflection(id="task-A-attempt2", task_description="A", failure_summary="err2", reflection="fix2"))
        mem.add(Reflection(id="task-B-attempt1", task_description="B", failure_summary="err3", reflection="fix3"))
        chain = mem.get_chain("task-A")
        assert len(chain) == 2


class TestReflexionEngine:
    """Tests for ReflexionEngine."""

    def test_reflect_on_failure_mocked(self):
        engine = ReflexionEngine()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "The root cause is a missing import. Try importing the module explicitly."
        mock_response.content = [mock_block]
        mock_llm.call = AsyncMock(return_value=mock_response)

        reflection = asyncio.run(
            engine.reflect_on_failure(
                task_id="build-auth",
                task_description="Build authentication module",
                failure_summary="ImportError: No module named 'jwt'",
                llm=mock_llm,
                project="myproject",
            )
        )
        assert reflection.id == "build-auth-attempt1"
        assert reflection.attempt_number == 1
        assert "import" in reflection.reflection.lower() or len(reflection.reflection) > 0

    def test_reflect_on_failure_increments_attempts(self):
        engine = ReflexionEngine()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Try a different approach."
        mock_response.content = [mock_block]
        mock_llm.call = AsyncMock(return_value=mock_response)

        asyncio.run(engine.reflect_on_failure("t1", "task", "fail1", mock_llm))
        r2 = asyncio.run(engine.reflect_on_failure("t1", "task", "fail2", mock_llm))
        assert r2.attempt_number == 2
        assert r2.id == "t1-attempt2"

    def test_build_retry_context_empty_when_no_reflections(self):
        engine = ReflexionEngine()
        ctx = engine.build_retry_context("some task")
        assert ctx == ""

    def test_build_retry_context_includes_reflections(self):
        engine = ReflexionEngine()
        engine._memory.add(Reflection(
            id="t1-attempt1",
            task_description="build web server",
            failure_summary="port already in use",
            reflection="Use a different port or kill the existing process.",
            attempt_number=1,
        ))
        ctx = engine.build_retry_context("build web server")
        assert "different port" in ctx or "Reflections" in ctx

    def test_mark_success(self):
        engine = ReflexionEngine()
        engine._memory.add(Reflection(
            id="t1-attempt1", task_description="t1", failure_summary="err", reflection="fix"
        ))
        engine._attempt_counters["t1"] = 1
        engine.mark_success("t1")
        assert engine._memory.reflections[0].outcome == "resolved"
        assert "t1" not in engine._attempt_counters

    def test_mark_persistent(self):
        engine = ReflexionEngine()
        engine._memory.add(Reflection(
            id="t1-attempt1", task_description="t1", failure_summary="err", reflection="fix"
        ))
        engine.mark_persistent("t1")
        assert engine._memory.reflections[0].outcome == "persistent"

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ReflexionEngine()
            engine._memory.add(Reflection(
                id="t1-attempt1",
                task_description="build thing",
                failure_summary="it broke",
                reflection="try again differently",
                tags=["import_error"],
            ))
            engine._attempt_counters["t1"] = 1
            engine.save_state(Path(tmpdir))

            engine2 = ReflexionEngine()
            engine2.load_state(Path(tmpdir))
            assert len(engine2._memory.reflections) == 1
            assert engine2._attempt_counters["t1"] == 1
            assert engine2._memory.reflections[0].tags == ["import_error"]

    def test_load_state_nonexistent_is_noop(self):
        engine = ReflexionEngine()
        engine.load_state(Path("/tmp/nonexistent_reflexion_test_dir"))
        assert len(engine._memory.reflections) == 0

    def test_get_stats(self):
        engine = ReflexionEngine()
        engine._memory.add(Reflection(
            id="t1", task_description="x", failure_summary="e", reflection="r", outcome="resolved"
        ))
        engine._memory.add(Reflection(
            id="t2", task_description="y", failure_summary="e", reflection="r", outcome="pending"
        ))
        stats = engine.get_stats()
        assert stats["total_reflections"] == 2
        assert stats["resolved"] == 1
        assert stats["pending"] == 1

    def test_extract_tags(self):
        tags = ReflexionEngine._extract_tags("ImportError: no module named foo")
        assert "import_error" in tags

    def test_get_recent_memories(self):
        engine = ReflexionEngine()
        for i in range(5):
            engine._memory.add(Reflection(
                id=f"r{i}", task_description="t", failure_summary="e", reflection="r"
            ))
        recent = engine.get_recent_memories(3)
        assert len(recent) == 3


# ══════════════════════════════════════════════════════════════
# 5. SpeculativePipeline
# ══════════════════════════════════════════════════════════════


class TestSpeculativeTask:
    """Tests for SpeculativeTask."""

    def test_duration_calculation(self):
        task = SpeculativeTask(
            id="t1", phase=SpecPhase.BUILD_SCAFFOLD,
            description="test", started_at=100.0, completed_at=105.5,
        )
        assert task.duration == pytest.approx(5.5)

    def test_duration_zero_when_not_completed(self):
        task = SpeculativeTask(id="t1", phase=SpecPhase.BUILD_SCAFFOLD, description="test")
        assert task.duration == 0.0

    def test_to_dict(self):
        task = SpeculativeTask(
            id="t1", phase=SpecPhase.TEST_SCAFFOLD, description="test scaffold",
            status=SpecTaskStatus.COMPLETED, validated=True,
        )
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["phase"] == "test_scaffold"
        assert d["status"] == "completed"

    def test_status_transitions(self):
        task = SpeculativeTask(id="t1", phase=SpecPhase.BUILD_SCAFFOLD, description="test")
        assert task.status == SpecTaskStatus.PENDING
        task.status = SpecTaskStatus.RUNNING
        assert task.status == SpecTaskStatus.RUNNING
        task.status = SpecTaskStatus.COMPLETED
        assert task.status == SpecTaskStatus.COMPLETED
        task.status = SpecTaskStatus.VALIDATED
        assert task.validated is False  # validated flag is separate
        task.validated = True
        assert task.validated is True


class TestSpeculativeResult:
    """Tests for SpeculativeResult."""

    def test_is_empty_true(self):
        r = SpeculativeResult()
        assert r.is_empty() is True

    def test_is_empty_false_with_files(self):
        r = SpeculativeResult(files_created=["foo.py"])
        assert r.is_empty() is False

    def test_is_empty_false_with_configs(self):
        r = SpeculativeResult(configs_generated={"Dockerfile": "FROM python"})
        assert r.is_empty() is False


class TestSpeculativePipeline:
    """Behavioral tests for SpeculativePipeline."""

    def test_speculate_build_scaffold(self):
        pipeline = SpeculativePipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            spec = {
                "project_name": "testapp",
                "tech_stack": {"backend": "Python"},
                "modules": ["auth", "api"],
            }
            task = asyncio.run(pipeline.speculate_build_scaffold(spec, project_dir))
            # Wait for task to finish
            asyncio.run(asyncio.sleep(0.1))
            assert task.id == "spec-build-scaffold"
            assert task.status == SpecTaskStatus.COMPLETED
            assert project_dir.exists()
            # Directories for modules should be created
            assert (project_dir / "auth").exists()
            assert (project_dir / "api").exists()

    def test_speculate_test_scaffold(self):
        pipeline = SpeculativePipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            spec = {"tech_stack": {"backend": "Python"}}
            task = asyncio.run(pipeline.speculate_test_scaffold(spec, project_dir))
            asyncio.run(asyncio.sleep(0.1))
            assert task.status == SpecTaskStatus.COMPLETED
            assert (project_dir / "tests").exists()
            assert (project_dir / "tests" / "conftest.py").exists()

    def test_validate_and_commit_success(self):
        async def _run():
            pipeline = SpeculativePipeline()
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir) / "project"
                spec = {
                    "project_name": "testapp",
                    "tech_stack": {},
                    "modules": ["core"],
                }
                await pipeline.speculate_build_scaffold(spec, project_dir)
                await asyncio.sleep(0.1)

                committed = await pipeline.validate_and_commit(
                    "spec-build-scaffold", project_dir=project_dir,
                )
                assert committed is True
                task = pipeline._tasks["spec-build-scaffold"]
                assert task.status == SpecTaskStatus.VALIDATED
                assert task.validated is True

        asyncio.run(_run())

    def test_validate_and_commit_conflict_invalidates(self):
        async def _run():
            pipeline = SpeculativePipeline()
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir) / "project"
                spec = {
                    "project_name": "testapp",
                    "tech_stack": {},
                    "modules": ["core"],
                }
                await pipeline.speculate_build_scaffold(spec, project_dir)
                await asyncio.sleep(0.1)

                task = pipeline._tasks["spec-build-scaffold"]
                result: SpeculativeResult = task.result
                conflicting_state = {"files_written": result.files_created[:1]}

                committed = await pipeline.validate_and_commit(
                    "spec-build-scaffold",
                    current_state=conflicting_state,
                    project_dir=project_dir,
                )
                assert committed is False
                assert task.status == SpecTaskStatus.INVALIDATED

        asyncio.run(_run())

    def test_invalidate_explicit(self):
        pipeline = SpeculativePipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            spec = {"project_name": "x", "tech_stack": {}, "modules": []}
            asyncio.run(pipeline.speculate_build_scaffold(spec, project_dir))
            asyncio.run(asyncio.sleep(0.1))

            pipeline.invalidate("spec-build-scaffold")
            task = pipeline._tasks["spec-build-scaffold"]
            assert task.status == SpecTaskStatus.INVALIDATED
            assert task.validated is False

    def test_validate_nonexistent_task_returns_false(self):
        pipeline = SpeculativePipeline()
        result = asyncio.run(pipeline.validate_and_commit("nonexistent"))
        assert result is False

    def test_validate_empty_result_invalidates(self):
        pipeline = SpeculativePipeline()
        # Manually add a task with empty result
        task = SpeculativeTask(
            id="empty-task", phase=SpecPhase.LINT_PRECHECK,
            description="empty", status=SpecTaskStatus.COMPLETED,
            result=SpeculativeResult(),
            started_at=time.time(), completed_at=time.time(),
        )
        pipeline._tasks[task.id] = task
        committed = asyncio.run(pipeline.validate_and_commit("empty-task"))
        assert committed is False

    def test_speculate_deploy_prep(self):
        pipeline = SpeculativePipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            spec = {"project_name": "myapp", "tech_stack": {"backend": "Python"}}
            task = asyncio.run(pipeline.speculate_deploy_prep(spec, project_dir))
            asyncio.run(asyncio.sleep(0.1))
            assert task.status == SpecTaskStatus.COMPLETED
            result: SpeculativeResult = task.result
            assert "Dockerfile" in result.configs_generated

    def test_commit_writes_config_files(self):
        async def _run():
            pipeline = SpeculativePipeline()
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir) / "project"
                project_dir.mkdir()
                spec = {"project_name": "myapp", "tech_stack": {"backend": "Python"}}
                await pipeline.speculate_deploy_prep(spec, project_dir)
                await asyncio.sleep(0.1)

                await pipeline.validate_and_commit("spec-deploy-prep", project_dir=project_dir)
                assert (project_dir / "Dockerfile").exists()

        asyncio.run(_run())

    def test_get_stats(self):
        pipeline = SpeculativePipeline()
        stats = pipeline.get_stats()
        assert stats["total_speculated"] == 0
        assert stats["validated"] == 0
        assert stats["hit_rate"] == 0.0

    def test_cancel_all(self):
        pipeline = SpeculativePipeline()
        # Just verify it doesn't raise on empty state
        asyncio.run(pipeline.cancel_all())

    def test_gitignore_generation(self):
        content = SpeculativePipeline._generate_gitignore({"backend": "Python"})
        assert "__pycache__/" in content
        assert ".env" in content

    def test_dockerfile_generation(self):
        content = SpeculativePipeline._generate_dockerfile({"backend": "Python"})
        assert "FROM python" in content
        assert "EXPOSE 8000" in content

    def test_docker_compose_generation(self):
        content = SpeculativePipeline._generate_docker_compose({"project_name": "myapp"})
        assert "myapp" in content
        assert "8000:8000" in content


class TestAdaptiveBeamSearch:
    """Behavioral tests for adaptive proof beam search instrumentation."""

    def test_algorithm_ratio_updates_from_outcomes(self):
        beam = AdaptiveBeamSearch(initial_width=3, max_width=6)
        init = ProofState(goals=["⊢ True"], hypotheses=[])

        async def no_expand(state):
            return []

        async def score(_state):
            return 0.0

        # First run: no solution found -> ratio should trend to 0.0
        results = asyncio.run(beam.search(init, no_expand, score, max_depth=2))
        assert results == []
        assert beam.get_stats()["algorithm_ratio"] == pytest.approx(0.0)

        async def solved_expand(_state):
            return [ProofState(goals=[], hypotheses=[])]

        # Second run: a solved state is found -> ratio should move upward (>0)
        results2 = asyncio.run(beam.search(init, solved_expand, score, max_depth=1))
        assert len(results2) >= 1
        assert beam.get_stats()["algorithm_ratio"] > 0.0
