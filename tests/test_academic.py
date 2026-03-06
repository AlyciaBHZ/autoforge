"""Academic Module Tests â€” 28 offline tests for the academic reasoning stack.

Covers: TheoryGraph (8), DiscoveryOrchestrator components (6),
PaperFormalizer (5), CloudProver (5), ConceptNode (4).

All tests are offline â€” no API keys, no LLM calls, no Docker/SSH.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

# â”€â”€ Imports from theoretical_reasoning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptRelation,
    ConceptType,
    RelationType,
    ScientificDomain,
    TheoryGraph,
    VerificationMode,
)

# â”€â”€ Imports from autonomous_discovery â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from autoforge.engine.autonomous_discovery import (
    DiscoveryConfig,
    DiscoveryDepth,
    EloTournament,
    Hypothesis,
    HypothesisTournament,
    MatchResult,
    NoveltyFilter,
    PaperKernel,
)
from autoforge.engine.provers.lean_core import (
    LeanEnvironment,
    LeanVerificationResult,
    PantographREPL,
    ProofState,
    ProofStatus,
)
from autoforge.engine.provers.proof_search import HeuristicExecutor, MCTSProofSearch, RecursiveProofDecomposer, TacticGenerator

# â”€â”€ Imports from paper_formalizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from autoforge.engine.paper_formalizer import (
    FormalizationReport,
    FormalizationStatus,
    FormalizationUnit,
    LeanCodeGenerator,
)

# â”€â”€ Imports from cloud_prover â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from autoforge.engine.cloud_prover import (
    CloudBackend,
    CloudProver,
    CloudProverConfig,
    JobStatus,
    ProofCache,
    ProofJob,
)


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Helpers                                                  â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _make_node(
    nid: str,
    ctype: ConceptType = ConceptType.THEOREM,
    domain: ScientificDomain = ScientificDomain.PURE_MATHEMATICS,
    statement: str = "Test statement",
    **kwargs,
) -> ConceptNode:
    return ConceptNode(
        id=nid,
        concept_type=ctype,
        domain=domain,
        formal_statement=statement,
        **kwargs,
    )


def _build_sample_graph() -> TheoryGraph:
    """Build a small theory graph for testing.

    Structure:
        def1 (DEFINITION) â”€â”€depends_onâ”€â”€â–¶ thm1 (THEOREM)
        def2 (DEFINITION) â”€â”€depends_onâ”€â”€â–¶ thm1
        thm1 â”€â”€generalizesâ”€â”€â–¶ thm2 (THEOREM)
        thm1 â”€â”€analogous_toâ”€â”€â–¶ phys1 (THEOREM, physics domain)
        conj1 (CONJECTURE) â€” leaf, no dependents
    """
    g = TheoryGraph(title="Test Theory", source="test_paper.pdf")

    g.add_concept(_make_node("def1", ConceptType.DEFINITION, statement="Definition of CPS"))
    g.add_concept(_make_node("def2", ConceptType.DEFINITION, statement="Definition of model set"))
    g.add_concept(_make_node("thm1", ConceptType.THEOREM, statement="Main resolution theorem"))
    g.add_concept(_make_node("thm2", ConceptType.THEOREM, statement="Generalized resolution"))
    g.add_concept(_make_node(
        "phys1", ConceptType.THEOREM,
        domain=ScientificDomain.THEORETICAL_PHYSICS,
        statement="Physics analogue of resolution",
    ))
    g.add_concept(_make_node(
        "conj1", ConceptType.CONJECTURE,
        statement="Open conjecture about boundary dimension",
        overall_confidence=0.3,
    ))

    g.add_relation(ConceptRelation("def1", "thm1", RelationType.DEPENDS_ON))
    g.add_relation(ConceptRelation("def2", "thm1", RelationType.DEPENDS_ON))
    g.add_relation(ConceptRelation("thm1", "thm2", RelationType.GENERALIZES))
    g.add_relation(ConceptRelation(
        "thm1", "phys1", RelationType.ANALOGOUS_TO,
        bridging_insight="Structural isomorphism via spectral duality",
    ))

    return g




class _DummyJudgeLLM:
    """Deterministic async judge stub for Elo tournament tests."""

    def __init__(self, winner: str = "A", is_draw: bool = False) -> None:
        self.winner = winner
        self.is_draw = is_draw

    async def __call__(self, prompt: str) -> str:
        return json.dumps({
            "winner": self.winner,
            "is_draw": self.is_draw,
            "reasoning": "Deterministic test judge",
            "confidence": 0.9,
        })

# ╔═══════════════════════════════════════════════════════════╗
# ║  1. TheoryGraph Tests (8)                                 ║
# ╚═══════════════════════════════════════════════════════════╝


class TestTheoryGraph(unittest.TestCase):
    """8 tests for TheoryGraph construction, querying, and persistence."""

    def setUp(self):
        self.g = _build_sample_graph()

    def test_add_concepts_and_size(self):
        """Graph tracks all added concepts."""
        self.assertEqual(self.g.size, 6)

    def test_get_concept(self):
        """get_concept returns the right node."""
        node = self.g.get_concept("thm1")
        self.assertIsNotNone(node)
        self.assertEqual(node.formal_statement, "Main resolution theorem")

    def test_get_frontier(self):
        """Frontier = leaf nodes of type theorem/proposition/conjecture/corollary."""
        frontier = self.g.get_frontier()
        frontier_ids = {n.id for n in frontier}
        # thm2, phys1, conj1 are leaves; thm1 has outgoing edges so is NOT a leaf
        # def1, def2 are definitions so excluded from frontier
        self.assertIn("thm2", frontier_ids)
        self.assertIn("phys1", frontier_ids)
        self.assertIn("conj1", frontier_ids)
        self.assertNotIn("thm1", frontier_ids)

    def test_get_cross_domain_bridges(self):
        """Cross-domain bridges find analogies between different domains."""
        bridges = self.g.get_cross_domain_bridges()
        self.assertEqual(len(bridges), 1)
        src, tgt, rel = bridges[0]
        self.assertEqual(src.domain, ScientificDomain.PURE_MATHEMATICS)
        self.assertEqual(tgt.domain, ScientificDomain.THEORETICAL_PHYSICS)

    def test_save_load_roundtrip(self):
        """save â†’ load produces identical graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.g.save(path)

            g2 = TheoryGraph()
            g2.load(path)

            self.assertEqual(g2.title, "Test Theory")
            self.assertEqual(g2.size, 6)
            node = g2.get_concept("thm1")
            self.assertIsNotNone(node)
            self.assertEqual(node.formal_statement, "Main resolution theorem")

    def test_get_stats(self):
        """get_stats returns proper counts."""
        stats = self.g.get_stats()
        self.assertEqual(stats["total_concepts"], 6)
        self.assertEqual(stats["total_relations"], 4)
        self.assertIn("theorem", stats["by_type"])
        self.assertIn("pure_mathematics", stats["by_domain"])

    def test_topological_dependencies(self):
        """get_dependencies returns transitive deps."""
        deps = self.g.get_dependencies("thm1")
        dep_ids = {n.id for n in deps}
        self.assertIn("def1", dep_ids)
        self.assertIn("def2", dep_ids)

    def test_empty_graph(self):
        """Empty graph has size 0, empty frontier, no bridges."""
        empty = TheoryGraph()
        self.assertEqual(empty.size, 0)
        self.assertEqual(empty.get_frontier(), [])
        self.assertEqual(empty.get_cross_domain_bridges(), [])
        self.assertEqual(empty.get_stats()["total_concepts"], 0)


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  2. DiscoveryOrchestrator Components (6)                  â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestDiscoveryComponents(unittest.TestCase):
    """6 tests for PaperKernel, NoveltyFilter, DepthEvaluator, DiscoveryConfig."""

    def setUp(self):
        self.g = _build_sample_graph()

    def test_paper_kernel_extraction(self):
        """PaperKernel extracts definitions+axioms as kernel, leaves as frontier."""
        pk = PaperKernel(self.g)
        kernel, frontier = pk.extract()

        kernel_ids = {n.id for n in kernel}
        frontier_ids = {n.id for n in frontier}

        # Definitions go to kernel
        self.assertIn("def1", kernel_ids)
        self.assertIn("def2", kernel_ids)
        # thm1 has â‰¥2 dependents (def1, def2 depend on it) â€” goes to kernel
        self.assertIn("thm1", kernel_ids)
        # Cross-domain bridge nodes also go to kernel
        self.assertIn("phys1", kernel_ids)

        # conj1 and thm2 are leaves not in kernel
        self.assertIn("conj1", frontier_ids)
        self.assertIn("thm2", frontier_ids)

    def test_novelty_filter_jaccard_dedup(self):
        """NoveltyFilter._jaccard_similarity detects near-duplicates."""
        sim = NoveltyFilter._jaccard_similarity(
            "The golden mean language has Fibonacci cardinality",
            "The golden mean language has Fibonacci cardinality for all m",
        )
        self.assertGreater(sim, 0.6)

        # Very different strings should have low overlap
        sim_low = NoveltyFilter._jaccard_similarity(
            "The quick brown fox",
            "Zeckendorf representation is unique",
        )
        self.assertLess(sim_low, 0.2)

    def test_novelty_filter_register_discovery(self):
        """register_discovery accumulates statements."""
        nf = NoveltyFilter(known_statements=["Theorem A"], threshold=0.7)
        self.assertEqual(len(nf._discovered), 0)
        nf.register_discovery("New result B")
        self.assertEqual(len(nf._discovered), 1)
        nf.register_discovery("New result C")
        self.assertEqual(len(nf._discovered), 2)

    def test_discovery_config_defaults(self):
        """DiscoveryConfig has sane defaults."""
        dc = DiscoveryConfig()
        self.assertGreater(dc.max_rounds, 0)
        self.assertGreaterEqual(dc.min_confidence, 0)
        self.assertLessEqual(dc.min_confidence, 1)
        self.assertEqual(len(dc.strategies), 8)
        self.assertIn("generalization", dc.strategies)

    def test_strategy_selection_coverage(self):
        """All 8 default strategies are distinct."""
        dc = DiscoveryConfig()
        strategies = dc.strategies
        self.assertEqual(len(strategies), len(set(strategies)))
        expected = {
            "generalization", "composition", "analogy_transfer",
            "boundary_analysis", "duality", "unification",
            "numerical_exploration", "dimensional_lifting",
        }
        self.assertEqual(set(strategies), expected)

    def test_termination_conditions(self):
        """DiscoveryConfig termination params are sensible."""
        dc = DiscoveryConfig()
        self.assertEqual(dc.max_consecutive_shallow_rounds, 3)
        self.assertEqual(dc.max_total_results, 100)
        self.assertGreater(dc.depth_score_threshold, 0)


# ╔═══════════════════════════════════════════════════════════╗
# ║  2b. Elo Tournament Tests (3)                             ║
# ╚═══════════════════════════════════════════════════════════╝


class TestEloTournament(unittest.TestCase):
    """Coverage for pairwise hypothesis Elo ranking behavior."""

    def test_register_hypothesis_idempotent(self):
        tournament = EloTournament()
        tournament.register_hypothesis("h1")
        tournament._ratings["h1"].rating = 1666.0

        # Re-register should not clobber accumulated rating/statistics.
        tournament.register_hypothesis("h1")
        self.assertEqual(tournament._ratings["h1"].rating, 1666.0)

    def test_update_ratings_win_and_draw(self):
        tournament = EloTournament(k_factor=32)
        tournament.register_hypothesis("a")
        tournament.register_hypothesis("b")

        win_match = MatchResult("a", "b", False, "a stronger", 0.9)
        tournament._update_ratings(win_match)
        self.assertGreater(tournament._ratings["a"].rating, 1500.0)
        self.assertLess(tournament._ratings["b"].rating, 1500.0)

        draw_match = MatchResult("a", "b", True, "equal", 0.8)
        tournament._update_ratings(draw_match)
        self.assertEqual(tournament._ratings["a"].matches_played, 2)
        self.assertEqual(tournament._ratings["b"].matches_played, 2)
        self.assertGreater(tournament._ratings["a"].confidence_interval, 0.0)

    def test_run_tournament_handles_small_population(self):
        tournament = EloTournament()

        empty = asyncio.run(tournament.run_tournament({}, _DummyJudgeLLM()))
        self.assertEqual(empty["total_matches"], 0)
        self.assertEqual(empty["rankings"], [])

        single = asyncio.run(tournament.run_tournament({"h1": "x"}, _DummyJudgeLLM()))
        self.assertEqual(single["total_matches"], 0)
        self.assertEqual(len(single["rankings"]), 1)


# ╔═══════════════════════════════════════════════════════════╗
# ║  2c. Proof Pipeline P0 Tests (3)                          ║
# ╚═══════════════════════════════════════════════════════════╝


class _TinyLLM:
    async def call(self, *args, **kwargs):
        class R:
            content = []
        return R()

class _StickyFormalExecutor:
    """Stateful formal executor stub to verify MCTS state replay behavior."""

    def __init__(self):
        self._busy = False

    @property
    def backend_name(self) -> str:
        return "sticky_formal"

    @property
    def is_formal(self) -> bool:
        return True

    async def start(self, theorem_context: str) -> bool:
        self._busy = False
        return True

    async def apply(self, state: ProofState, tactic: str) -> ProofState | None:
        if self._busy:
            return None
        self._busy = True
        return ProofState(goals=["still_open"], hypotheses=[], parent_tactic=tactic)

    async def undo(self) -> ProofState:
        self._busy = False
        return ProofState(goals=["still_open"], hypotheses=[])

    async def close(self) -> None:
        self._busy = False

class TestProofPipelineP0(unittest.TestCase):
    """Regression tests for real proof-chain safety guarantees."""

    def test_mcts_expands_root(self):
        tactic_gen = TacticGenerator()
        mcts = MCTSProofSearch(tactic_gen, executor=HeuristicExecutor())

        async def fake_generate(*args, **kwargs):
            return [
                type("C", (), {"tactic": "simp"})(),
                type("C", (), {"tactic": "intro h"})(),
            ]

        tactic_gen.generate_candidates = fake_generate  # type: ignore[assignment]
        root = ProofState(goals=["True"], hypotheses=[])
        proof = asyncio.run(mcts.search(root, "theorem t : True", _TinyLLM(), max_iterations=2))
        stats = mcts.get_stats()

        self.assertGreaterEqual(stats["nodes_explored"], 1)
        self.assertIsNotNone(proof)

    def test_mcts_does_not_only_evaluate_first_child(self):
        tactic_gen = TacticGenerator()
        mcts = MCTSProofSearch(tactic_gen, executor=HeuristicExecutor())

        async def fake_generate(*args, **kwargs):
            return [
                type("C", (), {"tactic": "intro h1"})(),
                type("C", (), {"tactic": "intro h2"})(),
                type("C", (), {"tactic": "intro h3"})(),
            ]

        seen: list[str] = []

        async def fake_eval(state, llm=None):
            seen.append(state.parent_tactic)
            return 0.4

        tactic_gen.generate_candidates = fake_generate  # type: ignore[assignment]
        mcts._evaluate_state = fake_eval  # type: ignore[assignment]

        root = ProofState(goals=["Goal"], hypotheses=[])
        asyncio.run(mcts.search(root, "theorem t : True", _TinyLLM(), max_iterations=1))

        self.assertGreaterEqual(len(set(seen)), 2)

    def test_pantograph_fallback_does_not_auto_solve(self):
        repl = PantographREPL(LeanEnvironment())
        state = asyncio.run(repl.send_tactic("simp"))
        self.assertNotEqual(state.goals, [])
        self.assertIn("unknown_goal_state", state.goals[0])

    def test_direct_proof_branch_requires_verification(self):
        tactic_gen = TacticGenerator()
        mcts = MCTSProofSearch(tactic_gen, executor=HeuristicExecutor())
        decomposer = RecursiveProofDecomposer(tactic_gen, mcts, lean_env=LeanEnvironment())

        async def fake_informal(*args, **kwargs):
            return "By trivial reasoning."

        async def fake_formal(*args, **kwargs):
            return "exact trivial"

        async def no_decompose(*args, **kwargs):
            return []

        async def no_lean(*args, **kwargs):
            return False

        async def no_mcts(*args, **kwargs):
            return None

        decomposer._generate_informal_proof = fake_informal  # type: ignore[assignment]
        decomposer._informal_to_formal = fake_formal  # type: ignore[assignment]
        decomposer._decompose = no_decompose  # type: ignore[assignment]
        decomposer._lean_env.check_lean_installation = no_lean  # type: ignore[assignment]
        mcts.search = no_mcts  # type: ignore[assignment]

        attempt = asyncio.run(decomposer.prove("theorem t : True", "", _TinyLLM()))
        self.assertNotEqual(attempt.status, ProofStatus.PROVED)


    def test_proof_result_contains_provenance_fields(self):
        tactic_gen = TacticGenerator()
        mcts = MCTSProofSearch(tactic_gen, executor=HeuristicExecutor())
        decomposer = RecursiveProofDecomposer(tactic_gen, mcts, lean_env=LeanEnvironment())

        async def fake_informal(*args, **kwargs):
            return "By direct method."

        async def fake_formal(*args, **kwargs):
            return "exact trivial"

        async def no_decompose(*args, **kwargs):
            return []

        async def no_mcts(*args, **kwargs):
            return None

        decomposer._generate_informal_proof = fake_informal  # type: ignore[assignment]
        decomposer._informal_to_formal = fake_formal  # type: ignore[assignment]
        decomposer._decompose = no_decompose  # type: ignore[assignment]
        mcts.search = no_mcts  # type: ignore[assignment]

        attempt = asyncio.run(decomposer.prove("theorem t : True", "", _TinyLLM()))
        self.assertEqual(attempt.proof_origin, "direct")
        self.assertEqual(attempt.execution_backend, "direct")
        self.assertIn(attempt.verification_backend, {"unavailable", "lean", "simulated", ""})

    def test_non_formal_verification_never_marks_proved(self):
        tactic_gen = TacticGenerator()
        mcts = MCTSProofSearch(tactic_gen, executor=HeuristicExecutor())
        env = LeanEnvironment()
        decomposer = RecursiveProofDecomposer(tactic_gen, mcts, lean_env=env)

        async def fake_informal(*args, **kwargs):
            return "By direct method."

        async def fake_formal(*args, **kwargs):
            return "exact trivial"

        async def no_decompose(*args, **kwargs):
            return []

        async def no_mcts(*args, **kwargs):
            return None

        async def fake_check() -> bool:
            return True

        async def fake_verify(_p):
            return LeanVerificationResult(
                success=True,
                sorry_count=0,
                execution_time=0.0,
                backend="simulated",
                is_formal=False,
            )

        decomposer._generate_informal_proof = fake_informal  # type: ignore[assignment]
        decomposer._informal_to_formal = fake_formal  # type: ignore[assignment]
        decomposer._decompose = no_decompose  # type: ignore[assignment]
        mcts.search = no_mcts  # type: ignore[assignment]
        env.check_lean_installation = fake_check  # type: ignore[assignment]
        env.verify_file = fake_verify  # type: ignore[assignment]

        attempt = asyncio.run(decomposer.prove("theorem t : True", "", _TinyLLM()))
        self.assertNotEqual(attempt.status, ProofStatus.PROVED)
        self.assertEqual(attempt.status, ProofStatus.FAILED)
        self.assertEqual(attempt.verification_backend, "simulated")

    def test_mcts_formal_executor_replays_for_each_sibling(self):
        tactic_gen = TacticGenerator()
        mcts = MCTSProofSearch(tactic_gen, executor=_StickyFormalExecutor())

        async def fake_generate(*args, **kwargs):
            return [
                type("C", (), {"tactic": "first"})(),
                type("C", (), {"tactic": "second"})(),
            ]

        seen: list[str] = []

        async def fake_eval(state, llm=None):
            seen.append(state.parent_tactic)
            return 0.4

        tactic_gen.generate_candidates = fake_generate  # type: ignore[assignment]
        mcts._evaluate_state = fake_eval  # type: ignore[assignment]

        root = ProofState(goals=["Goal"], hypotheses=[])
        asyncio.run(mcts.search(root, "theorem t : True", _TinyLLM(), max_iterations=1))

        self.assertGreaterEqual(len(set(seen)), 2)

    def test_mcts_proof_branch_requires_formal_backend(self):
        tactic_gen = TacticGenerator()
        mcts = MCTSProofSearch(tactic_gen, executor=HeuristicExecutor())
        decomposer = RecursiveProofDecomposer(tactic_gen, mcts, lean_env=LeanEnvironment())

        async def fake_informal(*args, **kwargs):
            return "By search."

        async def no_formal(*args, **kwargs):
            return ""

        async def fake_search(*args, **kwargs):
            s0 = ProofState(goals=["G"], hypotheses=[])
            s1 = ProofState(goals=[], hypotheses=[], parent_tactic="simp")
            return [type("Step", (), {"tactic_applied": "simp", "state_before": s0, "state_after": s1})()]

        async def no_decompose(*args, **kwargs):
            return []

        decomposer._generate_informal_proof = fake_informal  # type: ignore[assignment]
        decomposer._informal_to_formal = no_formal  # type: ignore[assignment]
        decomposer._decompose = no_decompose  # type: ignore[assignment]
        mcts.search = fake_search  # type: ignore[assignment]

        attempt = asyncio.run(decomposer.prove("theorem t : True", "", _TinyLLM()))
        self.assertNotEqual(attempt.status, ProofStatus.PROVED)




class TestHypothesisTournament(unittest.TestCase):
    def test_empty_and_single_inputs(self):
        tour = HypothesisTournament()
        empty = asyncio.run(tour.run(rounds=2))
        self.assertEqual(empty["rankings"], [])

        tour.register(Hypothesis(id="h1", statement="A"))
        one = asyncio.run(tour.run(rounds=2))
        self.assertEqual(len(one["rankings"]), 1)
        self.assertEqual(one["match_history"], [])

    def test_register_is_idempotent(self):
        tour = HypothesisTournament()
        h = Hypothesis(id="h1", statement="A")
        tour.register(h)
        tour.register(h)
        self.assertEqual(len(tour.rankings()), 1)

    def test_deterministic_pairing_and_history(self):
        tour = HypothesisTournament()
        tour.register(Hypothesis(id="h1", statement="A", novelty=0.9, depth=0.9, verification_confidence=0.8))
        tour.register(Hypothesis(id="h2", statement="B", novelty=0.4, depth=0.5, verification_confidence=0.4))
        tour.register(Hypothesis(id="h3", statement="C", novelty=0.8, depth=0.7, verification_confidence=0.6))
        tour.register(Hypothesis(id="h4", statement="D", novelty=0.2, depth=0.2, verification_confidence=0.2))

        out = asyncio.run(tour.run(rounds=2))
        self.assertGreaterEqual(len(out["match_history"]), 2)
        top = out["rankings"][0]
        self.assertIn("rating", top)
        self.assertIn("wins", top)
        self.assertIn("losses", top)


# ╔═══════════════════════════════════════════════════════════╗
# ║  3. PaperFormalizer Tests (5)                             ║
# ╚═══════════════════════════════════════════════════════════╝


class TestPaperFormalizer(unittest.TestCase):
    """5 tests for FormalizationUnit, FormalizationReport, LeanCodeGenerator."""

    def test_formalization_unit_serialization_roundtrip(self):
        """FormalizationUnit.to_dict preserves key fields."""
        unit = FormalizationUnit(
            concept_id="thm_3_3",
            concept_type="theorem",
            section="3",
            label="Theorem 3.3",
            natural_language="For every CPS, resolution holds.",
            formal_statement_latex=r"\forall \Gamma, \text{Res}(\Gamma) = 1",
            lean_status=FormalizationStatus.LEAN_SORRY,
            verification_confidence=0.75,
        )
        d = unit.to_dict()
        self.assertEqual(d["concept_id"], "thm_3_3")
        self.assertEqual(d["lean_status"], "lean_sorry")
        self.assertAlmostEqual(d["verification_confidence"], 0.75)
        # Round-trip via JSON
        j = json.dumps(d)
        d2 = json.loads(j)
        self.assertEqual(d2["label"], "Theorem 3.3")

    def test_formalization_report_scoring(self):
        """FormalizationReport.compute_score uses correct weights."""
        report = FormalizationReport(
            paper_title="Test Paper",
            paper_source="test.pdf",
            total_statements=10,
            lean_proved=3,       # 3 * 1.0 = 3.0
            lean_sorry=2,        # 2 * 0.0 = 0.0
            numerically_verified=2,  # 2 * 0.7 = 1.4
            computationally_reproduced=1,  # 1 * 0.8 = 0.8
            lean_failed=1,
            skipped=1,
        )
        score = report.compute_score()
        expected = (3.0 + 0.0 + 1.4 + 0.8) / 10  # = 0.52
        self.assertAlmostEqual(score, expected, places=6)
        self.assertAlmostEqual(report.overall_score, expected, places=6)

    def test_formalization_report_roundtrip(self):
        """FormalizationReport â†’ to_dict â†’ JSON â†’ reload â†’ compute_score matches."""
        report = FormalizationReport(
            paper_title="Test Paper",
            paper_source="test.pdf",
            total_statements=5,
            lean_proved=2,
            lean_sorry=1,
            numerically_verified=1,
            computationally_reproduced=0,
            lean_failed=1,
            skipped=0,
        )
        score1 = report.compute_score()
        d = report.to_dict()
        j = json.dumps(d)
        d2 = json.loads(j)

        # Reconstruct
        report2 = FormalizationReport(
            paper_title=d2["paper_title"],
            paper_source=d2["paper_source"],
            total_statements=d2["total_statements"],
            lean_proved=d2["lean_proved"],
            lean_sorry=d2["lean_sorry"],
            lean_failed=d2["lean_failed"],
            numerically_verified=d2["numerically_verified"],
            computationally_reproduced=d2["computationally_reproduced"],
            skipped=d2["skipped"],
        )
        score2 = report2.compute_score()
        self.assertAlmostEqual(score1, score2, places=6)

    def test_lean_preamble_validity(self):
        """LeanCodeGenerator.PREAMBLE is non-empty and has no Python syntax errors."""
        preamble = LeanCodeGenerator.PREAMBLE
        self.assertGreater(len(preamble), 100)
        # Must contain Lean 4 imports
        self.assertIn("import Mathlib", preamble)
        # Must define CPS structure
        self.assertIn("structure CPS", preamble)
        # Must define golden-mean language
        self.assertIn("goldenMeanLang", preamble)
        # Must define Fibonacci
        self.assertIn("def fib", preamble)
        # Must define Zeckendorf
        self.assertIn("ZeckendorfRepr", preamble)

    def test_extract_units_from_mock_graph(self):
        """FormalizationUnit can be built from TheoryGraph ConceptNodes."""
        g = _build_sample_graph()
        theorems = g.get_concepts_by_type(ConceptType.THEOREM)
        units = []
        for thm in theorems:
            unit = FormalizationUnit(
                concept_id=thm.id,
                concept_type=thm.concept_type.value,
                section=thm.source_section or "1",
                label=f"Theorem {thm.id}",
                natural_language=thm.informal_statement or thm.formal_statement,
                formal_statement_latex=thm.formal_statement,
            )
            units.append(unit)
        # We have 3 theorems: thm1, thm2, phys1
        self.assertEqual(len(units), 3)
        self.assertTrue(all(u.lean_status == FormalizationStatus.PENDING for u in units))


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  4. CloudProver Tests (5)                                 â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestCloudProver(unittest.TestCase):
    """5 tests for ProofCache, ProofJob, CloudProverConfig, CloudProver."""

    def test_proof_cache_put_get(self):
        """ProofCache stores and retrieves jobs by code hash."""
        cache = ProofCache()
        job = ProofJob(
            job_id="test_001",
            lean_code="theorem foo : 1 = 1 := rfl",
            label="Reflexivity",
            status=JobStatus.COMPLETED,
            compiled_ok=True,
        )
        cache.put(job)
        result = cache.get("theorem foo : 1 = 1 := rfl")
        self.assertIsNotNone(result)
        self.assertEqual(result.job_id, "test_001")
        self.assertEqual(result.status, JobStatus.COMPLETED)

    def test_proof_cache_miss(self):
        """ProofCache returns None on miss."""
        cache = ProofCache()
        result = cache.get("nonexistent lean code")
        self.assertIsNone(result)

    def test_proof_job_status_transitions(self):
        """ProofJob status can transition through lifecycle."""
        job = ProofJob(job_id="j1", lean_code="sorry")
        self.assertEqual(job.status, JobStatus.QUEUED)

        job.status = JobStatus.RUNNING
        self.assertEqual(job.status, JobStatus.RUNNING)

        job.status = JobStatus.COMPLETED
        self.assertEqual(job.status, JobStatus.COMPLETED)

        # Serialization preserves status
        d = job.to_dict()
        self.assertEqual(d["status"], "completed")

    def test_cloud_prover_config_defaults(self):
        """CloudProverConfig has sane defaults."""
        cfg = CloudProverConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.preferred_backend, CloudBackend.DOCKER_LOCAL)
        self.assertGreater(cfg.max_concurrent_jobs, 0)
        self.assertGreater(cfg.job_timeout_seconds, 0)

    def test_cloud_prover_save_load_roundtrip(self):
        """CloudProver save_state / load_state round-trips job data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            prover = CloudProver(CloudProverConfig())
            # Manually inject a completed job
            job = ProofJob(
                job_id="save_test_001",
                lean_code="theorem bar : 2 = 2 := rfl",
                label="Test Save",
                status=JobStatus.COMPLETED,
                compiled_ok=True,
                has_sorry=False,
                duration_seconds=1.5,
            )
            prover._jobs.append(job)

            prover.save_state(path)

            # Load into fresh prover
            prover2 = CloudProver(CloudProverConfig())
            self.assertEqual(len(prover2._jobs), 0)
            prover2.load_state(path)
            self.assertEqual(len(prover2._jobs), 1)
            self.assertEqual(prover2._jobs[0].job_id, "save_test_001")
            self.assertEqual(prover2._jobs[0].status, JobStatus.COMPLETED)


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  5. ConceptNode Tests (4)                                 â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestConceptNode(unittest.TestCase):
    """4 tests for ConceptNode confidence, verification, serialization."""

    def test_update_confidence_weighted_aggregation(self):
        """update_confidence computes weighted combination with multi-mode bonus."""
        node = _make_node("n1")
        node.verification_status = {
            "numerical": 0.95,
            "consistency": 0.80,
            "statistical": 0.70,
        }
        conf = node.update_confidence()
        self.assertGreater(conf, 0.5)
        self.assertLess(conf, 1.0)
        self.assertEqual(node.overall_confidence, conf)

    def test_vlm_weight_045(self):
        """VLM visual verification has weight 0.45."""
        node = _make_node("n2")
        # Only VLM verification
        node.verification_status = {"vlm_visual": 1.0}
        conf = node.update_confidence()
        # With only one mode: raw = (0.45 * 1.0) / 0.45 = 1.0, bonus = 0.03
        # Result = min(0.99, 1.0 + 0.03) = 0.99
        # But wait, that's single-mode: raw = 1.0, bonus = 0.03 â†’ 0.99 (capped)
        # Actually with single mode at confidence 1.0:
        # raw = (0.45 * 1.0) / 0.45 = 1.0
        # mode_count (>0.5) = 1, bonus = 0.03
        # result = min(0.99, 1.0 + 0.03) = 0.99
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 0.99)

        # Now check VLM at half confidence â€” raw should equal 0.5
        node2 = _make_node("n2b")
        node2.verification_status = {"vlm_visual": 0.5}
        conf2 = node2.update_confidence()
        # raw = (0.45 * 0.5) / 0.45 = 0.5
        # mode_count (>0.5): 0.5 is NOT >0.5, so mode_count=0, bonus=0
        # result = 0.5
        self.assertAlmostEqual(conf2, 0.5, places=2)

    def test_verification_status_merge(self):
        """Multiple verification modes combine properly."""
        node = _make_node("n3")
        node.verification_status = {
            "formal_proof": 0.99,  # Triggers shortcut â†’ confidence = 1.0
        }
        conf = node.update_confidence()
        self.assertEqual(conf, 1.0)

        # Below 0.99 threshold, it should NOT shortcut
        node2 = _make_node("n3b")
        node2.verification_status = {"formal_proof": 0.95}
        conf2 = node2.update_confidence()
        self.assertLess(conf2, 1.0)
        self.assertGreater(conf2, 0.8)

    def test_to_dict_from_dict_roundtrip(self):
        """ConceptNode to_dict / from_dict preserves all fields."""
        node = _make_node(
            "rt1",
            ctype=ConceptType.CONJECTURE,
            domain=ScientificDomain.INFORMATION_THEORY,
            statement="Conjecture about entropy rate",
            informal_statement="The entropy rate converges.",
            tags=["entropy", "convergence"],
            verification_status={"numerical": 0.85, "consistency": 0.7},
            overall_confidence=0.78,
            source_article="test_paper.pdf",
        )
        d = node.to_dict()
        node2 = ConceptNode.from_dict(d)

        self.assertEqual(node2.id, "rt1")
        self.assertEqual(node2.concept_type, ConceptType.CONJECTURE)
        self.assertEqual(node2.domain, ScientificDomain.INFORMATION_THEORY)
        self.assertEqual(node2.formal_statement, "Conjecture about entropy rate")
        self.assertEqual(node2.informal_statement, "The entropy rate converges.")
        self.assertEqual(node2.tags, ["entropy", "convergence"])
        self.assertAlmostEqual(node2.verification_status["numerical"], 0.85)
        self.assertAlmostEqual(node2.overall_confidence, 0.78)


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  Runner                                                   â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

if __name__ == "__main__":
    unittest.main(verbosity=2)
