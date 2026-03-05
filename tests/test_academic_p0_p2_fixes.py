from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoforge.engine.curriculum_learning import CurriculumScheduler
from autoforge.engine.literature_search import (
    LiteratureSearchConfig,
    LiteratureSearchEngine,
)
from autoforge.engine.paper_formalizer import FormalizationUnit, PaperFormalizer
from autoforge.engine.recursive_decomp_prover import RecursiveDecompProver
from autoforge.engine.self_play_conjecture import SelfPlayEngine
from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptRelation,
    ConceptType,
    RelationType,
    ScientificDomain,
    TheoryGraph,
)
from autoforge.engine.world_model import QueryFilter, WorldModel


def _make_node(
    node_id: str,
    *,
    ctype: ConceptType = ConceptType.THEOREM,
    domain: ScientificDomain = ScientificDomain.PURE_MATHEMATICS,
    statement: str = "placeholder statement",
    confidence: float = 0.8,
) -> ConceptNode:
    return ConceptNode(
        id=node_id,
        concept_type=ctype,
        domain=domain,
        formal_statement=statement,
        overall_confidence=confidence,
    )


class _StubLLM:
    async def generate(self, prompt: str, **_: object) -> str:
        if "Generate the conjectures now" in prompt:
            return (
                '[{"title":"Identity Conjecture",'
                '"statement":"forall n, n = n",'
                '"reason":"Reflexivity basis",'
                '"hint":"Use reflexivity"}]'
            )
        return "PROVED. therefore q.e.d."


class _FakeCloudJob:
    def __init__(self, compiled_ok: bool = True) -> None:
        self.compiled_ok = compiled_ok
        self.errors: list[str] = []
        self.result = "ok"
        self.status = "completed"


class _FakeCloudProver:
    def __init__(self) -> None:
        self.called = False
        self.last_code = ""
        self.last_label = ""

    async def verify_lean(self, lean_code: str, label: str = "") -> _FakeCloudJob:
        self.called = True
        self.last_code = lean_code
        self.last_label = label
        return _FakeCloudJob(compiled_ok=True)


class _ProbeLiteratureSearch(LiteratureSearchEngine):
    def __init__(self) -> None:
        super().__init__(
            LiteratureSearchConfig(
                semantic_scholar_enabled=True,
                arxiv_enabled=True,
                max_results_per_query=2,
                min_relevance_threshold=0.0,
            )
        )
        self.urls: list[str] = []

    async def _http_get(self, url: str) -> str:
        self.urls.append(url)
        if "semanticscholar" in url:
            return '{"data":[]}'
        return '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class TestAcademicP0P2Fixes(unittest.IsolatedAsyncioTestCase):
    async def test_world_model_add_discovery_query_and_reload(self) -> None:
        graph = TheoryGraph(title="wm")
        wm = WorldModel(graph)
        node = _make_node("c1", statement="golden ratio relation", confidence=0.81)
        node.tags = ["golden", "ratio"]

        await wm.add_discovery(node, round_number=1, strategy="unit_test")

        hits = wm.query(QueryFilter(min_confidence=0.5, tags=["golden"]))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].id, "c1")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "world_model.json"
            await wm.save(save_path)
            loaded = WorldModel()
            await loaded.load(save_path)
            self.assertEqual(len(loaded.graph.nodes), 1)
            self.assertEqual(loaded.get_session_stats()["discovery_count"], 1)

    async def test_curriculum_scheduler_batch_not_empty_on_modern_graph(self) -> None:
        graph = TheoryGraph(title="curriculum")
        graph.add_concept(_make_node("def1", ctype=ConceptType.DEFINITION, statement="definition"))
        graph.add_concept(_make_node("thm1", ctype=ConceptType.THEOREM, statement="main theorem"))
        graph.add_relation(ConceptRelation("def1", "thm1", RelationType.DEPENDS_ON))

        scheduler = CurriculumScheduler(graph)
        batch = scheduler.get_next_batch(batch_size=5)
        self.assertGreaterEqual(len(batch), 1)
        self.assertIn("def1", {node.id for node in batch})

    async def test_self_play_engine_produces_and_promotes_conjecture(self) -> None:
        graph = TheoryGraph(title="self_play")
        graph.add_concept(
            _make_node("seed_thm", ctype=ConceptType.THEOREM, statement="forall n, n = n")
        )

        engine = SelfPlayEngine(max_rounds=1, conjectures_per_round=1)
        proved = await engine.run(graph, _StubLLM())

        self.assertEqual(len(proved), 1)
        theorem_ids = {node.id for node in graph.get_concepts_by_type(ConceptType.THEOREM)}
        self.assertTrue(any(node_id.startswith("thm_conj_") for node_id in theorem_ids))

    async def test_recursive_decomp_prover_handles_modern_concept_schema(self) -> None:
        prover = RecursiveDecompProver(max_depth=1, max_attempts_per_goal=1)
        concept = _make_node("goal", ctype=ConceptType.THEOREM, statement="1 = 1")
        result = await prover.prove(concept, _StubLLM())
        self.assertEqual(result.status, "proved")
        self.assertTrue(result.proof)

    async def test_literature_queries_are_encoded_and_https(self) -> None:
        engine = _ProbeLiteratureSearch()
        await engine.search("graph neural networks")

        s2_urls = [u for u in engine.urls if "semanticscholar" in u]
        self.assertTrue(s2_urls)
        self.assertIn("query=graph+neural+networks", s2_urls[0])

        arxiv_urls = [u for u in engine.urls if "arxiv" in u]
        self.assertTrue(arxiv_urls)
        self.assertTrue(arxiv_urls[0].startswith("https://"))
        self.assertIn("search_query=all%3Agraph+neural+networks", arxiv_urls[0])

    async def test_formalizer_uses_cloud_prover_when_local_lean_missing(self) -> None:
        unit = FormalizationUnit(
            concept_id="thm1",
            concept_type="theorem",
            section="1",
            label="Theorem (thm1)",
            natural_language="Trivial theorem",
            formal_statement_latex="True",
            lean_code="theorem t : True := by trivial",
        )
        cloud = _FakeCloudProver()
        formalizer = PaperFormalizer()

        with patch("shutil.which", return_value=None):
            ok = await formalizer._try_lean_compile(unit, cloud_prover=cloud)

        self.assertTrue(ok)
        self.assertTrue(cloud.called)
        self.assertIn("import Mathlib", cloud.last_code)
        self.assertEqual(cloud.last_label, unit.label)


if __name__ == "__main__":
    unittest.main()

