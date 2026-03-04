"""Backward compatibility — lean_prover is now in autoforge.engine.provers.

All classes and functions have been moved to:
  - autoforge.engine.provers.lean_core: Core data structures, LeanEnvironment, LeanProver
  - autoforge.engine.provers.proof_search: TacticGenerator, MCTSProofSearch, RecursiveProofDecomposer, ConjectureEngine
  - autoforge.engine.provers.proof_library: FoundationBuilder, ArticleFormalizer, MathlibPremiseSelector, ProofRepairEngine, PaperReviewPipeline

This file re-exports everything so existing imports continue to work:
  from autoforge.engine.lean_prover import LeanProver  # still works
"""
from autoforge.engine.provers.lean_core import *  # noqa: F401,F403
from autoforge.engine.provers.proof_search import *  # noqa: F401,F403
from autoforge.engine.provers.proof_library import *  # noqa: F401,F403
