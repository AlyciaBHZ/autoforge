"""Backward compatibility — multi_prover is now in autoforge.engine.provers.multi_prover.

This file re-exports everything so existing imports continue to work:
  from autoforge.engine.multi_prover import MultiProverEngine  # still works
"""
from autoforge.engine.provers.multi_prover import *  # noqa: F401,F403
