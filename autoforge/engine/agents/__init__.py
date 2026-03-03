"""AutoForge Agent implementations."""

from autoforge.engine.agents.architect import ArchitectAgent
from autoforge.engine.agents.builder import BuilderAgent
from autoforge.engine.agents.director import DirectorAgent, DirectorFixAgent
from autoforge.engine.agents.gardener import GardenerAgent
from autoforge.engine.agents.reviewer import ReviewerAgent
from autoforge.engine.agents.scanner import ScannerAgent
from autoforge.engine.agents.tester import TesterAgent

AGENT_REGISTRY = {
    "director": DirectorAgent,
    "director_fix": DirectorFixAgent,
    "architect": ArchitectAgent,
    "builder": BuilderAgent,
    "reviewer": ReviewerAgent,
    "tester": TesterAgent,
    "gardener": GardenerAgent,
    "scanner": ScannerAgent,
}

__all__ = [
    "DirectorAgent",
    "DirectorFixAgent",
    "ArchitectAgent",
    "BuilderAgent",
    "ReviewerAgent",
    "TesterAgent",
    "GardenerAgent",
    "ScannerAgent",
    "AGENT_REGISTRY",
]
