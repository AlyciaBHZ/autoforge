"""AutoForge Agent implementations."""

from engine.agents.architect import ArchitectAgent
from engine.agents.builder import BuilderAgent
from engine.agents.director import DirectorAgent, DirectorFixAgent
from engine.agents.gardener import GardenerAgent
from engine.agents.reviewer import ReviewerAgent
from engine.agents.scanner import ScannerAgent
from engine.agents.tester import TesterAgent

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
