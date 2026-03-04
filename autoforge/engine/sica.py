"""SICA — Self-Improving Coding Agent.

Inspired by ICLR 2025 Workshop paper: agents that can edit their own
code/scripts to improve their capabilities over time.

Combined with STO (Self-Taught Optimizer, NeurIPS 2025) recursive
self-optimization pattern: the Gardener not only optimizes generated
project code, but can also propose improvements to AutoForge's own
agent prompts, tool definitions, and workflow scripts.

Key mechanisms:
  1. **Constitution Self-Edit**: Agents propose edits to their own
     constitution .md files based on performance data
  2. **Tool Script Generation**: Agents can write new tool wrapper
     scripts that extend their capabilities
  3. **Workflow Mutation**: The system can propose changes to the
     phase pipeline itself (e.g., adding a pre-verify lint step)
  4. **Safety Guardrails**: All self-modifications go through
     validation before being applied

SWE-Bench Verified improvement: 17% → 53% with self-improvement.

References:
  - SICA: Self-Improving Coding Agent (ICLR 2025 Workshop)
  - STO: Self-Taught Optimizer (NeurIPS 2025)
"""

from __future__ import annotations

import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class SelfEditProposal:
    """A proposed self-modification to the system."""
    id: str
    edit_type: str          # "constitution", "tool_script", "workflow", "config"
    target_file: str        # Relative path within constitution/ or engine/
    description: str        # What the edit does
    original_content: str   # Content before edit (for rollback)
    proposed_content: str   # Proposed new content
    rationale: str          # Why this edit should help
    expected_impact: str    # Expected improvement
    confidence: float = 0.5 # How confident the system is this will help
    status: str = "proposed"  # proposed, validated, applied, rolled_back, rejected
    applied_at: float = 0.0
    performance_before: float = 0.0
    performance_after: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "edit_type": self.edit_type,
            "target_file": self.target_file,
            "description": self.description,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "status": self.status,
            "performance_before": self.performance_before,
            "performance_after": self.performance_after,
        }


@dataclass
class ImprovementRecord:
    """Record of a self-improvement attempt and its outcome."""
    proposal_id: str
    applied: bool
    fitness_before: float
    fitness_after: float
    kept: bool = False  # Was the improvement retained?
    timestamp: float = field(default_factory=time.time)

    @property
    def improvement(self) -> float:
        return self.fitness_after - self.fitness_before


# ──────────────────────────────────────────────
# SICA Engine
# ──────────────────────────────────────────────


class SICAEngine:
    """Self-Improving Coding Agent engine.

    Manages the recursive self-optimization loop:

    1. **Observe**: Collect performance data from recent runs
    2. **Propose**: Generate self-edit proposals using LLM
    3. **Validate**: Safety-check proposals before applying
    4. **Apply**: Make the modification
    5. **Evaluate**: Measure if the modification helped
    6. **Decide**: Keep or roll back the change

    Safety guardrails:
    - No edits to core engine files (orchestrator, config, etc.)
    - All edits are reversible (original content preserved)
    - Changes must pass validation before application
    - Automatic rollback if performance degrades by >10%
    """

    # Files that CANNOT be self-modified (safety boundary)
    PROTECTED_FILES = {
        "orchestrator.py", "config.py", "llm_router.py",
        "sandbox.py", "lock_manager.py", "sica.py",
    }

    # Only these edit types are allowed
    ALLOWED_EDIT_TYPES = {"constitution", "tool_script", "workflow"}

    # Maximum proposals per run
    MAX_PROPOSALS_PER_RUN = 3

    # Performance degradation threshold for automatic rollback
    ROLLBACK_THRESHOLD = -0.10  # 10% degradation

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".autoforge"
        self.base_dir = base_dir
        self._proposals: list[SelfEditProposal] = []
        self._history: list[ImprovementRecord] = []
        self._load_history()

    # ──────── Propose Self-Edits ────────

    async def propose_improvements(
        self,
        performance_data: dict[str, Any],
        constitution_dir: Path,
        llm: Any,
    ) -> list[SelfEditProposal]:
        """Analyse recent performance and propose self-improvements.

        Args:
            performance_data: {
                "recent_runs": [...],
                "common_failures": [...],
                "avg_quality": float,
                "avg_test_pass_rate": float,
                "bottleneck_agents": [...],
            }
            constitution_dir: Path to constitution/ directory
            llm: LLM router instance
        """
        from autoforge.engine.llm_router import TaskComplexity

        # Gather current constitution files
        agent_prompts = {}
        agents_dir = constitution_dir / "agents"
        if agents_dir.exists():
            for md_file in agents_dir.glob("*.md"):
                agent_prompts[md_file.stem] = md_file.read_text(encoding="utf-8")[:2000]

        prompt = (
            "You are a meta-optimizer analysing a multi-agent coding system. "
            "Based on recent performance data, propose specific improvements "
            "to the agent instruction files (constitutions).\n\n"
            f"## Performance Data\n"
            f"Average quality score: {performance_data.get('avg_quality', 'N/A')}/10\n"
            f"Average test pass rate: {performance_data.get('avg_test_pass_rate', 'N/A')}\n"
            f"Bottleneck agents: {performance_data.get('bottleneck_agents', [])}\n\n"
        )

        common_failures = performance_data.get("common_failures", [])
        if common_failures:
            prompt += "## Common Failures\n"
            for f in common_failures[:5]:
                prompt += f"- {f}\n"
            prompt += "\n"

        prompt += "## Current Agent Constitutions (summaries)\n"
        for role, content in agent_prompts.items():
            prompt += f"### {role}\n{content[:500]}...\n\n"

        prompt += (
            "## Task\n"
            "Propose 1-3 targeted edits to agent constitutions. "
            "Each proposal should:\n"
            "1. Target a specific agent and section\n"
            "2. Explain what to change and why\n"
            "3. Provide the exact new instruction text\n\n"
            "Output JSON array:\n"
            "```json\n"
            '[\n'
            '  {\n'
            '    "target_agent": "builder",\n'
            '    "description": "Add error recovery instructions",\n'
            '    "new_instruction": "When encountering import errors...",\n'
            '    "rationale": "Builder frequently fails on missing imports",\n'
            '    "confidence": 0.7\n'
            '  }\n'
            ']\n'
            "```"
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.HIGH,
                system="You are a meta-learning system that improves AI agent "
                       "prompts based on empirical performance data.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            proposals = self._parse_proposals(text, constitution_dir)
            self._proposals.extend(proposals)
            logger.info(f"[SICA] Generated {len(proposals)} improvement proposals")
            return proposals

        except Exception as e:
            logger.warning(f"[SICA] Proposal generation failed: {e}")
            return []

    def _parse_proposals(
        self, text: str, constitution_dir: Path,
    ) -> list[SelfEditProposal]:
        """Parse LLM output into validated proposals."""
        import re
        proposals = []

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            raw_text = match.group(1).strip()
        else:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                raw_text = text[start:end + 1]
            else:
                return []

        try:
            items = json.loads(raw_text)
        except json.JSONDecodeError:
            return []

        for item in items[:self.MAX_PROPOSALS_PER_RUN]:
            if not isinstance(item, dict):
                continue

            target_agent = item.get("target_agent", "")
            if not target_agent:
                continue

            target_file = f"agents/{target_agent}.md"
            full_path = constitution_dir / target_file

            # Read original content
            original = ""
            if full_path.exists():
                original = full_path.read_text(encoding="utf-8")

            proposal_id = hashlib.sha256(
                f"{target_file}-{time.time()}".encode()
            ).hexdigest()[:12]

            proposal = SelfEditProposal(
                id=proposal_id,
                edit_type="constitution",
                target_file=target_file,
                description=item.get("description", ""),
                original_content=original,
                proposed_content=item.get("new_instruction", ""),
                rationale=item.get("rationale", ""),
                expected_impact=item.get("expected_impact", "quality improvement"),
                confidence=float(item.get("confidence", 0.5)),
            )
            proposals.append(proposal)

        return proposals

    # ──────── Validate Proposals ────────

    def validate_proposal(self, proposal: SelfEditProposal) -> tuple[bool, str]:
        """Validate a self-edit proposal against safety guardrails.

        Returns (is_valid, reason).
        """
        # Check edit type
        if proposal.edit_type not in self.ALLOWED_EDIT_TYPES:
            return False, f"Edit type '{proposal.edit_type}' not allowed"

        # Check protected files
        filename = Path(proposal.target_file).name
        if filename in self.PROTECTED_FILES:
            return False, f"File '{filename}' is protected and cannot be self-modified"

        # Check proposed content is not empty
        if not proposal.proposed_content.strip():
            return False, "Proposed content is empty"

        # Check proposed content is not suspiciously short
        if len(proposal.proposed_content.strip()) < 20:
            return False, "Proposed content too short (< 20 chars)"

        # Check confidence threshold
        if proposal.confidence < 0.3:
            return False, f"Confidence too low ({proposal.confidence:.1f} < 0.3)"

        # Check we haven't applied too many proposals recently
        recent_applied = [
            r for r in self._history
            if r.applied and (time.time() - r.timestamp) < 3600  # last hour
        ]
        if len(recent_applied) >= self.MAX_PROPOSALS_PER_RUN * 2:
            return False, "Too many recent modifications — cooling down"

        proposal.status = "validated"
        return True, "OK"

    # ──────── Apply Proposals ────────

    def apply_proposal(
        self,
        proposal: SelfEditProposal,
        constitution_dir: Path,
        current_fitness: float = 0.0,
    ) -> bool:
        """Apply a validated proposal by modifying the target file.

        The original content is preserved for potential rollback.
        """
        if proposal.status != "validated":
            logger.warning(f"[SICA] Cannot apply unvalidated proposal {proposal.id}")
            return False

        target_path = constitution_dir / proposal.target_file
        try:
            if target_path.exists():
                # Append new instruction to existing constitution
                current = target_path.read_text(encoding="utf-8")
                updated = current + (
                    f"\n\n## Self-Improvement: {proposal.description}\n"
                    f"<!-- SICA proposal {proposal.id} -->\n"
                    f"{proposal.proposed_content}\n"
                )
                target_path.write_text(updated, encoding="utf-8")
            else:
                logger.warning(f"[SICA] Target file not found: {target_path}")
                return False

            proposal.status = "applied"
            proposal.applied_at = time.time()
            proposal.performance_before = current_fitness

            self._history.append(ImprovementRecord(
                proposal_id=proposal.id,
                applied=True,
                fitness_before=current_fitness,
                fitness_after=0.0,  # Will be updated after evaluation
            ))

            logger.info(f"[SICA] Applied proposal {proposal.id}: {proposal.description}")
            self._save_history()
            return True

        except Exception as e:
            logger.error(f"[SICA] Failed to apply proposal {proposal.id}: {e}")
            return False

    # ──────── Evaluate & Rollback ────────

    def evaluate_and_decide(
        self,
        proposal: SelfEditProposal,
        new_fitness: float,
        constitution_dir: Path,
    ) -> bool:
        """Evaluate if an applied proposal improved performance.

        If performance degraded beyond threshold, automatically rolls back.

        Returns True if the change was kept, False if rolled back.
        """
        proposal.performance_after = new_fitness
        delta = new_fitness - proposal.performance_before

        # Update history record
        for record in reversed(self._history):
            if record.proposal_id == proposal.id:
                record.fitness_after = new_fitness
                break

        if delta < self.ROLLBACK_THRESHOLD:
            # Performance degraded — roll back
            logger.info(
                f"[SICA] Rolling back proposal {proposal.id}: "
                f"fitness {proposal.performance_before:.2f} → {new_fitness:.2f} "
                f"(delta={delta:+.2f})"
            )
            self._rollback(proposal, constitution_dir)
            return False

        # Change is acceptable — keep it
        logger.info(
            f"[SICA] Keeping proposal {proposal.id}: "
            f"fitness {proposal.performance_before:.2f} → {new_fitness:.2f} "
            f"(delta={delta:+.2f})"
        )

        for record in reversed(self._history):
            if record.proposal_id == proposal.id:
                record.kept = True
                break

        self._save_history()
        return True

    def _rollback(
        self,
        proposal: SelfEditProposal,
        constitution_dir: Path,
    ) -> None:
        """Roll back a proposal by restoring original content."""
        target_path = constitution_dir / proposal.target_file
        try:
            if proposal.original_content:
                target_path.write_text(proposal.original_content, encoding="utf-8")
                proposal.status = "rolled_back"
                logger.info(f"[SICA] Rolled back {proposal.id}")
            else:
                logger.warning(f"[SICA] No original content to restore for {proposal.id}")
        except Exception as e:
            logger.error(f"[SICA] Rollback failed for {proposal.id}: {e}")

        self._save_history()

    # ──────── Tool Script Generation ────────

    async def generate_tool_script(
        self,
        need_description: str,
        llm: Any,
    ) -> SelfEditProposal | None:
        """Generate a new tool wrapper script to extend agent capabilities.

        This is the 'self-skill generation' aspect: when agents encounter
        a repeated need (e.g. 'parse YAML configs'), the system generates
        a reusable tool script.
        """
        from autoforge.engine.llm_router import TaskComplexity

        prompt = (
            f"Generate a Python tool script for the following capability:\n\n"
            f"Need: {need_description}\n\n"
            f"The script should:\n"
            f"1. Be a standalone async function\n"
            f"2. Accept input as a dict and return a string result\n"
            f"3. Have proper error handling\n"
            f"4. Include a docstring\n\n"
            f"Output ONLY the Python code, wrapped in ```python``` blocks."
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You write clean, focused Python tool scripts.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            import re
            match = re.search(r"```python\s*\n?(.*?)\n?```", text, re.DOTALL)
            if not match:
                return None

            code = match.group(1).strip()

            # Basic validation: must define at least one async function
            if "async def" not in code:
                return None

            script_id = hashlib.sha256(
                f"tool-{need_description}-{time.time()}".encode()
            ).hexdigest()[:12]

            proposal = SelfEditProposal(
                id=script_id,
                edit_type="tool_script",
                target_file=f"tools/auto_{script_id}.py",
                description=f"Auto-generated tool: {need_description}",
                original_content="",
                proposed_content=code,
                rationale=f"Needed capability: {need_description}",
                expected_impact="Extended agent toolset",
                confidence=0.6,
            )
            return proposal

        except Exception as e:
            logger.warning(f"[SICA] Tool script generation failed: {e}")
            return None

    # ──────── Statistics ────────

    def get_improvement_stats(self) -> dict[str, Any]:
        """Get statistics about self-improvement history."""
        if not self._history:
            return {"total_proposals": 0, "message": "No self-improvement history yet"}

        applied = [r for r in self._history if r.applied]
        kept = [r for r in applied if r.kept]
        improvements = [r.improvement for r in applied if r.fitness_after > 0]

        return {
            "total_proposals": len(self._proposals),
            "total_applied": len(applied),
            "total_kept": len(kept),
            "total_rolled_back": len(applied) - len(kept),
            "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "best_improvement": max(improvements) if improvements else 0,
            "keep_rate": len(kept) / len(applied) if applied else 0,
        }

    # ──────── Persistence ────────

    def _save_history(self) -> None:
        """Save improvement history to disk."""
        path = self.base_dir / "sica_history.json"
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "history": [
                    {
                        "proposal_id": r.proposal_id,
                        "applied": r.applied,
                        "fitness_before": r.fitness_before,
                        "fitness_after": r.fitness_after,
                        "kept": r.kept,
                        "timestamp": r.timestamp,
                    }
                    for r in self._history[-100:]  # Keep last 100
                ],
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[SICA] Could not save history: {e}")

    def _load_history(self) -> None:
        """Load improvement history from disk."""
        path = self.base_dir / "sica_history.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for r in data.get("history", []):
                self._history.append(ImprovementRecord(
                    proposal_id=r["proposal_id"],
                    applied=r["applied"],
                    fitness_before=r["fitness_before"],
                    fitness_after=r.get("fitness_after", 0),
                    kept=r.get("kept", False),
                    timestamp=r.get("timestamp", 0),
                ))
            logger.info(f"[SICA] Loaded {len(self._history)} improvement records")
        except Exception as e:
            logger.debug(f"[SICA] Could not load history: {e}")
