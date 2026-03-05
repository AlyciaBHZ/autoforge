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

import base64
import hashlib
import json
import logging
import random
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# A/B Testing for Constitution Edits
# ──────────────────────────────────────────────


class ConstitutionABTester:
    """A/B test constitution changes using Welch's t-test.

    Tracks quality scores for control (original) vs treatment (edited)
    constitutions and determines if changes are statistically significant.
    """

    def __init__(self) -> None:
        self._results_a: list[float] = []  # Control (original)
        self._results_b: list[float] = []  # Treatment (edited)

    def record_result(self, variant: str, quality_score: float) -> None:
        """Record a quality score for variant A or B."""
        if variant.lower() == "a":
            self._results_a.append(quality_score)
        else:
            self._results_b.append(quality_score)

    def is_improvement(self, confidence: float = 0.95) -> tuple[bool, dict]:
        """Use Welch's t-test to determine if B > A.

        Returns (is_improved, details_dict).
        confidence: 0.95 for 95% confidence level (two-tailed).
        """
        if len(self._results_a) < 3 or len(self._results_b) < 3:
            return False, {"reason": "insufficient_data", "n_a": len(self._results_a), "n_b": len(self._results_b)}

        import statistics
        import math

        mean_a = statistics.mean(self._results_a)
        mean_b = statistics.mean(self._results_b)
        var_a = statistics.variance(self._results_a)
        var_b = statistics.variance(self._results_b)
        n_a, n_b = len(self._results_a), len(self._results_b)

        # Welch's t-statistic (for unequal variances)
        se = (var_a / n_a + var_b / n_b) ** 0.5
        if se == 0:
            return mean_b > mean_a, {"t": float('inf'), "se": 0, "mean_a": mean_a, "mean_b": mean_b}

        t_stat = (mean_b - mean_a) / se

        # Approximate p-value using normal distribution for large samples
        p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))

        # One-tailed test: is B > A?
        is_improved = p_value < (1 - confidence) and mean_b > mean_a

        return is_improved, {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "t_stat": t_stat,
            "p_value": p_value,
            "effect_size": (mean_b - mean_a) / max(se, 0.001),
            "n_a": n_a,
            "n_b": n_b,
        }


class ConstitutionAnalyzer:
    """Analyze constitution rules for conflicts and redundancy.

    Uses Jaccard similarity and logical pattern matching to detect
    contradictions and duplications in a set of rules.
    """

    def analyze(self, rules: list[str]) -> dict[str, Any]:
        """Check for conflicts and redundancy in a set of rules.

        Returns dict with:
        - conflicts: list of (rule_i_idx, rule_j_idx, reason)
        - redundant: list of (rule_i_idx, rule_j_idx, similarity_score)
        """
        conflicts = []
        redundant = []

        for i, r1 in enumerate(rules):
            for j, r2 in enumerate(rules[i+1:], i+1):
                r1_lower = r1.lower()
                r2_lower = r2.lower()

                # Check for potential contradiction
                if self._may_conflict(r1_lower, r2_lower):
                    conflicts.append((i, j, "Potential contradiction"))

                # Check for high similarity (redundancy)
                sim = self._jaccard_similarity(r1_lower, r2_lower)
                if sim > 0.7:
                    redundant.append((i, j, f"Similarity={sim:.2f}"))

        return {"conflicts": conflicts, "redundant": redundant}

    def _may_conflict(self, a: str, b: str) -> bool:
        """Detect if two rules may contradict."""
        negators = ['never', 'always', 'must not', 'should not', 'do not', 'cannot']

        a_has_negation = any(neg in a for neg in negators)
        b_has_negation = any(neg in b for neg in negators)

        # Only flag if one has negation but the other doesn't, and they share key terms
        if a_has_negation != b_has_negation:
            shared = set(a.split()) & set(b.split()) - {'the', 'a', 'an', 'is', 'to'}
            if len(shared) > 3:
                return True

        return False

    def _jaccard_similarity(self, a: str, b: str) -> float:
        """Calculate Jaccard similarity between two rule texts."""
        ta = set(a.split())
        tb = set(b.split())
        inter = ta & tb
        union = ta | tb
        return len(inter) / len(union) if union else 0.0


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
        self._file_edit_stack: dict[str, list[str]] = {}  # target_file → list of proposal_ids in apply order
        self._file_base_content: dict[str, str] = {}  # target_file → original content before first SICA edit
        self._ab_testers: dict[str, ConstitutionABTester] = {}  # proposal_id → A/B tester
        self._constitution_analyzer = ConstitutionAnalyzer()
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

        Performs:
        1. Type and file checks
        2. Content length/validity checks
        3. Confidence threshold checks
        4. Rate limiting (cooldown after many edits)
        5. Constitution conflict analysis (for constitution edits)
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

        # For constitution edits, analyze for conflicts with existing rules
        if proposal.edit_type == "constitution":
            # Parse existing rules from original content
            existing_rules = self._extract_rules_from_markdown(proposal.original_content)
            proposed_rules = self._extract_rules_from_markdown(proposal.proposed_content)

            if existing_rules and proposed_rules:
                combined_rules = existing_rules + proposed_rules
                analysis = self._constitution_analyzer.analyze(combined_rules)
                if analysis["conflicts"]:
                    logger.warning(
                        f"[SICA] Proposal {proposal.id} creates conflicts: {analysis['conflicts'][:2]}"
                    )
                    # Don't reject, just warn and lower confidence slightly
                    proposal.confidence *= 0.8

        proposal.status = "validated"
        return True, "OK"

    def _extract_rules_from_markdown(self, markdown_text: str) -> list[str]:
        """Extract rule text from markdown (lines starting with -, *, or -)."""
        rules = []
        for line in markdown_text.split('\n'):
            if line.strip().startswith(('-', '*', '+')):
                rules.append(line.strip()[1:].strip())
        return rules

    # ──────── Apply Proposals ────────

    def apply_proposal(
        self,
        proposal: SelfEditProposal,
        constitution_dir: Path,
        current_fitness: float = 0.0,
    ) -> bool:
        """Apply a validated proposal by modifying the target file.

        The original content is preserved for potential rollback. Proposals are
        tracked in a per-file stack to support multi-proposal rollback awareness.
        """
        if proposal.status != "validated":
            logger.warning(f"[SICA] Cannot apply unvalidated proposal {proposal.id}")
            return False

        target_path = constitution_dir / proposal.target_file
        try:
            # Read current file content
            if target_path.exists():
                proposal.original_content = target_path.read_text(encoding="utf-8")
            else:
                proposal.original_content = ""

            # On first edit to this file, store the pristine base content
            # so _replay_proposals_without has a reliable starting point
            if proposal.target_file not in self._file_base_content:
                self._file_base_content[proposal.target_file] = proposal.original_content

            if proposal.edit_type == "constitution":
                # Constitution edits: append to existing file, cleanup old SICA sections
                if target_path.exists():
                    current = target_path.read_text(encoding="utf-8")

                    # Count existing SICA sections marked with "## Self-Improvement:"
                    sica_sections = re.findall(r'## Self-Improvement:', current)

                    # If there are already more than 3 SICA sections, remove the oldest one
                    if len(sica_sections) > 3:
                        # Find and remove the first SICA section (oldest)
                        pattern = r'\n\n## Self-Improvement:.*?(?=\n\n## Self-Improvement:|$)'
                        current = re.sub(pattern, '', current, count=1, flags=re.DOTALL)

                    updated = current + (
                        f"\n\n## Self-Improvement: {proposal.description}\n"
                        f"<!-- SICA proposal {proposal.id} -->\n"
                        f"{proposal.proposed_content}\n"
                    )
                    target_path.write_text(updated, encoding="utf-8")
                else:
                    logger.warning(f"[SICA] Constitution file not found: {target_path}")
                    proposal.status = "failed"
                    return False
            elif proposal.edit_type == "tool_script":
                # Tool scripts are new files — create parent dirs and write
                target_path.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    f"# Auto-generated tool script — SICA proposal {proposal.id}\n"
                    f"# {proposal.description}\n\n"
                )
                target_path.write_text(header + proposal.proposed_content, encoding="utf-8")
            elif proposal.edit_type == "workflow":
                # Workflow definitions are new files — create parent dirs and write
                target_path.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    f"<!-- Auto-generated workflow — SICA proposal {proposal.id} -->\n"
                    f"<!-- {proposal.description} -->\n\n"
                )
                target_path.write_text(header + proposal.proposed_content, encoding="utf-8")
            else:
                logger.warning(f"[SICA] Unknown edit type '{proposal.edit_type}' for proposal {proposal.id}")
                proposal.status = "failed"
                return False

            # Track this proposal in the edit stack for its target file
            if proposal.target_file not in self._file_edit_stack:
                self._file_edit_stack[proposal.target_file] = []
            self._file_edit_stack[proposal.target_file].append(proposal.id)

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
        control_fitness: float | None = None,
    ) -> bool:
        """Evaluate if an applied proposal improved performance.

        If performance degraded beyond threshold, automatically rolls back.

        Optionally uses A/B testing (Welch's t-test) if control_fitness is provided.

        Args:
            proposal: The proposal to evaluate
            new_fitness: Fitness after applying proposal
            constitution_dir: Path to constitution directory
            control_fitness: Optional baseline fitness for A/B testing

        Returns True if the change was kept, False if rolled back.
        """
        proposal.performance_after = new_fitness
        delta = new_fitness - proposal.performance_before

        # Initialize A/B tester for this proposal if needed
        if proposal.id not in self._ab_testers:
            self._ab_testers[proposal.id] = ConstitutionABTester()

        ab_tester = self._ab_testers[proposal.id]

        # Record baseline (control) if provided
        if control_fitness is not None:
            ab_tester.record_result("a", control_fitness)

        # Record treatment result
        ab_tester.record_result("b", new_fitness)

        # Use A/B test if we have enough samples
        ab_improved, ab_details = ab_tester.is_improvement(confidence=0.95)

        # Safety check: warn if multiple proposals target the same file
        if proposal.target_file in self._file_edit_stack:
            stack = self._file_edit_stack[proposal.target_file]
            if len(stack) > 1:
                logger.warning(
                    f"[SICA] Multiple proposals ({len(stack)}) target file "
                    f"{proposal.target_file}. Rolling back {proposal.id} may "
                    f"affect other proposals: {stack}"
                )

        # Update history record
        for record in reversed(self._history):
            if record.proposal_id == proposal.id:
                record.fitness_after = new_fitness
                break

        # Decision logic: use A/B test if available, otherwise simple delta threshold
        should_rollback = False
        decision_method = "delta_threshold"

        if ab_improved:
            decision_method = "ab_test_improvement"
            logger.info(f"[SICA] A/B test shows improvement: {ab_details}")
        elif ab_details.get("n_a", 0) >= 3:
            # A/B test completed but no significant improvement
            decision_method = "ab_test_no_improvement"
            should_rollback = True
            logger.info(f"[SICA] A/B test shows no significant improvement: {ab_details}")
        elif delta < self.ROLLBACK_THRESHOLD:
            # Fallback to delta threshold if A/B test insufficient
            should_rollback = True
            logger.info(
                f"[SICA] Rolling back proposal {proposal.id}: "
                f"fitness {proposal.performance_before:.2f} → {new_fitness:.2f} "
                f"(delta={delta:+.2f})"
            )

        if should_rollback:
            self._rollback(proposal, constitution_dir)
            return False

        # Change is acceptable — keep it
        logger.info(
            f"[SICA] Keeping proposal {proposal.id}: "
            f"fitness {proposal.performance_before:.2f} → {new_fitness:.2f} "
            f"(delta={delta:+.2f}, method={decision_method})"
        )

        for record in reversed(self._history):
            if record.proposal_id == proposal.id:
                record.kept = True
                break

        self._save_history()
        return True

    def _replay_proposals_without(
        self,
        target_file: str,
        exclude_id: str,
        constitution_dir: Path,
    ) -> None:
        """Reconstruct a file by replaying all applied proposals except the excluded one.

        This handles intermediate rollbacks where other proposals have been applied
        after the one being rolled back. Instead of simply restoring original_content,
        we replay all other proposals in order to preserve their changes.

        Args:
            target_file: The target file path (relative)
            exclude_id: The proposal ID to skip during replay
            constitution_dir: Path to the constitution directory
        """
        target_path = constitution_dir / target_file
        stack = self._file_edit_stack.get(target_file, [])

        if not stack:
            logger.warning(f"[SICA] No edit stack for {target_file} during replay")
            return

        # Use the stored base content (pristine state before any SICA edits)
        base_content = self._file_base_content.get(target_file, "")

        # Fallback: if base content not stored (e.g. after restart), try the
        # first proposal's original_content, then current disk state
        if not base_content:
            for proposal_id in stack:
                for p in self._proposals:
                    if p.id == proposal_id:
                        base_content = p.original_content
                        break
                if base_content:
                    break
        if not base_content and target_path.exists():
            logger.warning(f"[SICA] Using current disk content as replay base for {target_file}")
            base_content = target_path.read_text(encoding="utf-8")

        # Replay all proposals except the excluded one
        current_content = base_content
        for proposal_id in stack:
            if proposal_id == exclude_id:
                continue

            # Find the proposal and apply its content changes
            proposal = None
            for p in self._proposals:
                if p.id == proposal_id:
                    proposal = p
                    break

            if not proposal:
                logger.warning(f"[SICA] Could not find proposal {proposal_id} for replay")
                continue

            if proposal.edit_type == "constitution":
                # For constitution edits, append the SICA section again
                current_content += (
                    f"\n\n## Self-Improvement: {proposal.description}\n"
                    f"<!-- SICA proposal {proposal.id} -->\n"
                    f"{proposal.proposed_content}\n"
                )

        # Write the reconstructed content
        try:
            target_path.write_text(current_content, encoding="utf-8")
            logger.info(f"[SICA] Replayed proposals for {target_file} (excluded {exclude_id})")
        except Exception as e:
            logger.error(f"[SICA] Failed to replay proposals for {target_file}: {e}")

    def _rollback(
        self,
        proposal: SelfEditProposal,
        constitution_dir: Path,
    ) -> None:
        """Roll back a proposal by restoring original content or replaying other proposals.

        If the proposal being rolled back is the last one in the stack for its target file,
        restore the original_content. Otherwise, replay all proposals except this one to
        preserve changes from other proposals that came after it.
        """
        target_path = constitution_dir / proposal.target_file
        stack = self._file_edit_stack.get(proposal.target_file, [])

        try:
            if proposal.target_file in self._file_edit_stack and proposal.id in stack:
                is_last = stack[-1] == proposal.id

                if is_last:
                    # Simple case: just restore original content
                    if proposal.original_content:
                        target_path.write_text(proposal.original_content, encoding="utf-8")
                        proposal.status = "rolled_back"
                        logger.info(f"[SICA] Rolled back {proposal.id} (was last in stack)")
                    else:
                        logger.warning(f"[SICA] No original content to restore for {proposal.id}")
                else:
                    # Complex case: replay all proposals except this one
                    logger.info(
                        f"[SICA] Rolling back {proposal.id} (intermediate in stack); "
                        f"replaying {len(stack) - 1} other proposals"
                    )
                    self._replay_proposals_without(proposal.target_file, proposal.id, constitution_dir)
                    proposal.status = "rolled_back"

                # Remove from stack
                self._file_edit_stack[proposal.target_file].remove(proposal.id)
            else:
                # Fallback: file not in stack (shouldn't happen, but be safe)
                if proposal.original_content:
                    target_path.write_text(proposal.original_content, encoding="utf-8")
                    proposal.status = "rolled_back"
                    logger.info(f"[SICA] Rolled back {proposal.id} (not in stack)")
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


# ──────────────────────────────────────────────
# Darwin Gödel Machine — Self-Rewriting Agent
# ──────────────────────────────────────────────


@dataclass
class RewriteCandidate:
    """A candidate rewrite of a module or constitution file."""
    candidate_id: str
    target_module: str      # Name of the module/file being rewritten
    original_code: str      # Original source code
    rewritten_code: str     # Proposed rewritten code
    test_results: dict[str, Any] = field(default_factory=dict)  # {test_name: passed, ...}
    fitness_delta: float = 0.0  # Improvement over baseline (delta)
    generation: int = 0     # Which generation this candidate came from
    parent_id: str = ""     # Parent candidate ID (empty for generation 0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize candidate to dict, compressing code with zlib+base64."""
        # Compress code to save space
        original_compressed = base64.b64encode(
            zlib.compress(self.original_code.encode('utf-8'))
        ).decode('utf-8')
        rewritten_compressed = base64.b64encode(
            zlib.compress(self.rewritten_code.encode('utf-8'))
        ).decode('utf-8')

        return {
            "candidate_id": self.candidate_id,
            "target_module": self.target_module,
            "original_code": original_compressed,
            "rewritten_code": rewritten_compressed,
            "test_results": self.test_results,
            "fitness_delta": self.fitness_delta,
            "generation": self.generation,
            "parent_id": self.parent_id,
        }


@dataclass
class RewriteConfig:
    """Configuration for Darwin self-rewriting evolution."""
    max_generations: int = 5
    population_size: int = 4
    mutation_rate: float = 0.3        # Probability of mutation during evolution
    crossover_rate: float = 0.5       # Probability of crossover
    fitness_threshold: float = 0.0    # Minimum fitness delta to keep a candidate
    model: str = "claude-sonnet-4-20250514"  # Model for rewrite generation
    allowed_modules: list[str] = field(default_factory=lambda: [
        "constitution",
        "workflows",
    ])
    protected_modules: list[str] = field(default_factory=lambda: [
        "orchestrator.py", "config.py", "llm_router.py",
        "sandbox.py", "lock_manager.py", "sica.py",
        "agent_base.py", "search_tree.py",
    ])


class DarwinSelfRewriter:
    """Darwin Gödel Machine style self-rewriting agent.

    Uses evolutionary algorithms to automatically improve the system's own
    code by:
    1. Proposing multiple candidate rewrites
    2. Evaluating each candidate through actual task performance
    3. Selecting the best performers
    4. Breeding new candidates from winners
    5. Persisting the best improvements

    Safety: Only constitution and workflow files can be rewritten.
    All other system modules are protected.
    """

    def __init__(
        self,
        config: RewriteConfig | None = None,
        sica_engine: SICAEngine | None = None,
        base_dir: Path | None = None,
    ) -> None:
        """Initialize Darwin self-rewriter.

        Args:
            config: RewriteConfig for evolution parameters
            sica_engine: Parent SICA engine for applying approved changes
            base_dir: Base directory for state persistence
        """
        self.config = config or RewriteConfig()
        self.sica_engine = sica_engine or SICAEngine()
        if base_dir is None:
            base_dir = Path.home() / ".autoforge"
        self.base_dir = base_dir

        self._population: list[RewriteCandidate] = []
        self._generation_history: list[list[RewriteCandidate]] = []
        self._baseline_fitness: dict[str, float] = {}  # Per-module baseline
        self._evolution_state_file = base_dir / "darwin_evolution_state.json"

        logger.info("[Darwin] Self-rewriter initialized")
        self._load_evolution_state()

    # ──────── Candidate Generation ────────

    async def generate_rewrite_candidate(
        self,
        target_module: str,
        performance_context: dict[str, Any],
        llm: Any,
    ) -> RewriteCandidate | None:
        """Generate a candidate rewrite of target_module using LLM.

        Args:
            target_module: Name of module/constitution to rewrite
            performance_context: Dict with performance metrics, errors, suggestions
            llm: LLM client (e.g., Anthropic)

        Returns:
            RewriteCandidate with proposed rewrite, or None if generation failed
        """
        try:
            # Validate module is allowed
            if any(p in target_module for p in self.config.protected_modules):
                logger.warning(f"[Darwin] Cannot rewrite protected module: {target_module}")
                return None

            allowed = any(
                a in target_module for a in self.config.allowed_modules
            )
            if not allowed:
                logger.warning(f"[Darwin] Module not in allowed list: {target_module}")
                return None

            # Load original code
            module_path = self.base_dir.parent / target_module
            if not module_path.exists():
                logger.warning(f"[Darwin] Module not found: {module_path}")
                return None

            original_code = module_path.read_text(encoding="utf-8")

            # Build prompt for LLM
            prompt = self._build_rewrite_prompt(
                target_module, original_code, performance_context
            )

            # Call LLM
            response = await llm.messages.create(
                model=self.config.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract rewritten code from response
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Look for code block
            match = re.search(r"```(?:python|markdown)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if not match:
                logger.warning("[Darwin] No code block found in LLM response")
                return None

            rewritten_code = match.group(1).strip()

            # Validate rewritten code
            if not rewritten_code or len(rewritten_code) < 10:
                logger.warning("[Darwin] Rewritten code too short or empty")
                return None

            # Create candidate
            candidate_id = hashlib.sha256(
                f"darwin-{target_module}-{time.time()}".encode()
            ).hexdigest()[:12]

            candidate = RewriteCandidate(
                candidate_id=candidate_id,
                target_module=target_module,
                original_code=original_code,
                rewritten_code=rewritten_code,
                generation=len(self._generation_history),
                parent_id="",
            )

            logger.info(f"[Darwin] Generated candidate {candidate_id} for {target_module}")
            return candidate

        except Exception as e:
            logger.error(f"[Darwin] Candidate generation failed: {e}")
            return None

    def _build_rewrite_prompt(
        self,
        target_module: str,
        original_code: str,
        performance_context: dict[str, Any],
    ) -> str:
        """Build prompt for LLM to rewrite a module."""
        context_str = json.dumps(performance_context, indent=2)[:2000]

        return f"""You are a self-improving AI system. Your task is to rewrite a module to improve performance.

**Target Module**: {target_module}

**Original Code** (first 2000 chars):
```
{original_code[:2000]}
```

**Performance Context**:
{context_str}

**Task**: Rewrite this module to address the performance issues. Focus on:
1. Fixing identified bugs or inefficiencies
2. Improving error handling
3. Enhancing clarity and maintainability
4. Optimizing critical paths

Return ONLY the complete rewritten code in a markdown code block.
Do NOT include explanations or markdown outside the code block."""

    # ──────── Evaluation ────────

    async def evaluate_candidate(
        self,
        candidate: RewriteCandidate,
        test_runner: Any = None,
    ) -> None:
        """Evaluate a candidate's fitness through testing.

        Runs the candidate against a test suite and records results.
        Fitness is computed as: (tests_passed / total_tests) - baseline_for_module

        Args:
            candidate: Candidate to evaluate
            test_runner: Optional test runner (e.g., pytest harness)
        """
        try:
            # For now, simulate test results based on code quality heuristics
            # In production, this would run actual test suite
            results = self._run_candidate_tests(candidate, test_runner)

            passed = sum(1 for v in results.values() if v)
            total = len(results) if results else 1
            test_pass_rate = passed / total if total > 0 else 0

            # Get baseline
            baseline = self._baseline_fitness.get(candidate.target_module, 0.5)

            # Fitness delta
            candidate.fitness_delta = test_pass_rate - baseline
            candidate.test_results = results

            logger.info(
                f"[Darwin] {candidate.candidate_id} fitness_delta={candidate.fitness_delta:.3f} "
                f"(pass_rate={test_pass_rate:.2%}, baseline={baseline:.2%})"
            )

        except Exception as e:
            logger.error(f"[Darwin] Evaluation failed for {candidate.candidate_id}: {e}")
            candidate.fitness_delta = -0.5  # Penalize failed evaluation

    def _run_candidate_tests(
        self,
        candidate: RewriteCandidate,
        test_runner: Any,
    ) -> dict[str, bool]:
        """Run tests against a candidate (simplified version).

        In production, this would:
        1. Temporarily apply the rewrite
        2. Run test suite
        3. Restore original
        4. Return detailed results

        For now, use heuristics.
        """
        results = {}

        # Heuristic 1: Check syntax validity
        try:
            compile(candidate.rewritten_code, '<rewritten>', 'exec')
            results['syntax_valid'] = True
        except SyntaxError:
            results['syntax_valid'] = False
            return results

        # Heuristic 2: Code quality checks
        has_triple_double = '"""' in candidate.rewritten_code
        has_triple_single = "'''" in candidate.rewritten_code
        results['has_docstrings'] = has_triple_double or has_triple_single
        results['has_error_handling'] = 'except' in candidate.rewritten_code or 'try' in candidate.rewritten_code
        results['has_logging'] = 'logger' in candidate.rewritten_code
        results['maintains_api'] = self._check_api_compatibility(candidate)
        results['improves_clarity'] = len(candidate.rewritten_code) >= len(candidate.original_code) * 0.8

        return results

    def _check_api_compatibility(self, candidate: RewriteCandidate) -> bool:
        """Check if rewritten code maintains API compatibility."""
        # Extract function/class signatures from both versions
        orig_defs = re.findall(r'^(?:async\s+)?def\s+(\w+)', candidate.original_code, re.MULTILINE)
        new_defs = re.findall(r'^(?:async\s+)?def\s+(\w+)', candidate.rewritten_code, re.MULTILINE)

        # All original public defs should be in new version
        public_orig = [d for d in orig_defs if not d.startswith('_')]
        return all(d in new_defs for d in public_orig)

    # ──────── Mutation & Crossover ────────

    async def mutate(
        self,
        candidate: RewriteCandidate,
        llm: Any,
    ) -> RewriteCandidate | None:
        """Apply random mutations to a candidate to explore variation.

        Mutations include:
        - Refactoring variable names
        - Adding/removing error handling
        - Restructuring control flow
        - Optimizing loops

        Args:
            candidate: Candidate to mutate
            llm: LLM client

        Returns:
            New mutated candidate, or None if mutation failed
        """
        try:
            mutation_types = [
                "refactor_names",
                "enhance_error_handling",
                "optimize_loops",
                "simplify_logic",
            ]
            mutation_type = random.choice(mutation_types)

            prompt = f"""You are optimizing code through mutation.

**Mutation Type**: {mutation_type}

**Original Code**:
```python
{candidate.rewritten_code[:2000]}
```

Apply ONE small, focused {mutation_type} mutation. Return only the mutated code block."""

            response = await llm.messages.create(
                model=self.config.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            match = re.search(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if not match:
                return None

            mutated_code = match.group(1).strip()

            # Create mutated candidate
            mutant_id = hashlib.sha256(
                f"{candidate.candidate_id}-mutate-{time.time()}".encode()
            ).hexdigest()[:12]

            mutant = RewriteCandidate(
                candidate_id=mutant_id,
                target_module=candidate.target_module,
                original_code=candidate.original_code,
                rewritten_code=mutated_code,
                generation=candidate.generation + 1,
                parent_id=candidate.candidate_id,
            )

            logger.info(f"[Darwin] Mutated {candidate.candidate_id} -> {mutant_id}")
            return mutant

        except Exception as e:
            logger.warning(f"[Darwin] Mutation failed: {e}")
            return None

    async def crossover(
        self,
        parent1: RewriteCandidate,
        parent2: RewriteCandidate,
        llm: Any,
    ) -> RewriteCandidate | None:
        """Combine best parts of two candidates through crossover.

        Args:
            parent1, parent2: Parent candidates to blend
            llm: LLM client

        Returns:
            New candidate blending both parents, or None if crossover failed
        """
        try:
            prompt = f"""You are combining two code variations to create a hybrid.

**Parent 1** (first half):
```python
{parent1.rewritten_code[:1500]}
```

**Parent 2** (alternative):
```python
{parent2.rewritten_code[:1500]}
```

Create a hybrid that takes the best parts from both while maintaining correctness.
Return ONLY the complete hybrid code in a markdown block."""

            response = await llm.messages.create(
                model=self.config.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            match = re.search(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if not match:
                return None

            hybrid_code = match.group(1).strip()

            # Create offspring
            offspring_id = hashlib.sha256(
                f"{parent1.candidate_id}-x-{parent2.candidate_id}-{time.time()}".encode()
            ).hexdigest()[:12]

            offspring = RewriteCandidate(
                candidate_id=offspring_id,
                target_module=parent1.target_module,
                original_code=parent1.original_code,
                rewritten_code=hybrid_code,
                generation=max(parent1.generation, parent2.generation) + 1,
                parent_id=f"{parent1.candidate_id}+{parent2.candidate_id}",
            )

            logger.info(f"[Darwin] Crossover {parent1.candidate_id} x {parent2.candidate_id} -> {offspring_id}")
            return offspring

        except Exception as e:
            logger.warning(f"[Darwin] Crossover failed: {e}")
            return None

    # ──────── Evolution Loop ────────

    async def evolve_generation(
        self,
        target_module: str,
        performance_context: dict[str, Any],
        llm: Any,
    ) -> list[RewriteCandidate]:
        """Run one generation of evolution.

        Steps:
        1. Select parents from current population
        2. Apply crossover and mutation
        3. Evaluate new candidates
        4. Select survivors

        Returns:
            List of surviving candidates for next generation
        """
        logger.info(f"[Darwin] Evolving generation {len(self._generation_history) + 1}")

        # Phase 1: Selection
        parents = self._select_parents(self._population)
        logger.info(f"[Darwin] Selected {len(parents)} parents")

        # Phase 2: Variation (crossover + mutation)
        offspring = []

        # Crossover
        if len(parents) >= 2:
            for i in range(0, len(parents) - 1, 2):
                child = await self.crossover(parents[i], parents[i + 1], llm)
                if child:
                    offspring.append(child)

        # Mutation
        for parent in parents:
            if len(offspring) < self.config.population_size:
                mutant = await self.mutate(parent, llm)
                if mutant:
                    offspring.append(mutant)

        logger.info(f"[Darwin] Generated {len(offspring)} offspring")

        # Phase 3: Evaluation
        for child in offspring:
            await self.evaluate_candidate(child)

        # Phase 4: Selection (elitist)
        survivors = self._select_survivors(self._population, offspring)
        logger.info(f"[Darwin] Selected {len(survivors)} survivors")

        self._generation_history.append(survivors)
        self._population = survivors

        return survivors

    async def run_evolution(
        self,
        target_module: str,
        performance_context: dict[str, Any],
        llm: Any,
        generations: int | None = None,
    ) -> RewriteCandidate | None:
        """Run multi-generation evolution to find best rewrite.

        Args:
            target_module: Module to evolve
            performance_context: Performance data for context
            llm: LLM client
            generations: Number of generations (uses config.max_generations if None)

        Returns:
            Best candidate found, or None if evolution failed
        """
        if generations is None:
            generations = self.config.max_generations

        logger.info(f"[Darwin] Starting evolution for {target_module} ({generations} generations)")

        # Initialize population (generation 0)
        self._population = []
        self._generation_history = []

        for _ in range(self.config.population_size):
            candidate = await self.generate_rewrite_candidate(
                target_module, performance_context, llm
            )
            if candidate:
                await self.evaluate_candidate(candidate)
                self._population.append(candidate)

        logger.info(f"[Darwin] Initialized population with {len(self._population)} candidates")

        if not self._population:
            logger.error("[Darwin] Failed to initialize population")
            return None

        # Record baseline
        best_first_gen = max(self._population, key=lambda c: c.fitness_delta)
        self._baseline_fitness[target_module] = (
            best_first_gen.fitness_delta + 0.5
        )  # Baseline is current best

        # Evolve
        for gen in range(1, generations):
            survivors = await self.evolve_generation(
                target_module, performance_context, llm
            )
            if not survivors:
                logger.warning(f"[Darwin] Evolution stopped at generation {gen}")
                break

        # Save state after evolution completes
        self._save_evolution_state()

        # Return best overall
        best = max(self._population, key=lambda c: c.fitness_delta)
        logger.info(
            f"[Darwin] Evolution complete. Best candidate: {best.candidate_id} "
            f"(fitness_delta={best.fitness_delta:.3f})"
        )

        return best

    def _select_parents(self, population: list[RewriteCandidate]) -> list[RewriteCandidate]:
        """Tournament selection: pick best candidates for breeding.

        Uses tournament size of 2 (binary tournament).
        """
        import random

        if not population:
            return []

        parents = []
        tournament_size = min(2, len(population))

        for _ in range(min(2, len(population))):  # Select 2 parents
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda c: c.fitness_delta)
            parents.append(winner)

        return parents

    def _select_survivors(
        self,
        population: list[RewriteCandidate],
        offspring: list[RewriteCandidate],
    ) -> list[RewriteCandidate]:
        """Elitist selection: keep best from both parents and offspring.

        Returns:
            Best candidates up to population_size limit
        """
        combined = population + offspring
        combined.sort(key=lambda c: c.fitness_delta, reverse=True)

        survivors = [
            c for c in combined[:self.config.population_size]
            if c.fitness_delta >= self.config.fitness_threshold
        ]

        if not survivors:
            # Always keep at least the best one
            survivors = [combined[0]]

        return survivors

    # ──────── Apply Best Candidate ────────

    async def apply_best(self, constitution_dir: Path) -> bool:
        """Apply the best candidate found to the actual system.

        Uses SICAEngine to formally apply the rewrite as a SelfEditProposal.

        Args:
            constitution_dir: Directory containing constitution files

        Returns:
            True if successfully applied, False otherwise
        """
        if not self._population:
            logger.warning("[Darwin] No candidates to apply")
            return False

        best = max(self._population, key=lambda c: c.fitness_delta)

        if best.fitness_delta < self.config.fitness_threshold:
            logger.warning(
                f"[Darwin] Best candidate fitness ({best.fitness_delta:.3f}) "
                f"below threshold ({self.config.fitness_threshold})"
            )
            return False

        try:
            # Create a SelfEditProposal
            proposal = SelfEditProposal(
                id=best.candidate_id,
                edit_type="constitution",
                target_file=best.target_module,
                description=f"Darwin evolved {best.target_module}",
                original_content=best.original_code,
                proposed_content=best.rewritten_code,
                rationale="Evolved through multi-generation self-improvement",
                expected_impact=f"Fitness improvement of {best.fitness_delta:.2%}",
                confidence=0.7,
            )

            # Validate with SICA
            is_valid, reason = self.sica_engine.validate_proposal(proposal)
            if not is_valid:
                logger.warning("[Darwin] Proposal failed SICA validation")
                return False

            # Apply with SICA
            applied = await self.sica_engine.apply_proposal(
                proposal, constitution_dir
            )
            if applied:
                logger.info(f"[Darwin] Successfully applied best candidate {best.candidate_id}")
                return True
            else:
                logger.warning("[Darwin] SICA failed to apply proposal")
                return False

        except Exception as e:
            logger.error(f"[Darwin] Failed to apply best candidate: {e}")
            return False

    # ──────── Statistics & Persistence ────────

    def get_evolution_stats(self) -> dict[str, Any]:
        """Get statistics about evolution runs.

        Returns:
            Dict with per-generation statistics
        """
        if not self._generation_history:
            return {"message": "No evolution history"}

        stats = {
            "total_generations": len(self._generation_history),
            "per_generation": [],
        }

        for gen, candidates in enumerate(self._generation_history):
            gen_stats = {
                "generation": gen,
                "population_size": len(candidates),
                "avg_fitness": sum(c.fitness_delta for c in candidates) / len(candidates) if candidates else 0,
                "max_fitness": max((c.fitness_delta for c in candidates), default=0),
                "best_candidate_id": max(
                    (c.candidate_id for c in candidates),
                    key=lambda cid: max((c.fitness_delta for c in candidates if c.candidate_id == cid), default=0),
                    default=None,
                ),
            }
            stats["per_generation"].append(gen_stats)

        # Overall best
        all_candidates = [c for gen in self._generation_history for c in gen]
        if all_candidates:
            best = max(all_candidates, key=lambda c: c.fitness_delta)
            stats["best_overall"] = {
                "candidate_id": best.candidate_id,
                "fitness_delta": best.fitness_delta,
                "generation": best.generation,
                "target_module": best.target_module,
            }

        return stats

    def _save_evolution_state(self) -> None:
        """Save evolution history to disk."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "generation_history": [
                    [c.to_dict() for c in gen]
                    for gen in self._generation_history
                ],
                "baseline_fitness": self._baseline_fitness,
                "timestamp": time.time(),
            }
            self._evolution_state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"[Darwin] Saved evolution state to {self._evolution_state_file}")
        except Exception as e:
            logger.warning(f"[Darwin] Could not save evolution state: {e}")

    def _load_evolution_state(self) -> None:
        """Load evolution history from disk, decompressing code."""
        if not self._evolution_state_file.exists():
            return
        try:
            data = json.loads(self._evolution_state_file.read_text(encoding="utf-8"))
            for gen_data in data.get("generation_history", []):
                generation = []
                for candidate_dict in gen_data:
                    # Decompress code if it exists
                    original_code = ""
                    rewritten_code = ""

                    if "original_code" in candidate_dict and candidate_dict["original_code"]:
                        try:
                            original_code = zlib.decompress(
                                base64.b64decode(candidate_dict["original_code"])
                            ).decode('utf-8')
                        except Exception as e:
                            logger.warning(f"[Darwin] Failed to decompress original_code: {e}")

                    if "rewritten_code" in candidate_dict and candidate_dict["rewritten_code"]:
                        try:
                            rewritten_code = zlib.decompress(
                                base64.b64decode(candidate_dict["rewritten_code"])
                            ).decode('utf-8')
                        except Exception as e:
                            logger.warning(f"[Darwin] Failed to decompress rewritten_code: {e}")

                    candidate = RewriteCandidate(
                        candidate_id=candidate_dict["candidate_id"],
                        target_module=candidate_dict["target_module"],
                        original_code=original_code,
                        rewritten_code=rewritten_code,
                        test_results=candidate_dict.get("test_results", {}),
                        fitness_delta=candidate_dict["fitness_delta"],
                        generation=candidate_dict["generation"],
                        parent_id=candidate_dict.get("parent_id", ""),
                    )
                    generation.append(candidate)
                self._generation_history.append(generation)

            self._baseline_fitness = data.get("baseline_fitness", {})
            logger.info(f"[Darwin] Loaded evolution state with {len(self._generation_history)} generations")
        except Exception as e:
            logger.debug(f"[Darwin] Could not load evolution state: {e}")

