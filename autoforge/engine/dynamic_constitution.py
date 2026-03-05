"""Dynamic constitution — adaptive prompt generation based on project context.

Instead of static markdown constitution files, this module lets the Director
generate project-specific supplementary instructions that are injected into
agent system prompts at runtime.

Example: If the spec involves WebSocket-heavy code, the Builder's system
prompt gets an appendix about async error handling, connection lifecycle, etc.

This also implements the self-skill generation mechanism: when agents encounter
repeated failure patterns, they can generate new constitution rules or tool
wrappers that are saved to a project-level knowledge base.

Reference: Meta-learning through code — agents extend their own capabilities
by writing rules/tools, not by changing model weights.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConstitutionPatch:
    """A dynamically generated supplement to an agent's constitution."""
    id: str
    target_role: str           # Which agent this applies to ("builder", "architect", etc.)
    content: str               # The actual instruction text
    source: str                # Where it came from ("director", "meta_learning", "user")
    priority: int = 0          # Higher = more important (injected earlier in prompt)
    project_specific: bool = True  # Only applies to current project
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_role": self.target_role,
            "content": self.content,
            "source": self.source,
            "priority": self.priority,
            "project_specific": self.project_specific,
        }


@dataclass
class LearnedRule:
    """A rule learned from failure patterns — part of the self-skill system."""
    id: str
    pattern: str              # What failure pattern this addresses
    rule: str                 # The rule/instruction to prevent it
    source_role: str = ""     # Which agent role this rule originated from (empty = all)
    confidence: float = 0.0   # How well this rule works (updated over time)
    times_applied: int = 0
    times_helped: int = 0     # Times it prevented the failure
    created_at: float = field(default_factory=time.time)


class DynamicConstitution:
    """Manages dynamic constitution patches and learned rules.

    Workflow:
    1. After SPEC phase: Director generates project-specific patches
    2. During BUILD: patches are injected into agent system prompts
    3. On failure: meta-learning analyzes patterns and creates new rules
    4. Rules persist to disk for cross-project learning
    """

    def __init__(self, project_dir: Path | None = None) -> None:
        self._patches: list[ConstitutionPatch] = []
        self._learned_rules: list[LearnedRule] = []
        self.project_dir = project_dir
        self._knowledge_base_path = project_dir / ".autoforge" / "knowledge_base.json" if project_dir else None

        # Load persisted knowledge if available
        self._load_knowledge_base()

    def add_patch(self, patch: ConstitutionPatch) -> None:
        """Add a dynamic constitution patch."""
        self._patches.append(patch)
        logger.info(f"[Constitution] Added patch '{patch.id}' for {patch.target_role}: "
                     f"{patch.content[:80]}...")

    def get_patches_for_role(self, role: str) -> list[ConstitutionPatch]:
        """Get all patches applicable to a specific agent role."""
        patches = [p for p in self._patches if p.target_role == role or p.target_role == "all"]
        patches.sort(key=lambda p: p.priority, reverse=True)
        return patches

    def build_supplementary_prompt(self, role: str) -> str:
        """Build the supplementary prompt section for an agent.

        This gets appended to the agent's static system prompt.
        """
        patches = self.get_patches_for_role(role)
        rules = self._get_applicable_rules(role)

        if not patches and not rules:
            return ""

        parts = ["\n\n## Dynamic Instructions (Project-Specific)\n"]

        if patches:
            parts.append("### Context-Aware Guidelines\n")
            for patch in patches:
                parts.append(f"- {patch.content}\n")

        if rules:
            parts.append("\n### Learned Best Practices\n")
            parts.append("(These rules were learned from past experience — follow them.)\n")
            for rule in rules:
                parts.append(f"- {rule.rule}\n")

        return "".join(parts)

    async def generate_patches_from_spec(
        self,
        spec: dict[str, Any],
        llm: Any,
    ) -> list[ConstitutionPatch]:
        """Have the Director analyze the spec and generate context-specific patches.

        This is called once after the SPEC phase completes.
        """
        from autoforge.engine.llm_router import TaskComplexity

        tech_stack = json.dumps(spec.get("tech_stack", {}), indent=2)
        modules = json.dumps(spec.get("modules", []), indent=2)
        project_name = spec.get("project_name", "project")

        prompt = (
            f"Analyze this project specification and generate targeted instructions "
            f"for the development agents.\n\n"
            f"## Project: {project_name}\n"
            f"## Tech Stack\n```json\n{tech_stack}\n```\n\n"
            f"## Modules\n```json\n{modules}\n```\n\n"
            f"## Instructions\n"
            f"Generate specific, actionable guidelines for each agent role based on "
            f"this project's unique characteristics. Focus on:\n"
            f"- Common pitfalls for this tech stack\n"
            f"- Integration points between modules\n"
            f"- Performance considerations\n"
            f"- Security concerns specific to this project type\n\n"
            f"Output a JSON array of patches:\n"
            f"```json\n"
            f"[\n"
            f'  {{"role": "builder", "instruction": "Use connection pooling for database..."}},\n'
            f'  {{"role": "architect", "instruction": "Ensure auth module is dependency for..."}},\n'
            f"  ...\n"
            f"]\n"
            f"```\n"
            f"Generate 3-8 high-value instructions. Be specific, not generic."
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a senior tech lead preparing project-specific guidelines.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            from autoforge.engine.utils import extract_json_list_from_text
            try:
                raw = extract_json_list_from_text(text)
            except ValueError:
                raw = []

            patches = []
            for i, item in enumerate(raw):
                if isinstance(item, dict):
                    patch = ConstitutionPatch(
                        id=f"spec-{project_name}-{i:02d}",
                        target_role=item.get("role", "builder"),
                        content=item.get("instruction", ""),
                        source="director",
                        priority=10 - i,  # Earlier items are higher priority
                    )
                    self.add_patch(patch)
                    patches.append(patch)

            logger.info(f"[Constitution] Generated {len(patches)} patches from spec")
            return patches

        except Exception as e:
            logger.warning(f"[Constitution] Failed to generate patches: {e}")
            return []

    async def learn_from_failure(
        self,
        failure_context: dict[str, Any],
        llm: Any,
    ) -> LearnedRule | None:
        """Analyze a failure and create a new rule to prevent it in the future.

        This is the core of the self-skill / meta-learning system.

        Args:
            failure_context: {
                "task_description": str,
                "error": str,
                "agent_role": str,
                "files_involved": list[str],
                "turn_count": int,
                "approach_used": str,
            }
        """
        from autoforge.engine.llm_router import TaskComplexity

        prompt = (
            f"A development agent has failed. Analyze the failure and create a "
            f"reusable rule to prevent this in the future.\n\n"
            f"## Failure Details\n"
            f"Agent: {failure_context.get('agent_role', 'builder')}\n"
            f"Task: {failure_context.get('task_description', '')}\n"
            f"Error:\n```\n{failure_context.get('error', '')[:1000]}\n```\n"
            f"Approach: {failure_context.get('approach_used', '')}\n"
            f"Turns taken: {failure_context.get('turn_count', 0)}\n\n"
            f"## Instructions\n"
            f"1. Identify the root cause of failure\n"
            f"2. Create a concise, actionable rule that would prevent this\n"
            f"3. The rule should be general enough to apply to similar situations\n\n"
            f"Output JSON:\n"
            f'{{"pattern": "description of failure pattern", '
            f'"rule": "the preventive rule", '
            f'"confidence": 0.0-1.0}}'
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a post-mortem analyst. Create clear, actionable rules "
                       "from failure analysis. Be specific and concise.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            import re
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                rule = LearnedRule(
                    id=f"rule-{int(time.time()) % 100000}",
                    pattern=data.get("pattern", ""),
                    rule=data.get("rule", ""),
                    confidence=float(data.get("confidence", 0.5)),
                )
                self._learned_rules.append(rule)
                self._save_knowledge_base()

                logger.info(f"[MetaLearning] New rule '{rule.id}': {rule.rule[:80]}...")
                return rule

        except Exception as e:
            logger.warning(f"[MetaLearning] Failed to learn from failure: {e}")

        return None

    def record_rule_outcome(self, rule_id: str, helped: bool) -> None:
        """Record whether a learned rule helped or not.

        This feedback loop improves rule quality over time.
        """
        for rule in self._learned_rules:
            if rule.id == rule_id:
                rule.times_applied += 1
                if helped:
                    rule.times_helped += 1
                # Update confidence based on success rate
                if rule.times_applied > 0:
                    rule.confidence = rule.times_helped / rule.times_applied
                self._save_knowledge_base()
                break

    def _get_applicable_rules(self, role: str) -> list[LearnedRule]:
        """Get learned rules applicable to a role, with confidence above threshold."""
        return [
            r for r in self._learned_rules
            if r.confidence >= 0.3  # Only include rules that have shown some value
            and (not r.source_role or r.source_role == role)
        ]

    def _save_knowledge_base(self) -> None:
        """Persist learned rules to disk."""
        if not self._knowledge_base_path:
            return
        try:
            self._knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "learned_rules": [
                    {
                        "id": r.id,
                        "pattern": r.pattern,
                        "rule": r.rule,
                        "source_role": r.source_role,
                        "confidence": r.confidence,
                        "times_applied": r.times_applied,
                        "times_helped": r.times_helped,
                        "created_at": r.created_at,
                    }
                    for r in self._learned_rules
                ],
                "patches": [p.to_dict() for p in self._patches if not p.project_specific],
            }
            self._knowledge_base_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"Could not save knowledge base: {e}")

    def _load_knowledge_base(self) -> None:
        """Load persisted rules from disk."""
        if not self._knowledge_base_path or not self._knowledge_base_path.exists():
            return
        try:
            data = json.loads(self._knowledge_base_path.read_text(encoding="utf-8"))
            for r in data.get("learned_rules", []):
                self._learned_rules.append(LearnedRule(
                    id=r["id"],
                    pattern=r["pattern"],
                    rule=r["rule"],
                    source_role=r.get("source_role", ""),
                    confidence=r.get("confidence", 0.5),
                    times_applied=r.get("times_applied", 0),
                    times_helped=r.get("times_helped", 0),
                    created_at=r.get("created_at", 0),
                ))
            logger.info(f"[Constitution] Loaded {len(self._learned_rules)} learned rules")
        except Exception as e:
            logger.debug(f"Could not load knowledge base: {e}")
