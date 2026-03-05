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


# ──────────────────────────────────────────────
# Failure Pattern Mining & Bayesian Confidence
# ──────────────────────────────────────────────


class FailurePatternMiner:
    """Mine frequent failure patterns using association rules.

    Uses Apriori-style algorithm to find common combinations of
    failure tags (e.g., 'import_error' + 'timeout' may indicate
    a specific architectural issue).
    """

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5) -> None:
        """Initialize pattern miner.

        Args:
            min_support: Minimum support threshold (fraction of failures)
            min_confidence: Minimum confidence for patterns
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._failure_log: list[set[str]] = []

    def add_failure(self, tags: set[str]) -> None:
        """Record a failure with associated tags."""
        self._failure_log.append(tags)

    def mine_patterns(self) -> list[dict[str, Any]]:
        """Find frequent failure patterns using Apriori-like algorithm.

        Returns list of patterns with structure:
        {
            "pattern": set of tags,
            "support": fraction of failures containing this pattern,
            "confidence": conditional probability,
            "count": number of failures with this pattern,
        }
        """
        if len(self._failure_log) < 5:
            return []

        total = len(self._failure_log)

        # Count single items
        item_counts: dict[str, int] = {}
        for tags in self._failure_log:
            for tag in tags:
                item_counts[tag] = item_counts.get(tag, 0) + 1

        # Frequent singles
        freq_1 = {item for item, count in item_counts.items()
                  if count / total >= self.min_support}

        # Mine frequent pairs
        patterns = []
        items = sorted(list(freq_1))

        for i, a in enumerate(items):
            for b in items[i+1:]:
                pair_count = sum(1 for tags in self._failure_log if {a, b} <= tags)
                support = pair_count / total

                if support >= self.min_support:
                    # Confidence: P(b|a) or P(a|b), take the max
                    conf_ab = pair_count / max(item_counts[a], 1)
                    conf_ba = pair_count / max(item_counts[b], 1)
                    max_conf = max(conf_ab, conf_ba)

                    if max_conf >= self.min_confidence:
                        patterns.append({
                            "pattern": {a, b},
                            "support": support,
                            "confidence": max_conf,
                            "count": pair_count,
                        })

        # Sort by confidence
        patterns.sort(key=lambda x: x["confidence"], reverse=True)
        return patterns


class BayesianRuleConfidence:
    """Track rule confidence using Beta-Binomial model.

    Uses Thompson Sampling for exploration-exploitation and
    provides credible intervals for uncertainty quantification.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Initialize with Beta prior.

        Args:
            prior_alpha: Alpha parameter of Beta prior
            prior_beta: Beta parameter of Beta prior
        """
        self.alpha = prior_alpha
        self.beta = prior_beta

    def update(self, success: bool) -> None:
        """Update posterior with observation."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    @property
    def mean(self) -> float:
        """Expected value of the posterior Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% credible interval via normal approximation.

        For large sample sizes, uses normal approximation to Beta.
        """
        n = self.alpha + self.beta
        p = self.mean

        # Standard error using Beta variance formula
        se = (p * (1 - p) / n) ** 0.5 if n > 1 else 0.5

        lower = max(0.0, p - 1.96 * se)
        upper = min(1.0, p + 1.96 * se)

        return (lower, upper)

    def sample(self) -> float:
        """Thompson Sampling: sample from Beta posterior.

        Useful for exploration-exploitation strategies.
        """
        import random
        return random.betavariate(self.alpha, self.beta)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
        }


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
    project_name: str = ""     # Project that generated this patch (for grouping)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_role": self.target_role,
            "content": self.content,
            "source": self.source,
            "priority": self.priority,
            "project_specific": self.project_specific,
            "project_name": self.project_name,
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

        # Track rule performance deltas for causal inference
        # Maps rule_id → list of (score_with_rule, baseline_score) tuples
        self._rule_deltas: dict[str, list[tuple[float, float]]] = {}

        # Algorithmic components for rule discovery
        self._failure_pattern_miner = FailurePatternMiner(min_support=0.1, min_confidence=0.5)
        self._rule_confidence_tracker: dict[str, BayesianRuleConfidence] = {}

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
                    # Priority formula: 10 - i means first LLM outputs get highest priority.
                    # LLMs typically front-load the most important guidance, so this ordering
                    # ensures critical instructions are injected early in agent prompts.
                    patch = ConstitutionPatch(
                        id=f"spec-{project_name}-{i:02d}",
                        target_role=item.get("role", "builder"),
                        content=item.get("instruction", ""),
                        source="director",
                        priority=10 - i,
                        project_name=project_name,
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
                new_pattern = data.get("pattern", "")
                new_rule = data.get("rule", "")
                new_confidence = float(data.get("confidence", 0.5))
                source_role = failure_context.get("agent_role", "")

                # Check for duplicate rules with >70% word overlap in pattern
                existing_rule = self._find_similar_rule(
                    new_pattern, source_role, similarity_threshold=0.7
                )

                if existing_rule:
                    # Update confidence of existing rule instead of creating duplicate
                    old_confidence = existing_rule.confidence
                    existing_rule.confidence = (old_confidence + new_confidence) / 2
                    self._save_knowledge_base()
                    logger.info(
                        f"[MetaLearning] Updated existing rule '{existing_rule.id}': "
                        f"confidence {old_confidence:.2f} → {existing_rule.confidence:.2f}"
                    )
                    return existing_rule
                else:
                    # Create new rule
                    rule = LearnedRule(
                        id=f"rule-{int(time.time()) % 100000}",
                        pattern=new_pattern,
                        rule=new_rule,
                        source_role=source_role,
                        confidence=new_confidence,
                    )
                    self._learned_rules.append(rule)
                    self._save_knowledge_base()
                    logger.info(f"[MetaLearning] New rule '{rule.id}': {rule.rule[:80]}...")
                    return rule

        except Exception as e:
            logger.warning(f"[MetaLearning] Failed to learn from failure: {e}")

        return None

    @staticmethod
    def _wilson_score_lower(successes: int, total: int, z: float = 1.96) -> float:
        """Wilson score interval lower bound for binomial proportion.

        Computes a more conservative confidence interval than simple success ratio,
        especially valuable for small sample sizes. Uses the Wilson score interval
        (also known as the score-based confidence interval) which is more accurate
        than the normal approximation.

        Args:
            successes: Number of successes (wins)
            total: Total number of trials
            z: Z-score for desired confidence level (1.96 for 95% CI)

        Returns:
            Lower bound of the Wilson confidence interval [0, 1]
        """
        if total == 0:
            return 0.0

        p_hat = successes / total
        denominator = 1 + (z * z) / total

        centre_adjusted_success = p_hat + (z * z) / (2 * total)
        adjusted_standard_error = (
            (p_hat * (1 - p_hat) + z * z / (4 * total)) / total
        ) ** 0.5

        lower_bound = (
            (centre_adjusted_success - z * adjusted_standard_error) / denominator
        )

        return max(0.0, lower_bound)

    def record_rule_outcome_with_metrics(
        self, rule_id: str, score_with_rule: float, baseline_score: float,
    ) -> None:
        """Record rule outcome with actual causal metrics.

        Instead of binary helped/not-helped, this tracks the actual performance delta:
        score_with_rule - baseline_score. Uses Wilson score interval for confidence
        estimation, which accounts for sample size and is more statistically rigorous
        than naive success ratios.

        Args:
            rule_id: ID of the learned rule
            score_with_rule: Performance metric with the rule applied (e.g., test pass rate)
            baseline_score: Performance metric without the rule (baseline for comparison)
        """
        # Initialize delta tracking if not present
        if rule_id not in self._rule_deltas:
            self._rule_deltas[rule_id] = []

        # Record the delta observation (cap at 50 to prevent unbounded growth)
        self._rule_deltas[rule_id].append((score_with_rule, baseline_score))
        if len(self._rule_deltas[rule_id]) > 50:
            self._rule_deltas[rule_id] = self._rule_deltas[rule_id][-50:]

        # Find and update the rule
        for rule in self._learned_rules:
            if rule.id == rule_id:
                rule.times_applied += 1

                # Calculate whether this observation was "helpful"
                delta = score_with_rule - baseline_score
                if delta > 0:
                    rule.times_helped += 1

                # Update confidence using delta-based causal inference
                self._update_rule_confidence_from_deltas(rule)
                self._save_knowledge_base()
                break

    def _update_rule_confidence_from_deltas(self, rule: LearnedRule) -> None:
        """Update a rule's confidence based on recorded performance deltas.

        Uses the average delta and Wilson score interval to compute a confidence
        that reflects both the magnitude of improvement and the statistical certainty.
        """
        rule_id = rule.id
        if rule_id not in self._rule_deltas or not self._rule_deltas[rule_id]:
            return

        deltas = [score_with - baseline for score_with, baseline in self._rule_deltas[rule_id]]
        avg_delta = sum(deltas) / len(deltas)

        # Count how many observations had positive delta (helped)
        successes = sum(1 for d in deltas if d > 0)
        total = len(deltas)

        # Use Wilson score for statistically sound confidence
        wilson_lower = self._wilson_score_lower(successes, total)

        # Confidence = (delta > 0) AND (statistical significance via Wilson)
        # If avg_delta is negative, confidence should be low
        if avg_delta <= 0:
            rule.confidence = max(0.0, wilson_lower * 0.5)  # Penalize negative deltas
        else:
            # For positive deltas, use Wilson lower bound as confidence
            rule.confidence = wilson_lower

        logger.debug(
            f"[MetaLearning] Updated confidence for rule {rule.id}: "
            f"avg_delta={avg_delta:.4f}, wilson_lower={wilson_lower:.4f}, "
            f"confidence={rule.confidence:.4f}"
        )

    def add_failure(self, failure_tags: set[str]) -> None:
        """Record a failure with associated tags for pattern mining.

        Args:
            failure_tags: Set of tags describing the failure
                          e.g., {'timeout', 'import_error', 'async_issue'}
        """
        self._failure_pattern_miner.add_failure(failure_tags)

    def discover_rules_from_patterns(self) -> list[LearnedRule]:
        """Mine failure patterns and create preventive rules.

        Returns list of newly discovered rules (if any).
        """
        patterns = self._failure_pattern_miner.mine_patterns()
        new_rules = []

        for pattern in patterns:
            tags = sorted(list(pattern["pattern"]))
            tag_str = " + ".join(tags)

            # Generate rule text from pattern
            rule_text = f"Watch for combination of: {tag_str}. "
            rule_text += f"This pattern occurs in {pattern['support']:.1%} of failures. "
            rule_text += "Implement targeted prevention steps."

            # Create rule
            rule = LearnedRule(
                id=f"pattern-{int(time.time()) % 100000}",
                pattern=f"Failure pattern: {tag_str}",
                rule=rule_text,
                source_role="system",
                confidence=pattern["confidence"],
            )

            self._learned_rules.append(rule)
            new_rules.append(rule)

            logger.info(
                f"[MetaLearning] Discovered pattern-based rule: {rule.id} "
                f"(confidence={rule.confidence:.2f})"
            )

        if new_rules:
            self._save_knowledge_base()

        return new_rules

    def update_rule_confidence_bayesian(self, rule_id: str, success: bool) -> None:
        """Update rule confidence using Bayesian model.

        Args:
            rule_id: ID of the rule
            success: Whether applying the rule led to success
        """
        if rule_id not in self._rule_confidence_tracker:
            self._rule_confidence_tracker[rule_id] = BayesianRuleConfidence()

        tracker = self._rule_confidence_tracker[rule_id]
        tracker.update(success)

        # Find and update the rule's confidence
        for rule in self._learned_rules:
            if rule.id == rule_id:
                rule.confidence = tracker.mean
                ci_lower, ci_upper = tracker.confidence_interval
                logger.debug(
                    f"[MetaLearning] Updated {rule_id} confidence: {rule.confidence:.2f} "
                    f"(95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])"
                )
                break

    def get_rule_credible_interval(self, rule_id: str) -> tuple[float, float] | None:
        """Get 95% credible interval for a rule's effectiveness.

        Returns (lower, upper) or None if rule not tracked.
        """
        if rule_id in self._rule_confidence_tracker:
            return self._rule_confidence_tracker[rule_id].confidence_interval
        return None

    def record_rule_outcome(self, rule_id: str, helped: bool) -> None:
        """Record whether a learned rule helped or not.

        This feedback loop improves rule quality over time.

        For backward compatibility, this accepts a binary flag. Internally,
        it converts the boolean to moderate synthetic deltas (±0.1 around 0.5
        baseline) so that binary feedback doesn't swamp real metric deltas
        when both are used for the same rule.

        Prefer using record_rule_outcome_with_metrics() for more accurate
        causal inference with real performance metrics.
        """
        if helped:
            # Moderate positive delta (+0.1)
            self.record_rule_outcome_with_metrics(rule_id, score_with_rule=0.6, baseline_score=0.5)
        else:
            # Moderate negative delta (-0.1)
            self.record_rule_outcome_with_metrics(rule_id, score_with_rule=0.4, baseline_score=0.5)

    def get_rule_causal_evidence(self, rule_id: str) -> dict[str, Any]:
        """Get causal evidence summary for a rule.

        Returns a dictionary summarizing the causal evidence that a rule is effective:
        - avg_delta: Average improvement from applying the rule
        - n_observations: Number of times the rule has been evaluated
        - wilson_lower: Wilson score interval lower bound (statistically conservative estimate)
        - is_causal: Boolean indicating if the rule meets causal evidence threshold

        A rule is considered "causal" if:
        1. avg_delta > 0 (positive average effect)
        2. wilson_lower > 0 (statistically significant at 95% confidence)
        3. n_observations >= 3 (minimum sample size for reliability)

        Returns empty dict if rule not found or no delta data available.
        """
        if rule_id not in self._rule_deltas or not self._rule_deltas[rule_id]:
            return {}

        deltas = [score_with - baseline for score_with, baseline in self._rule_deltas[rule_id]]
        avg_delta = sum(deltas) / len(deltas)
        n_observations = len(deltas)

        # Count successes for Wilson score
        successes = sum(1 for d in deltas if d > 0)
        wilson_lower = self._wilson_score_lower(successes, n_observations)

        # Threshold for causality: positive delta, statistically significant success
        # rate (Wilson lower bound > 0.25 means we're 95% confident the true success
        # rate exceeds 25%, a conservative gate), and minimum sample size.
        is_causal = (avg_delta > 0) and (wilson_lower > 0.25) and (n_observations >= 3)

        return {
            "rule_id": rule_id,
            "avg_delta": avg_delta,
            "n_observations": n_observations,
            "wilson_lower": wilson_lower,
            "is_causal": is_causal,
            "successes": successes,
        }

    def _find_similar_rule(
        self, pattern: str, source_role: str, similarity_threshold: float = 0.7
    ) -> LearnedRule | None:
        """Find an existing rule with similar pattern and source_role.

        Uses word overlap to detect duplicates: if >similarity_threshold of words
        match between patterns, returns the existing rule. Otherwise returns None.
        """
        if not pattern:
            return None

        pattern_words = set(pattern.lower().split())
        if not pattern_words:
            return None

        for rule in self._learned_rules:
            if rule.source_role != source_role:
                continue

            rule_words = set(rule.pattern.lower().split())
            if not rule_words:
                continue

            # Calculate word overlap ratio
            overlap = len(pattern_words & rule_words)
            max_len = max(len(pattern_words), len(rule_words))
            if max_len == 0:
                continue

            overlap_ratio = overlap / max_len
            if overlap_ratio >= similarity_threshold:
                return rule

        return None

    def _get_applicable_rules(self, role: str) -> list[LearnedRule]:
        """Get learned rules applicable to a role, with confidence above threshold.

        Applies temporal decay: rules older than 30 days have their effective
        confidence reduced by 50% to prevent stale rules from dominating.

        Prefers delta-based confidence (from causal metrics) over binary confidence
        when available. Falls back to traditional confidence ratio for backward
        compatibility.
        """
        current_time = time.time()
        applicable_rules = []

        for r in self._learned_rules:
            # Check role applicability
            if r.source_role and r.source_role != role:
                continue

            # Prefer delta-based confidence if available
            if r.id in self._rule_deltas and self._rule_deltas[r.id]:
                # Use the causal evidence as confidence
                causal_evidence = self.get_rule_causal_evidence(r.id)
                base_confidence = causal_evidence.get("wilson_lower", r.confidence)
            else:
                # Fall back to traditional ratio-based confidence
                base_confidence = r.confidence

            # Apply temporal decay: rules >30 days old get 50% confidence reduction
            days_since_created = (current_time - r.created_at) / (24 * 3600)
            decay_factor = 0.5 if days_since_created > 30 else 1.0
            effective_confidence = base_confidence * decay_factor

            # Only include rules that meet the confidence threshold after decay
            if effective_confidence >= 0.3:
                applicable_rules.append(r)

        return applicable_rules

    def _save_knowledge_base(self) -> None:
        """Persist learned rules to disk.

        Now persists ALL patches (including project_specific ones) with their project_name
        for cross-project learning and knowledge reuse.

        Also persists:
        - Rule deltas for causal inference analysis across sessions
        - Bayesian rule confidence trackers for Thompson Sampling
        - Failure pattern miner state for continued pattern discovery
        """
        if not self._knowledge_base_path:
            logger.warning("[Constitution] Cannot save knowledge base: no knowledge_base_path configured")
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
                "patches": [p.to_dict() for p in self._patches],  # Now includes ALL patches with project_name
                "rule_deltas": {
                    rule_id: [
                        {"score_with_rule": sw, "baseline_score": bs}
                        for sw, bs in deltas
                    ]
                    for rule_id, deltas in self._rule_deltas.items()
                },
                "bayesian_confidence": {
                    rule_id: tracker.to_dict()
                    for rule_id, tracker in self._rule_confidence_tracker.items()
                },
                "failure_log_size": len(self._failure_pattern_miner._failure_log),
            }
            self._knowledge_base_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"Could not save knowledge base: {e}")

    def _load_knowledge_base(self) -> None:
        """Load persisted rules from disk.

        Restores:
        - Learned rules with their confidence scores
        - Causal evaluation history (deltas) for continued training
        - Bayesian rule confidence trackers (for Thompson Sampling)
        - Failure pattern counts (for continued pattern discovery)
        """
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

            # Load rule deltas for causal inference
            for rule_id, delta_list in data.get("rule_deltas", {}).items():
                self._rule_deltas[rule_id] = [
                    (d["score_with_rule"], d["baseline_score"])
                    for d in delta_list
                ]

            # Load Bayesian confidence trackers
            for rule_id, tracker_data in data.get("bayesian_confidence", {}).items():
                tracker = BayesianRuleConfidence(
                    prior_alpha=1.0, prior_beta=1.0
                )
                # Restore posterior from persisted state
                tracker.alpha = tracker_data.get("alpha", 1.0)
                tracker.beta = tracker_data.get("beta", 1.0)
                self._rule_confidence_tracker[rule_id] = tracker

            logger.info(
                f"[Constitution] Loaded {len(self._learned_rules)} learned rules, "
                f"{len(self._rule_deltas)} rule delta histories, and "
                f"{len(self._rule_confidence_tracker)} Bayesian confidence trackers"
            )
        except Exception as e:
            logger.debug(f"Could not load knowledge base: {e}")
