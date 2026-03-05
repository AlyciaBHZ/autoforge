"""Academic peer review simulation — multi-reviewer evaluation pipeline.

This module simulates realistic academic peer review for research articles:

1. **Multi-role Reviewers**: Each reviewer has a specialized role (Expert,
   Methodology, Clarity, Novelty, Reproducibility, Meta-Reviewer) with
   role-specific review criteria and scoring frameworks.

2. **Structured Review Process**:
   - Individual reviews with detailed comments and scores
   - Author rebuttal to reviewer concerns
   - Meta-review synthesizing individual reviews and rebuttal
   - Optional iterative revision and re-review cycles

3. **Scoring Framework**: 5-point scales for soundness, presentation,
   contribution, novelty, reproducibility; 10-point overall scale with
   confidence ratings and acceptance decisions.

4. **Review Roles**:
   - EXPERT_REVIEWER: Domain expertise, technical correctness
   - METHODOLOGY_REVIEWER: Experimental design, statistical rigor
   - CLARITY_REVIEWER: Presentation, writing quality, figures
   - NOVELTY_REVIEWER: Originality, comparison to related work
   - REPRODUCIBILITY_REVIEWER: Code availability, details, datasets
   - META_REVIEWER: Synthesizes into recommendation (area chair)

References:
  - NeurIPS/ICML review guidelines
  - OpenReview review standards
  - Journal of Machine Learning Research review process
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════


class ReviewerRole(str, Enum):
    """Specialized reviewer roles in academic peer review."""
    EXPERT_REVIEWER = "expert_reviewer"
    METHODOLOGY_REVIEWER = "methodology_reviewer"
    CLARITY_REVIEWER = "clarity_reviewer"
    NOVELTY_REVIEWER = "novelty_reviewer"
    REPRODUCIBILITY_REVIEWER = "reproducibility_reviewer"
    META_REVIEWER = "meta_reviewer"


# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════


@dataclass
class ReviewScore:
    """Numerical scores for a peer review."""
    soundness: int  # 1-5: Is the methodology sound?
    presentation: int  # 1-5: Is it well-written?
    contribution: int  # 1-5: Does it contribute to field?
    novelty: int  # 1-5: How original is the work?
    reproducibility: int  # 1-5: Can results be reproduced?
    overall: int  # 1-10: Overall assessment
    confidence: int  # 1-5: Reviewer confidence
    decision: str  # strong_accept, accept, weak_accept, borderline, weak_reject, reject, strong_reject

    def to_dict(self) -> dict[str, Any]:
        return {
            "soundness": self.soundness,
            "presentation": self.presentation,
            "contribution": self.contribution,
            "novelty": self.novelty,
            "reproducibility": self.reproducibility,
            "overall": self.overall,
            "confidence": self.confidence,
            "decision": self.decision,
        }


@dataclass
class ReviewComment:
    """A single comment in a peer review."""
    section: str  # "summary", "strengths", "weaknesses", "questions", "suggestions"
    comment_type: str  # strength, weakness, question, suggestion, minor
    text: str
    severity: str  # critical, major, minor


@dataclass
class PeerReview:
    """A complete peer review from one reviewer."""
    reviewer_id: str
    reviewer_role: ReviewerRole
    scores: ReviewScore
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    questions: list[str]
    detailed_comments: list[ReviewComment]
    requested_changes: list[str]
    ethics_concerns: list[str]
    review_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "reviewer_role": self.reviewer_role.value,
            "scores": self.scores.to_dict(),
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "questions": self.questions,
            "detailed_comments": [
                {
                    "section": c.section,
                    "comment_type": c.comment_type,
                    "text": c.text,
                    "severity": c.severity,
                }
                for c in self.detailed_comments
            ],
            "requested_changes": self.requested_changes,
            "ethics_concerns": self.ethics_concerns,
            "review_time": self.review_time,
        }


@dataclass
class AuthorRebuttal:
    """Author's point-by-point response to reviews."""
    original_reviews: list[PeerReview]
    point_by_point_responses: list[dict[str, str]]  # maps concern to response
    changes_made: list[str]
    changes_declined: list[dict[str, str]]  # reason for decline

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_reviews": len(self.original_reviews),
            "point_by_point_responses": self.point_by_point_responses,
            "changes_made": self.changes_made,
            "changes_declined": self.changes_declined,
        }


@dataclass
class MetaReview:
    """Area chair synthesis of all reviews and rebuttal."""
    reviewer_id: str = "meta_reviewer"
    individual_reviews: list[PeerReview] = field(default_factory=list)
    rebuttal: AuthorRebuttal | None = None
    consensus_scores: ReviewScore = field(
        default_factory=lambda: ReviewScore(3, 3, 3, 3, 3, 6, 3, "borderline")
    )
    key_concerns: list[str] = field(default_factory=list)
    key_strengths: list[str] = field(default_factory=list)
    recommendation: str = "borderline"  # accept, reject, revise
    confidence: float = 0.5
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "num_reviews": len(self.individual_reviews),
            "consensus_scores": self.consensus_scores.to_dict(),
            "key_concerns": self.key_concerns,
            "key_strengths": self.key_strengths,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "summary": self.summary,
        }


# ══════════════════════════════════════════════════════════════
# Reviewer Agent
# ══════════════════════════════════════════════════════════════


class ReviewerAgent:
    """Agent that simulates an academic peer reviewer."""

    def __init__(
        self,
        role: ReviewerRole,
        expertise: list[str] | None = None,
    ):
        self.role = role
        self.expertise = expertise or []
        self.reviewer_id = f"{role.value}_{id(self)}"

    async def review(
        self,
        article_text: str,
        llm: Any,
        venue: str = "NeurIPS",
    ) -> PeerReview:
        """Generate a peer review for the given article.

        Args:
            article_text: The article to review
            llm: LLM router for generating review
            venue: Academic venue (NeurIPS, ICML, ICLR, etc.)

        Returns:
            PeerReview with scores, comments, and recommendation
        """
        start_time = time.time()

        prompt = self._build_reviewer_prompt(article_text, venue)

        try:
            response = await llm.call(
                complexity="standard",
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            review = self._parse_review(text)
            review.review_time = time.time() - start_time
            logger.info(
                f"[Review] {self.role.value}: decision={review.scores.decision}, "
                f"overall={review.scores.overall}/10"
            )
            return review

        except Exception as e:
            logger.error(f"[Review] {self.role.value} failed: {e}")
            # Return a default borderline review
            return self._default_review()

    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt."""
        role_prompts = {
            ReviewerRole.EXPERT_REVIEWER: (
                "You are an expert reviewer with deep domain knowledge. "
                "Focus on technical correctness, novelty of contributions, "
                "and significance of the work. Evaluate whether the work advances "
                "the field meaningfully."
            ),
            ReviewerRole.METHODOLOGY_REVIEWER: (
                "You are a methodology expert who evaluates experimental design. "
                "Focus on statistical rigor, proper baselines, ablation studies, "
                "and whether the claims are well-supported by experiments. "
                "Check for p-hacking, multiple comparisons, and generalization."
            ),
            ReviewerRole.CLARITY_REVIEWER: (
                "You are a writing and presentation expert. "
                "Focus on clarity of exposition, organization, figure/table quality, "
                "and accessibility. Evaluate whether a reader can follow the work."
            ),
            ReviewerRole.NOVELTY_REVIEWER: (
                "You are a literature expert focused on originality. "
                "Check whether the work is novel compared to related publications. "
                "Identify missing related work and assess the true contribution."
            ),
            ReviewerRole.REPRODUCIBILITY_REVIEWER: (
                "You are a reproducibility expert. "
                "Evaluate code availability, dataset accessibility, hyperparameter details, "
                "and whether results can be reproduced. Check for missing implementation details."
            ),
            ReviewerRole.META_REVIEWER: (
                "You are an area chair synthesizing reviews. "
                "Balance different reviewer perspectives, identify key concerns, "
                "and make a final recommendation aligned with venue standards."
            ),
        }
        return role_prompts.get(self.role, "You are an academic peer reviewer.")

    def _build_reviewer_prompt(self, article_text: str, venue: str) -> str:
        """Build role-specific review prompt."""
        base_prompt = f"""
## Task: Peer Review for {venue}

Review the following article. Provide a detailed, constructive review that would be
appropriate for {venue} standards.

## Article

{article_text[:3000]}...

## Review Format

Respond with a JSON object containing:
```json
{{
  "summary": "1-2 paragraph summary of the paper",
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1", "weakness 2", ...],
  "questions": ["question 1", "question 2", ...],
  "detailed_comments": [
    {{"section": "intro", "type": "weakness", "text": "...", "severity": "major"}}
  ],
  "requested_changes": ["change 1", "change 2"],
  "ethics_concerns": ["concern 1"] or [],
  "scores": {{
    "soundness": 1-5,
    "presentation": 1-5,
    "contribution": 1-5,
    "novelty": 1-5,
    "reproducibility": 1-5,
    "overall": 1-10,
    "confidence": 1-5
  }},
  "decision": "strong_accept|accept|weak_accept|borderline|weak_reject|reject|strong_reject"
}}
```

Be realistic and substantive in your review.
"""

        role_specifics = {
            ReviewerRole.EXPERT_REVIEWER: (
                "\nFocus on: technical soundness, whether claims are proven, "
                "correctness of theory, and significance of contributions."
            ),
            ReviewerRole.METHODOLOGY_REVIEWER: (
                "\nFocus on: experimental design, comparison to baselines, ablation studies, "
                "statistical significance, and reproducibility of results."
            ),
            ReviewerRole.CLARITY_REVIEWER: (
                "\nFocus on: writing quality, organization, clarity of presentation, "
                "quality of figures and tables, and overall readability."
            ),
            ReviewerRole.NOVELTY_REVIEWER: (
                "\nFocus on: originality compared to related work, identification of truly "
                "novel contributions, and adequacy of related work section."
            ),
            ReviewerRole.REPRODUCIBILITY_REVIEWER: (
                "\nFocus on: code availability, dataset details, hyperparameter specificity, "
                "whether someone could reproduce results, and transparency."
            ),
        }

        return base_prompt + role_specifics.get(self.role, "")

    def _parse_review(self, response: str) -> PeerReview:
        """Parse LLM response into a PeerReview object."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return self._default_review()

            scores_data = data.get("scores", {})
            scores = ReviewScore(
                soundness=self._clamp_score(scores_data.get("soundness", 3), 1, 5),
                presentation=self._clamp_score(scores_data.get("presentation", 3), 1, 5),
                contribution=self._clamp_score(scores_data.get("contribution", 3), 1, 5),
                novelty=self._clamp_score(scores_data.get("novelty", 3), 1, 5),
                reproducibility=self._clamp_score(scores_data.get("reproducibility", 3), 1, 5),
                overall=self._clamp_score(scores_data.get("overall", 6), 1, 10),
                confidence=self._clamp_score(scores_data.get("confidence", 3), 1, 5),
                decision=data.get("decision", "borderline"),
            )

            # Parse detailed comments
            detailed_comments = []
            for comment_data in data.get("detailed_comments", []):
                detailed_comments.append(
                    ReviewComment(
                        section=comment_data.get("section", "general"),
                        comment_type=comment_data.get("type", "suggestion"),
                        text=comment_data.get("text", ""),
                        severity=comment_data.get("severity", "minor"),
                    )
                )

            return PeerReview(
                reviewer_id=self.reviewer_id,
                reviewer_role=self.role,
                scores=scores,
                summary=data.get("summary", ""),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                questions=data.get("questions", []),
                detailed_comments=detailed_comments,
                requested_changes=data.get("requested_changes", []),
                ethics_concerns=data.get("ethics_concerns", []),
                review_time=0.0,
            )

        except Exception as e:
            logger.warning(f"Failed to parse review: {e}")
            return self._default_review()

    def _default_review(self) -> PeerReview:
        """Return a default borderline review."""
        return PeerReview(
            reviewer_id=self.reviewer_id,
            reviewer_role=self.role,
            scores=ReviewScore(3, 3, 3, 3, 3, 6, 3, "borderline"),
            summary="Review could not be generated.",
            strengths=[],
            weaknesses=[],
            questions=[],
            detailed_comments=[],
            requested_changes=[],
            ethics_concerns=[],
            review_time=0.0,
        )

    @staticmethod
    def _clamp_score(value: int | float, min_val: int, max_val: int) -> int:
        """Clamp a score to valid range."""
        return max(min_val, min(max_val, int(value)))


# ══════════════════════════════════════════════════════════════
# Author Agent
# ══════════════════════════════════════════════════════════════


class AuthorAgent:
    """Agent that writes rebuttals and revises articles."""

    async def write_rebuttal(
        self,
        article_text: str,
        reviews: list[PeerReview],
        llm: Any,
    ) -> AuthorRebuttal:
        """Write a point-by-point rebuttal to reviewer comments.

        Args:
            article_text: Original article
            reviews: List of peer reviews to respond to
            llm: LLM router for generating rebuttal

        Returns:
            AuthorRebuttal with responses to each concern
        """
        # Build rebuttal prompt
        reviews_summary = "\n".join(
            f"\n## Review {i + 1} ({review.reviewer_role.value})\n"
            f"Decision: {review.scores.decision}\n"
            f"Weaknesses:\n" + "\n".join(f"- {w}" for w in review.weaknesses[:3])
            for i, review in enumerate(reviews)
        )

        prompt = f"""
## Task: Write Author Rebuttal

You are an author responding to peer reviews. Write a constructive rebuttal that
addresses the main concerns while defending the contributions.

### Original Article (excerpt)
{article_text[:2000]}...

### Reviewer Feedback

{reviews_summary}

### Your Rebuttal

Write a JSON response:
```json
{{
  "point_by_point_responses": [
    {{"concern": "weakness from review", "response": "your response"}},
    ...
  ],
  "changes_made": ["change 1", "change 2"],
  "changes_declined": [
    {{"suggestion": "suggestion", "reason": "why you decline"}}
  ],
  "summary": "Overall response"
}}
```

Be professional, acknowledge valid criticisms, and explain where you disagree.
"""

        try:
            response = await llm.call(
                complexity="standard",
                system="You are an academic author responding to peer reviews constructively.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Parse rebuttal
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return AuthorRebuttal(
                    original_reviews=reviews,
                    point_by_point_responses=data.get("point_by_point_responses", []),
                    changes_made=data.get("changes_made", []),
                    changes_declined=data.get("changes_declined", []),
                )

        except Exception as e:
            logger.warning(f"Failed to generate rebuttal: {e}")

        # Default rebuttal
        return AuthorRebuttal(
            original_reviews=reviews,
            point_by_point_responses=[],
            changes_made=[],
            changes_declined=[],
        )

    async def revise_article(
        self,
        article_text: str,
        reviews: list[PeerReview],
        rebuttal: AuthorRebuttal,
        llm: Any,
    ) -> str:
        """Revise the article based on reviews and rebuttal.

        Args:
            article_text: Original article
            reviews: Peer reviews
            rebuttal: Author rebuttal
            llm: LLM router

        Returns:
            Revised article text
        """
        changes_to_make = rebuttal.changes_made[:5]  # Limit to top 5
        changes_text = "\n".join(f"- {c}" for c in changes_to_make)

        prompt = f"""
## Task: Revise Article Based on Feedback

Revise the article by implementing the following key changes:

{changes_text}

### Original Article (excerpt)
{article_text[:2500]}...

### Your Revision

Write a revised version of the article incorporating the changes. Focus on
the most impactful improvements.
"""

        try:
            response = await llm.call(
                complexity="standard",
                system="You are revising an academic paper based on reviewer feedback.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            return text

        except Exception as e:
            logger.warning(f"Failed to revise article: {e}")
            return article_text


# ══════════════════════════════════════════════════════════════
# Meta Reviewer Agent
# ══════════════════════════════════════════════════════════════


class MetaReviewerAgent:
    """Area chair agent that synthesizes reviews and makes final recommendation."""

    async def synthesize(
        self,
        reviews: list[PeerReview],
        rebuttal: AuthorRebuttal | None,
        llm: Any,
    ) -> MetaReview:
        """Synthesize individual reviews into a meta-review with recommendation.

        Args:
            reviews: List of peer reviews
            rebuttal: Optional author rebuttal
            llm: LLM router

        Returns:
            MetaReview with consensus scores and recommendation
        """
        consensus = self._compute_consensus(reviews)

        # Build synthesis prompt
        reviews_summary = "\n".join(
            f"**{review.reviewer_role.value}**: {review.scores.decision} "
            f"(overall={review.scores.overall}/10)\n"
            f"  - {' | '.join(review.strengths[:2])}\n"
            f"  - {' | '.join(review.weaknesses[:2])}"
            for review in reviews
        )

        rebuttal_text = ""
        if rebuttal:
            rebuttal_text = (
                f"\n\n## Author Rebuttal\n"
                f"Changes made: {len(rebuttal.changes_made)}\n"
                f"Changes declined: {len(rebuttal.changes_declined)}"
            )

        prompt = f"""
## Task: Area Chair Meta-Review

Synthesize the individual reviews and make a final recommendation.

## Individual Reviews
{reviews_summary}

{rebuttal_text}

## Your Meta-Review

Provide a JSON response:
```json
{{
  "consensus_scores": {{
    "soundness": 1-5,
    "presentation": 1-5,
    "contribution": 1-5,
    "novelty": 1-5,
    "reproducibility": 1-5,
    "overall": 1-10,
    "confidence": 1-5
  }},
  "key_strengths": ["strength 1", "strength 2"],
  "key_concerns": ["concern 1", "concern 2"],
  "recommendation": "accept|reject|revise",
  "confidence": 0.0-1.0,
  "summary": "Brief justification for recommendation"
}}
```

Be balanced and consider minority opinions. Make a clear recommendation.
"""

        try:
            response = await llm.call(
                complexity="standard",
                system="You are an area chair synthesizing peer reviews.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Parse meta-review
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                scores_data = data.get("consensus_scores", {})
                consensus = ReviewScore(
                    soundness=ReviewerAgent._clamp_score(scores_data.get("soundness", 3), 1, 5),
                    presentation=ReviewerAgent._clamp_score(scores_data.get("presentation", 3), 1, 5),
                    contribution=ReviewerAgent._clamp_score(scores_data.get("contribution", 3), 1, 5),
                    novelty=ReviewerAgent._clamp_score(scores_data.get("novelty", 3), 1, 5),
                    reproducibility=ReviewerAgent._clamp_score(scores_data.get("reproducibility", 3), 1, 5),
                    overall=ReviewerAgent._clamp_score(scores_data.get("overall", 6), 1, 10),
                    confidence=ReviewerAgent._clamp_score(scores_data.get("confidence", 3), 1, 5),
                    decision=data.get("recommendation", "borderline"),
                )

                return MetaReview(
                    individual_reviews=reviews,
                    rebuttal=rebuttal,
                    consensus_scores=consensus,
                    key_strengths=data.get("key_strengths", []),
                    key_concerns=data.get("key_concerns", []),
                    recommendation=data.get("recommendation", "borderline"),
                    confidence=float(data.get("confidence", 0.5)),
                    summary=data.get("summary", ""),
                )

        except Exception as e:
            logger.warning(f"Failed to synthesize meta-review: {e}")

        # Fallback: use consensus only
        return MetaReview(
            individual_reviews=reviews,
            rebuttal=rebuttal,
            consensus_scores=consensus,
            key_strengths=[],
            key_concerns=[],
            recommendation="borderline",
            confidence=0.5,
            summary="Meta-review synthesis failed.",
        )

    @staticmethod
    def _compute_consensus(reviews: list[PeerReview]) -> ReviewScore:
        """Compute weighted average consensus scores.

        Weight by reviewer confidence.
        """
        if not reviews:
            return ReviewScore(3, 3, 3, 3, 3, 6, 3, "borderline")

        total_confidence = sum(r.scores.confidence for r in reviews)
        if total_confidence == 0:
            total_confidence = len(reviews)

        soundness = sum(
            r.scores.soundness * r.scores.confidence for r in reviews
        ) / total_confidence
        presentation = sum(
            r.scores.presentation * r.scores.confidence for r in reviews
        ) / total_confidence
        contribution = sum(
            r.scores.contribution * r.scores.confidence for r in reviews
        ) / total_confidence
        novelty = sum(
            r.scores.novelty * r.scores.confidence for r in reviews
        ) / total_confidence
        reproducibility = sum(
            r.scores.reproducibility * r.scores.confidence for r in reviews
        ) / total_confidence
        overall = sum(
            r.scores.overall * r.scores.confidence for r in reviews
        ) / total_confidence
        confidence = sum(r.scores.confidence for r in reviews) / len(reviews)

        # Determine consensus decision
        avg_overall = overall
        if avg_overall >= 8:
            decision = "strong_accept"
        elif avg_overall >= 6.5:
            decision = "accept"
        elif avg_overall >= 5.5:
            decision = "weak_accept"
        elif avg_overall >= 4.5:
            decision = "borderline"
        elif avg_overall >= 3.5:
            decision = "weak_reject"
        elif avg_overall >= 2:
            decision = "reject"
        else:
            decision = "strong_reject"

        return ReviewScore(
            soundness=int(round(soundness)),
            presentation=int(round(presentation)),
            contribution=int(round(contribution)),
            novelty=int(round(novelty)),
            reproducibility=int(round(reproducibility)),
            overall=int(round(overall)),
            confidence=int(round(confidence)),
            decision=decision,
        )

    @staticmethod
    def _identify_disagreements(reviews: list[PeerReview]) -> list[str]:
        """Identify significant disagreements between reviewers."""
        if len(reviews) < 2:
            return []

        disagreements = []
        decisions = [r.scores.decision for r in reviews]
        decision_counts = {}
        for d in decisions:
            decision_counts[d] = decision_counts.get(d, 0) + 1

        if len(decision_counts) > 1:
            most_common = max(decision_counts, key=decision_counts.get)
            other_counts = sum(c for d, c in decision_counts.items() if d != most_common)
            if other_counts > 0:
                disagreements.append(
                    f"Reviewers disagreed on decision: {most_common} vs other views"
                )

        # Check for large score variance
        overalls = [r.scores.overall for r in reviews]
        if max(overalls) - min(overalls) >= 4:
            disagreements.append(
                f"Large variance in overall scores: {min(overalls)}-{max(overalls)}"
            )

        return disagreements


# ══════════════════════════════════════════════════════════════
# Peer Review Pipeline
# ══════════════════════════════════════════════════════════════


class PeerReviewPipeline:
    """Orchestrator for the full peer review simulation process."""

    def __init__(
        self,
        num_reviewers: int = 3,
        venue: str = "NeurIPS",
        include_rebuttal: bool = True,
    ):
        """Initialize the peer review pipeline.

        Args:
            num_reviewers: Number of reviewers to simulate (default 3)
            venue: Academic venue (NeurIPS, ICML, ICLR, etc.)
            include_rebuttal: Whether to include author rebuttal phase
        """
        self.num_reviewers = num_reviewers
        self.venue = venue
        self.include_rebuttal = include_rebuttal

    async def review_article(
        self,
        article_text: str,
        llm: Any,
        title: str = "",
    ) -> dict[str, Any]:
        """Conduct a complete peer review of an article.

        Args:
            article_text: The article to review
            llm: LLM router
            title: Optional article title for context

        Returns:
            Dictionary with reviews, rebuttal, meta_review, and report
        """
        logger.info(
            f"[PeerReview] Starting review for {self.venue}: {num_reviewers} reviewers"
        )

        # Step 1: Generate individual reviews
        reviewer_roles = self._select_reviewer_roles(self.num_reviewers)
        reviewers = [ReviewerAgent(role) for role in reviewer_roles]

        reviews = []
        for reviewer in reviewers:
            review = await reviewer.review(article_text, llm, self.venue)
            reviews.append(review)
            logger.info(
                f"[PeerReview] {reviewer.role.value}: "
                f"{review.scores.decision} ({review.scores.overall}/10)"
            )

        # Step 2: Optional author rebuttal
        rebuttal = None
        if self.include_rebuttal:
            author = AuthorAgent()
            rebuttal = await author.write_rebuttal(article_text, reviews, llm)
            logger.info(f"[PeerReview] Author rebuttal: {len(rebuttal.changes_made)} changes proposed")

        # Step 3: Meta-review synthesis
        meta_reviewer = MetaReviewerAgent()
        meta_review = await meta_reviewer.synthesize(reviews, rebuttal, llm)
        logger.info(
            f"[PeerReview] Meta-review: {meta_review.recommendation} "
            f"(confidence={meta_review.confidence:.2f})"
        )

        # Step 4: Generate report
        report = self._generate_report(reviews, rebuttal, meta_review)

        return {
            "title": title,
            "venue": self.venue,
            "reviews": [r.to_dict() for r in reviews],
            "rebuttal": rebuttal.to_dict() if rebuttal else None,
            "meta_review": meta_review.to_dict(),
            "report": report,
        }

    async def iterative_review(
        self,
        article_text: str,
        llm: Any,
        max_rounds: int = 2,
    ) -> dict[str, Any]:
        """Conduct iterative review: review → revise → re-review.

        Args:
            article_text: The article to review
            llm: LLM router
            max_rounds: Maximum revision rounds

        Returns:
            Dictionary with all rounds of reviews, revisions, and meta-review
        """
        current_text = article_text
        rounds = []

        for round_num in range(max_rounds):
            logger.info(f"[PeerReview] Starting iteration round {round_num + 1}/{max_rounds}")

            # Step 1: Review current version
            round_reviews = []
            reviewer_roles = self._select_reviewer_roles(self.num_reviewers)
            reviewers = [ReviewerAgent(role) for role in reviewer_roles]

            for reviewer in reviewers:
                review = await reviewer.review(current_text, llm, self.venue)
                round_reviews.append(review)

            # Step 2: Author rebuttal and revision
            author = AuthorAgent()
            rebuttal = await author.write_rebuttal(current_text, round_reviews, llm)
            revised_text = await author.revise_article(
                current_text, round_reviews, rebuttal, llm
            )

            rounds.append({
                "round": round_num + 1,
                "reviews": [r.to_dict() for r in round_reviews],
                "rebuttal": rebuttal.to_dict(),
                "revised": revised_text[:500],  # Store excerpt
            })

            current_text = revised_text

        # Final meta-review
        final_review_obj = []
        reviewer_roles = self._select_reviewer_roles(self.num_reviewers)
        reviewers = [ReviewerAgent(role) for role in reviewer_roles]

        for reviewer in reviewers:
            review = await reviewer.review(current_text, llm, self.venue)
            final_review_obj.append(review)

        meta_reviewer = MetaReviewerAgent()
        meta_review = await meta_reviewer.synthesize(final_review_obj, None, llm)

        return {
            "rounds": rounds,
            "final_reviews": [r.to_dict() for r in final_review_obj],
            "final_meta_review": meta_review.to_dict(),
        }

    @staticmethod
    def _select_reviewer_roles(num: int) -> list[ReviewerRole]:
        """Select diverse reviewer roles for a review panel."""
        all_roles = [
            ReviewerRole.EXPERT_REVIEWER,
            ReviewerRole.METHODOLOGY_REVIEWER,
            ReviewerRole.CLARITY_REVIEWER,
            ReviewerRole.NOVELTY_REVIEWER,
            ReviewerRole.REPRODUCIBILITY_REVIEWER,
        ]

        if num >= len(all_roles):
            return all_roles

        # Return a diverse subset
        step = max(1, len(all_roles) // num)
        return all_roles[::step][:num]

    @staticmethod
    def _generate_report(
        reviews: list[PeerReview],
        rebuttal: AuthorRebuttal | None,
        meta_review: MetaReview,
    ) -> str:
        """Generate a markdown summary report of the entire review process."""
        report = "# Peer Review Report\n\n"

        # Summary
        report += "## Summary\n\n"
        report += f"- **Recommendation**: {meta_review.recommendation.upper()}\n"
        report += f"- **Confidence**: {meta_review.confidence:.1%}\n"
        report += f"- **Reviewers**: {len(reviews)}\n\n"

        # Consensus Scores
        report += "## Consensus Scores\n\n"
        scores = meta_review.consensus_scores
        report += f"| Criterion | Score |\n"
        report += f"|-----------|-------|\n"
        report += f"| Soundness | {scores.soundness}/5 |\n"
        report += f"| Presentation | {scores.presentation}/5 |\n"
        report += f"| Contribution | {scores.contribution}/5 |\n"
        report += f"| Novelty | {scores.novelty}/5 |\n"
        report += f"| Reproducibility | {scores.reproducibility}/5 |\n"
        report += f"| **Overall** | **{scores.overall}/10** |\n\n"

        # Key Strengths
        if meta_review.key_strengths:
            report += "## Key Strengths\n\n"
            for strength in meta_review.key_strengths:
                report += f"- {strength}\n"
            report += "\n"

        # Key Concerns
        if meta_review.key_concerns:
            report += "## Key Concerns\n\n"
            for concern in meta_review.key_concerns:
                report += f"- {concern}\n"
            report += "\n"

        # Individual Reviews
        report += "## Individual Reviews\n\n"
        for i, review in enumerate(reviews, 1):
            report += f"### Review {i}: {review.reviewer_role.value}\n"
            report += f"**Decision**: {review.scores.decision}\n"
            report += f"**Overall**: {review.scores.overall}/10 | "
            report += f"**Confidence**: {review.scores.confidence}/5\n\n"

            if review.strengths:
                report += "**Strengths**:\n"
                for s in review.strengths[:3]:
                    report += f"- {s}\n"
                report += "\n"

            if review.weaknesses:
                report += "**Weaknesses**:\n"
                for w in review.weaknesses[:3]:
                    report += f"- {w}\n"
                report += "\n"

        # Author Rebuttal
        if rebuttal:
            report += "## Author Rebuttal\n\n"
            report += f"**Changes Made**: {len(rebuttal.changes_made)}\n"
            report += f"**Changes Declined**: {len(rebuttal.changes_declined)}\n\n"

            if rebuttal.changes_made:
                report += "**Implemented Changes**:\n"
                for change in rebuttal.changes_made[:5]:
                    report += f"- {change}\n"

        # Meta-Review Summary
        report += "\n## Area Chair Summary\n\n"
        report += meta_review.summary + "\n"

        return report
