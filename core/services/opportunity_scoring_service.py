"""Opportunity scoring service - rule-based scoring for investment opportunities."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class OpportunityScoringService:
    """Rule-based service for scoring investment opportunities.

    Single Responsibility: Score opportunities based on thesis components.
    Uses deterministic rules to generate scores, confidence, and recommendations.
    """

    def __init__(self):
        """Initialize scoring service."""
        self._base_score = 2.5
        self._theme_weight = 0.25
        self._signal_weight = 0.75
        self._risk_penalty = 0.25
        self._max_themes = 3
        self._max_signals = 3
        self._max_risks = 3

    def score_opportunity(
        self,
        key_themes: list,
        risks: list,
        investment_signals: list,
        sources: list
    ) -> Dict:
        """Score an opportunity based on thesis components.

        Args:
            key_themes: List of key themes identified
            risks: List of risks identified
            investment_signals: List of investment signals identified
            sources: List of source URLs

        Returns:
            Dictionary with:
                - score: float (0-5)
                - confidence_level: float (0-1)
                - recommendation: str ("Pursue" / "Investigate" / "Skip")
                - key_risks: List[str] (top 2-3 risks)
        """
        logger.info("Scoring opportunity based on thesis components")

        # Calculate score components
        theme_score = min(len(key_themes), self._max_themes) * self._theme_weight
        signal_score = min(len(investment_signals), self._max_signals) * self._signal_weight
        risk_penalty = min(len(risks), self._max_risks) * self._risk_penalty

        # Calculate final score (0-5)
        opportunity_score = self._base_score + signal_score + theme_score - risk_penalty
        opportunity_score = max(0.0, min(5.0, opportunity_score))  # Clamp 0-5

        # Calculate confidence based on data quality (sources, signals, risks)
        # - Source coverage: 5+ sources = 100%
        # - Signal strength: 3+ signals = 100%
        # - Risk balance: many risks reduce confidence
        source_factor = min(len(sources) / 5.0, 1.0)  # 5 sources = full confidence
        signal_factor = min(len(investment_signals) / 3.0, 1.0)  # 3+ signals = full confidence
        risk_factor = max(0.5, 1.0 - (len(risks) / 4.0))  # Risks reduce confidence (min 50%)

        # Weighted average: sources 40%, signals 40%, risk tolerance 20%
        confidence_level = (source_factor * 0.4 + signal_factor * 0.4 + risk_factor * 0.2)
        confidence_level = round(confidence_level, 2)

        # Generate recommendation
        recommendation = self._get_recommendation(opportunity_score)

        # Get top 2-3 risks
        key_risks = risks[:min(3, len(risks))]

        logger.info(f"Opportunity scored: {opportunity_score:.1f}/5, confidence: {confidence_level:.2f}")

        return {
            "score": round(opportunity_score, 1),
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "key_risks": key_risks,
        }

    def _get_recommendation(self, score: float) -> str:
        """Determine recommendation based on score.

        Args:
            score: Opportunity score (0-5)

        Returns:
            Recommendation string
        """
        if score >= 3.75:
            return "Pursue"
        elif score >= 2.5:
            return "Investigate"
        else:
            return "Skip"
