"""Service for managing human approval of opportunity scoring recommendations."""

import logging
from datetime import datetime
from typing import Dict

from core.models.thesis import StructuredThesis

logger = logging.getLogger(__name__)


class ApprovalService:
    """Handles human approval of opportunity score recommendations.

    Records and logs human decisions for governance and feedback tracking.
    """

    def __init__(self):
        """Initialize approval service."""
        logger.info("ApprovalService initialized")

    def record_approval(
        self,
        thesis: StructuredThesis,
        decision: str = "approve",
        notes: str = "",
    ) -> Dict:
        """Record human approval of a thesis recommendation.

        Args:
            thesis: StructuredThesis object that was approved.
            decision: Must be "approve".
            notes: Optional human notes explaining the decision.

        Returns:
            Dict with approval record including timestamp and decision.
        """
        if decision != "approve":
            raise ValueError(f"Invalid decision: {decision}. Must be 'approve'.")

        approval_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "opportunity_score": thesis.opportunity_score,
            "recommendation": thesis.recommendation,
            "confidence_level": thesis.confidence_level,
            "key_themes": thesis.key_themes,
            "risks": thesis.risks,
            "notes": notes,
        }

        # Log the decision
        log_message = (
            f"Thesis {decision.upper()}: "
            f"Score={thesis.opportunity_score:.2f}, "
            f"Recommendation={thesis.recommendation}, "
            f"Confidence={thesis.confidence_level*100:.0f}%"
        )
        if notes:
            log_message += f", Notes={notes}"

        logger.info(f"✓ {log_message}")

        return approval_record

    def get_approval_summary(self, approval_record: Dict) -> str:
        """Get human-readable summary of approval decision.

        Args:
            approval_record: Dict returned from record_approval.

        Returns:
            Human-readable summary string.
        """
        decision = approval_record["decision"].upper()
        timestamp = approval_record["timestamp"]
        return f"{decision} at {timestamp}"
