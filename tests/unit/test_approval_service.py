"""Unit tests for ApprovalService."""

import pytest
from core.models.thesis import StructuredThesis
from core.services.approval_service import ApprovalService


@pytest.fixture
def sample_thesis():
    """Create a sample thesis for testing."""
    return StructuredThesis(
        key_themes=["AI", "Digital Payments"],
        risks=["Regulatory", "Cybersecurity"],
        investment_signals=["Market Growth", "Disruption"],
        sources=["https://example.com/1", "https://example.com/2"],
        raw_output="Sample output",
        opportunity_score=3.8,
        confidence_level=0.77,
        recommendation="Pursue",
        key_risk_factors=["Regulatory"],
    )


@pytest.fixture
def approval_service():
    """Create an approval service instance."""
    return ApprovalService()


class TestApprovalRecording:
    """Test recording approval decisions."""

    def test_record_approval_decision(self, approval_service, sample_thesis):
        """Successfully record an approval decision."""
        record = approval_service.record_approval(sample_thesis, decision="approve")

        assert record["decision"] == "approve"
        assert record["opportunity_score"] == 3.8
        assert record["recommendation"] == "Pursue"
        assert record["confidence_level"] == 0.77
        assert "timestamp" in record
        assert record["key_themes"] == ["AI", "Digital Payments"]
        assert record["risks"] == ["Regulatory", "Cybersecurity"]

    def test_record_rejection_decision(self, approval_service, sample_thesis):
        """Successfully record a rejection decision."""
        record = approval_service.record_approval(sample_thesis, decision="reject")

        assert record["decision"] == "reject"
        assert record["opportunity_score"] == 3.8
        assert "timestamp" in record

    def test_record_approval_with_notes(self, approval_service, sample_thesis):
        """Record approval with human notes."""
        notes = "Strong signal but risky market timing"
        record = approval_service.record_approval(
            sample_thesis, decision="approve", notes=notes
        )

        assert record["decision"] == "approve"
        assert record["notes"] == notes

    def test_invalid_decision_raises_error(self, approval_service, sample_thesis):
        """Raises error for invalid decision value."""
        with pytest.raises(ValueError, match="Invalid decision"):
            approval_service.record_approval(sample_thesis, decision="maybe")


class TestApprovalSummary:
    """Test approval summary generation."""

    def test_approval_summary_generation(self, approval_service, sample_thesis):
        """Generate human-readable approval summary."""
        record = approval_service.record_approval(sample_thesis, decision="approve")
        summary = approval_service.get_approval_summary(record)

        assert "APPROVE" in summary
        assert record["timestamp"] in summary

    def test_rejection_summary_generation(self, approval_service, sample_thesis):
        """Generate human-readable rejection summary."""
        record = approval_service.record_approval(sample_thesis, decision="reject")
        summary = approval_service.get_approval_summary(record)

        assert "REJECT" in summary
        assert record["timestamp"] in summary


class TestApprovalMetadata:
    """Test metadata preservation in approval records."""

    def test_approval_preserves_thesis_data(self, approval_service, sample_thesis):
        """Approval record preserves thesis metadata."""
        record = approval_service.record_approval(sample_thesis, decision="approve")

        assert record["key_themes"] == sample_thesis.key_themes
        assert record["risks"] == sample_thesis.risks
        assert record["confidence_level"] == sample_thesis.confidence_level

    def test_multiple_approvals_independent(self, approval_service, sample_thesis):
        """Multiple approval records are independent."""
        thesis2 = StructuredThesis(
            key_themes=["Blockchain"],
            risks=["Regulatory"],
            investment_signals=["Web3"],
            sources=["https://example.com/3"],
            raw_output="Different output",
            opportunity_score=2.0,
            confidence_level=0.50,
            recommendation="Skip",
            key_risk_factors=[],
        )

        record1 = approval_service.record_approval(sample_thesis, decision="approve")
        record2 = approval_service.record_approval(thesis2, decision="reject")

        assert record1["opportunity_score"] == 3.8
        assert record2["opportunity_score"] == 2.0
        assert record1["recommendation"] != record2["recommendation"]
