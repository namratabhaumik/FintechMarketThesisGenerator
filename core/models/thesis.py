"""Thesis data models."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StructuredThesis:
    """Structured thesis output."""
    key_themes: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    investment_signals: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    raw_output: Optional[str] = None
