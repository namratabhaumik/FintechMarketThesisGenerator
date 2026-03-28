"""Track and log tool execution events during agent workflows."""

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ExecutionEvent:
    """Represents a single tool execution event."""

    def __init__(
        self,
        tool_name: str,
        status: str = "pending",
        timestamp: datetime = None,
        details: Dict[str, Any] = None,
    ):
        """Initialize execution event.

        Args:
            tool_name: Name of the tool that executed.
            status: Execution status ("pending", "executed", "failed").
            timestamp: When the event occurred.
            details: Optional metadata about execution (e.g., error message).
        """
        self.tool_name = tool_name
        self.status = status
        self.timestamp = timestamp or datetime.now()
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class ExecutionTracker:
    """Tracks and logs tool execution events.

    Single Responsibility: Record when tools execute and their outcomes.
    """

    def __init__(self):
        """Initialize execution tracker."""
        self._events: List[ExecutionEvent] = []

    def log_execution(
        self,
        tool_name: str,
        status: str = "executed",
        details: Dict[str, Any] = None,
    ) -> None:
        """Log a tool execution event.

        Args:
            tool_name: Name of the tool that executed.
            status: Execution status ("executed", "failed", "escalated").
            details: Optional metadata (error messages, etc.).
        """
        event = ExecutionEvent(
            tool_name=tool_name,
            status=status,
            details=details,
        )
        self._events.append(event)
        logger.info(f"Tool execution logged: {tool_name} - {status}")

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all execution events.

        Returns:
            List of execution events as dictionaries.
        """
        return [event.to_dict() for event in self._events]

    def get_executed_tools(self) -> List[str]:
        """Get list of tools that executed.

        Returns:
            List of tool names that successfully executed.
        """
        return [
            event.tool_name
            for event in self._events
            if event.status == "executed"
        ]

    def get_failed_tools(self) -> List[str]:
        """Get list of tools that failed.

        Returns:
            List of tool names that failed.
        """
        return [
            event.tool_name for event in self._events if event.status == "failed"
        ]

    def has_executed(self, tool_name: str) -> bool:
        """Check if a tool executed.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if tool executed, False otherwise.
        """
        return any(
            event.tool_name == tool_name and event.status == "executed"
            for event in self._events
        )

    def clear(self) -> None:
        """Clear all execution events."""
        self._events = []
        logger.info("Execution tracker cleared")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state.

        Returns:
            Dictionary with all execution events.
        """
        return {
            "total_events": len(self._events),
            "executed_count": len(self.get_executed_tools()),
            "failed_count": len(self.get_failed_tools()),
            "events": self.get_events(),
        }
