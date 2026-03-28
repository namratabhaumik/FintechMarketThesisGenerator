"""Detect hallucinated tool calls from LangGraph message history."""

import logging
from typing import List, Dict, Any

from langchain_core.messages import BaseMessage

from core.agents.tool_registry import is_valid_tool

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detect hallucinated tool calls from structured LLM tool_calls.

    Analyzes actual tool_calls from AIMessages (not regex on text),
    validating each against the tool registry.
    """

    def analyze(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Analyze LangGraph messages for hallucinated tool calls.

        Args:
            messages: List of LangGraph messages from the refinement run.

        Returns:
            Dictionary with tool_calls, valid/invalid tools, and summary.
        """
        tool_calls = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc["name"])

        valid = [t for t in tool_calls if is_valid_tool(t)]
        invalid = [t for t in tool_calls if not is_valid_tool(t)]

        result = {
            "tool_calls": tool_calls,
            "valid_tools": valid,
            "invalid_tools": invalid,
            "has_hallucinations": len(invalid) > 0,
            "summary": self._build_summary(tool_calls, valid, invalid),
        }

        logger.info(f"Tool call analysis: {result}")
        return result

    @staticmethod
    def _build_summary(
        tool_calls: List[str], valid: List[str], invalid: List[str],
    ) -> str:
        if not tool_calls:
            return "No tool calls detected."

        lines = [f"Total tool calls: {len(tool_calls)}"]
        if valid:
            lines.append(f"Valid tools: {', '.join(valid)}")
        if invalid:
            lines.append(
                f"Hallucinated tools (do not exist): {', '.join(invalid)}"
            )
        return "\n".join(lines)
