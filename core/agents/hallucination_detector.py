"""Detect and track hallucinated tool calls in LLM responses."""

import logging
import re
from typing import List, Dict, Any

from core.agents.tool_registry import AVAILABLE_TOOLS, is_valid_tool

logger = logging.getLogger(__name__)


class ToolCallExtractor:
    """Extracts tool call mentions from text using pattern matching.

    Single Responsibility: Extract tool names from LLM output using regex patterns.
    """

    @staticmethod
    def extract(text: str) -> List[str]:
        """Extract mentioned tool names from text.

        Args:
            text: LLM output text to analyze.

        Returns:
            List of tool names mentioned (including invalid ones).
        """
        tool_calls = []
        text_lower = text.lower()

        # Pattern 1: Direct tool name mentions (refine_thesis, structure_thesis, etc.)
        for tool_name in AVAILABLE_TOOLS.keys():
            if tool_name in text_lower:
                tool_calls.append(tool_name)

        # Pattern 2: Extract snake_case names after action verbs (called, invoke, use, etc.)
        # Matches: "called validate_sources", "invoked fetch_market_data", etc.
        action_pattern = r'(?:called|invoke|invoked|used|using|call|use)\s+(?:the\s+)?([a-z_]+?)(?:\s|,|\.|\))'
        for match in re.finditer(action_pattern, text_lower):
            tool_name = match.group(1).strip()
            if tool_name and len(tool_name) > 2 and tool_name not in tool_calls:
                tool_calls.append(tool_name)

        # Pattern 3: Extract names after "tool" keyword
        # Matches: "validate_sources tool", "fetch_market_data tool"
        tool_keyword_pattern = r'([a-z_]+)\s+(?:tool|function|operation)'
        for match in re.finditer(tool_keyword_pattern, text_lower):
            tool_name = match.group(1).strip()
            if tool_name and len(tool_name) > 2 and tool_name not in tool_calls:
                tool_calls.append(tool_name)

        # Pattern 4: Extract names after "X tool:"
        # Matches: "tool: validate_sources"
        colon_pattern = r'tool[s]?\s*:\s*([a-z_]+)'
        for match in re.finditer(colon_pattern, text_lower):
            tool_name = match.group(1).strip()
            if tool_name and len(tool_name) > 2 and tool_name not in tool_calls:
                tool_calls.append(tool_name)

        return list(set(tool_calls))  # Remove duplicates


class ToolValidator:
    """Validates tool calls against the available tools registry.

    Single Responsibility: Validate tool names against the tool registry.
    """

    @staticmethod
    def validate(tool_calls: List[str]) -> Dict[str, Any]:
        """Validate tool calls against available tools.

        Args:
            tool_calls: List of tool names to validate.

        Returns:
            Dictionary with:
                - valid_tools: Tools that exist in registry
                - invalid_tools: Tools that don't exist (hallucinations)
                - has_hallucinations: Boolean flag
        """
        valid = []
        invalid = []

        for tool in tool_calls:
            if is_valid_tool(tool):
                valid.append(tool)
            else:
                invalid.append(tool)

        return {
            "valid_tools": valid,
            "invalid_tools": invalid,
            "has_hallucinations": len(invalid) > 0,
        }


class HallucinationDetector:
    """Main detector for hallucinated tool calls in refinement responses.

    Single Responsibility: Orchestrate tool extraction and validation.
    Depends on: ToolCallExtractor, ToolValidator (Dependency Inversion).
    """

    def __init__(
        self,
        extractor: ToolCallExtractor = None,
        validator: ToolValidator = None,
    ):
        """Initialize detector with dependencies.

        Args:
            extractor: Tool call extractor (defaults to ToolCallExtractor).
            validator: Tool validator (defaults to ToolValidator).
        """
        self._extractor = extractor or ToolCallExtractor()
        self._validator = validator or ToolValidator()

    def analyze(self, llm_output: str) -> Dict[str, Any]:
        """Analyze LLM output for hallucinated tool calls.

        Args:
            llm_output: The text response from the LLM.

        Returns:
            Dictionary with:
                - tool_calls: List of all mentioned tools
                - valid_tools: Tools that exist
                - invalid_tools: Tools that don't exist (hallucinations)
                - has_hallucinations: Boolean flag
                - summary: Human-readable summary
        """
        # Step 1: Extract mentioned tools
        tool_calls = self._extractor.extract(llm_output)

        # Step 2: Validate against registry
        validation = self._validator.validate(tool_calls)

        # Step 3: Build summary
        summary = self._build_summary(tool_calls, validation)

        result = {
            "tool_calls": tool_calls,
            "valid_tools": validation["valid_tools"],
            "invalid_tools": validation["invalid_tools"],
            "has_hallucinations": validation["has_hallucinations"],
            "summary": summary,
        }

        # Log findings
        logger.info(f"Tool call analysis: {result}")

        return result

    @staticmethod
    def _build_summary(tool_calls: List[str], validation: Dict[str, Any]) -> str:
        """Build human-readable summary of tool calls.

        Args:
            tool_calls: All mentioned tools.
            validation: Validation results.

        Returns:
            Human-readable summary string.
        """
        if not tool_calls:
            return "No tool calls detected in output."

        lines = []
        lines.append(f"Total tool calls: {len(tool_calls)}")

        if validation["valid_tools"]:
            lines.append(f"✅ Valid tools: {', '.join(validation['valid_tools'])}")

        if validation["invalid_tools"]:
            lines.append(
                f"❌ Hallucinated tools (do not exist): {', '.join(validation['invalid_tools'])}"
            )

        return "\n".join(lines)
