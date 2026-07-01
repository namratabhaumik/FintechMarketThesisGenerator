"""Tool registry for agent hallucination prevention and detection."""

from typing import Dict

# Available tools in the refinement system
AVAILABLE_TOOLS: Dict[str, str] = {
    "refine_thesis": "Refine the investment thesis based on user feedback and source documents",
}


def get_available_tool_names() -> list[str]:
    """Get list of available tool names.

    Returns:
        List of tool identifiers that the agent can use.
    """
    return list(AVAILABLE_TOOLS.keys())


def is_valid_tool(tool_name: str) -> bool:
    """Check if a tool name is valid.

    Args:
        tool_name: Name of the tool to validate.

    Returns:
        True if tool exists, False otherwise.
    """
    return tool_name in AVAILABLE_TOOLS
