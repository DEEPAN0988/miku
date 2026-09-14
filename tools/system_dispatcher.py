"""
tools/system_dispatcher.py — System Dispatcher & Media Routing Facade

Re-exports core dispatcher symbols and context-aware media router.
"""

from tools.dispatcher import (
    CONFIDENCE_MARGIN,
    CONFIDENCE_RATIO,
    RiskLevel,
    TOOL_DESCRIPTORS,
    TOOL_REGISTRY,
    ToolCall,
    ToolResult,
    detect_query_argument,
    dispatch_tool,
    extract_deterministic_slot,
    parse_tool_call,
    rank_with_structure,
    resolve_intent_anchor,
    resolve_intent_with_confidence,
    route_media_play,
)

__all__ = [
    "CONFIDENCE_MARGIN",
    "CONFIDENCE_RATIO",
    "RiskLevel",
    "TOOL_DESCRIPTORS",
    "TOOL_REGISTRY",
    "ToolCall",
    "ToolResult",
    "detect_query_argument",
    "dispatch_tool",
    "extract_deterministic_slot",
    "parse_tool_call",
    "rank_with_structure",
    "resolve_intent_anchor",
    "resolve_intent_with_confidence",
    "route_media_play",
]
