"""
Agent tooling module for the metanalyst system.

This module contains all agent-related functionality including handoff tools,
agent definitions, and inter-agent communication mechanisms.
"""

from .handoffs import (
    create_handoff_tool,
    HandoffTarget,
    researcher_handoff,
    processor_handoff,
    retriever_handoff,
    writer_handoff,
    reviewer_handoff,
    analyst_handoff,
    editor_handoff,
    human_approval_handoff
)

__all__ = [
    "create_handoff_tool",
    "HandoffTarget",
    "researcher_handoff",
    "processor_handoff", 
    "retriever_handoff",
    "writer_handoff",
    "reviewer_handoff",
    "analyst_handoff",
    "editor_handoff",
    "human_approval_handoff"
]