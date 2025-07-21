"""
Specialized agents for the Metanalyst system.

This package contains all the specialized worker agents that are coordinated
by the central orchestrator in the hub-and-spoke architecture.
"""

from .base import BaseAgent
from .researcher import ResearcherAgent
from .processor import ProcessorAgent
from .retriever import RetrieverAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent
from .analyst import AnalystAgent
from .editor import EditorAgent
from .registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "ResearcherAgent", 
    "ProcessorAgent",
    "RetrieverAgent",
    "WriterAgent",
    "ReviewerAgent",
    "AnalystAgent",
    "EditorAgent",
    "AgentRegistry"
]