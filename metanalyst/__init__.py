"""
Metanalyst Agent - A multi-system agent for autonomous metanalysis generation.

This package provides a supervisor-workers architecture for automated 
meta-analysis generation using Python and LangGraph.
"""

__version__ = "0.1.0"
__author__ = "Nobrega Medtech"

from .core.state import MetanalysisState
from .core.orchestrator import OrchestratorAgent

__all__ = ["MetanalysisState", "OrchestratorAgent"]