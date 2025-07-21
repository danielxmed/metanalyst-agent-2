"""
Core state management for the Metanalyst Agent.

This module defines the state schema that will be shared across all agents
in the supervisor-workers architecture.
"""

from typing import Annotated, Dict, List, Optional, Any, Union
from typing_extensions import TypedDict
from datetime import datetime
from pydantic import BaseModel, Field
import json


class MetanalysisState(TypedDict):
    """
    Core state for the metanalysis process.
    
    This state is shared across all agents and contains all necessary information
    for the autonomous meta-analysis generation process.
    """
    
    # Core identification
    metanalysis_id: str
    
    # Communication and interaction
    human_messages: Annotated[List[Dict[str, Any]], lambda x, y: x + y]
    agents_messages: Annotated[List[Dict[str, Any]], lambda x, y: x + y]
    
    # Research and data collection
    urls_to_process: List[str]
    processed_urls: List[str]
    search_queries: List[str]
    
    # Analysis and insights
    objective_metrics: Dict[str, Any]
    insights: Dict[str, Any]
    feedbacks: Dict[str, Any]
    
    # Report generation
    report_drafts: Dict[str, Any]
    final_report_not_edited: Optional[str]
    final_report_edited: Optional[str]
    
    # Process control and metadata
    current_step: Optional[str]
    next_agent: Optional[str]
    error_messages: List[str]
    status: str  # "initialized", "in_progress", "completed", "error", "paused"
    
    # PICO framework for medical research
    pico: Dict[str, str]  # Population, Intervention, Comparison, Outcome
    
    # Research methodology
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    study_types: List[str]
    
    # Statistical analysis
    statistical_methods: List[str]
    effect_measures: List[str]
    heterogeneity_assessment: Dict[str, Any]
    
    # Quality assessment
    quality_scores: Dict[str, Any]
    risk_of_bias: Dict[str, Any]
    
    # Vector store and database references (NOT the actual content)
    vectorstore_refs: Dict[str, str]  # References to vector store collections
    database_refs: Dict[str, str]    # References to database entries
    
    # Timestamps and tracking
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    completed_at: Optional[datetime]


class AgentMessage(BaseModel):
    """Structure for agent-to-agent communication."""
    
    agent_name: str
    message_type: str  # "request", "response", "notification", "error"
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


class HumanMessage(BaseModel):
    """Structure for human-agent communication."""
    
    message_type: str  # "request", "feedback", "approval", "rejection"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    requires_response: bool = False


class PICOFramework(BaseModel):
    """PICO framework for structured medical research questions."""
    
    population: str = Field(description="Target population or participants")
    intervention: str = Field(description="Intervention being studied")
    comparison: str = Field(description="Comparison or control group")
    outcome: str = Field(description="Primary outcome measures")


class ResearchCriteria(BaseModel):
    """Research inclusion/exclusion criteria."""
    
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    study_types: List[str] = Field(default_factory=list)
    date_range: Optional[Dict[str, str]] = None
    language_restrictions: List[str] = Field(default_factory=list)


class StatisticalAnalysis(BaseModel):
    """Statistical analysis configuration and results."""
    
    methods: List[str] = Field(default_factory=list)
    effect_measures: List[str] = Field(default_factory=list)
    heterogeneity_tests: List[str] = Field(default_factory=list)
    subgroup_analyses: List[str] = Field(default_factory=list)
    sensitivity_analyses: List[str] = Field(default_factory=list)


def create_initial_state(
    metanalysis_id: str,
    human_request: str,
    pico: Optional[Dict[str, str]] = None
) -> MetanalysisState:
    """
    Create an initial state for a new meta-analysis.
    
    Args:
        metanalysis_id: Unique identifier for this meta-analysis
        human_request: Initial human request/question
        pico: Optional PICO framework structure
        
    Returns:
        Initial MetanalysisState
    """
    now = datetime.now()
    
    initial_human_message = HumanMessage(
        message_type="request",
        content=human_request,
        timestamp=now,
        requires_response=True
    )
    
    return MetanalysisState(
        metanalysis_id=metanalysis_id,
        human_messages=[initial_human_message.model_dump()],
        agents_messages=[],
        urls_to_process=[],
        processed_urls=[],
        search_queries=[],
        objective_metrics={},
        insights={},
        feedbacks={},
        report_drafts={},
        final_report_not_edited=None,
        final_report_edited=None,
        current_step="initialization",
        next_agent="orchestrator",
        error_messages=[],
        status="initialized",
        pico=pico or {},
        inclusion_criteria=[],
        exclusion_criteria=[],
        study_types=[],
        statistical_methods=[],
        effect_measures=[],
        heterogeneity_assessment={},
        quality_scores={},
        risk_of_bias={},
        vectorstore_refs={},
        database_refs={},
        created_at=now,
        updated_at=now,
        completed_at=None
    )


def update_state_timestamp(state: MetanalysisState) -> MetanalysisState:
    """Update the state timestamp."""
    state["updated_at"] = datetime.now()
    return state


def add_agent_message(
    state: MetanalysisState,
    agent_name: str,
    message_type: str,
    content: Dict[str, Any],
    correlation_id: Optional[str] = None
) -> MetanalysisState:
    """Add an agent message to the state."""
    message = AgentMessage(
        agent_name=agent_name,
        message_type=message_type,
        content=content,
        correlation_id=correlation_id
    )
    
    state["agents_messages"].append(message.model_dump())
    return update_state_timestamp(state)


def add_human_message(
    state: MetanalysisState,
    message_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    requires_response: bool = False
) -> MetanalysisState:
    """Add a human message to the state."""
    message = HumanMessage(
        message_type=message_type,
        content=content,
        metadata=metadata or {},
        requires_response=requires_response
    )
    
    state["human_messages"].append(message.model_dump())
    return update_state_timestamp(state)