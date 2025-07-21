"""
State management for the metanalyst agent system.

This module defines the state structure that maintains all information
throughout the meta-analysis process, including messages, URLs, insights,
and reports.
"""

from typing import Dict, List, Optional, Any, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
import json
from operator import add


class HumanMessage(BaseModel):
    """Structure for human messages in the system."""
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str
    message_type: str = "request"  # request, feedback, approval, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Structure for agent messages in the system."""
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_name: str
    content: str
    message_type: str = "response"  # response, status, error, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)


class URLToProcess(BaseModel):
    """Structure for URLs that need to be processed."""
    url: str
    source: str  # search_query, manual, reference, etc.
    priority: int = 1  # 1-5, higher is more priority
    metadata: Dict[str, Any] = Field(default_factory=dict)
    added_at: datetime = Field(default_factory=datetime.now)


class ProcessedURL(BaseModel):
    """Structure for URLs that have been processed."""
    url: str
    processed_at: datetime = Field(default_factory=datetime.now)
    status: str  # success, error, partial
    content_summary: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class SearchQuery(BaseModel):
    """Structure for search queries made during the process."""
    query: str
    search_engine: str = "tavily"
    timestamp: datetime = Field(default_factory=datetime.now)
    results_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObjectiveMetric(BaseModel):
    """Structure for objective metrics extracted from studies."""
    metric_name: str
    value: float
    confidence_interval: Optional[List[float]] = None
    p_value: Optional[float] = None
    study_id: str
    source_url: str
    extraction_method: str = "automated"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Insight(BaseModel):
    """Structure for insights generated during analysis."""
    insight_type: str  # statistical, clinical, methodological, etc.
    content: str
    confidence: float = Field(ge=0.0, le=1.0)  # 0-1 confidence score
    supporting_evidence: List[str] = Field(default_factory=list)
    generated_by: str  # agent name
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Feedback(BaseModel):
    """Structure for feedback from humans or agents."""
    feedback_type: str  # quality, direction, correction, etc.
    content: str
    source: str  # human, reviewer_agent, etc.
    target_component: str  # report, search, analysis, etc.
    timestamp: datetime = Field(default_factory=datetime.now)
    action_required: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReportDraft(BaseModel):
    """Structure for report drafts."""
    version: int
    content: str
    format_type: str = "html"  # html, markdown, latex
    sections: Dict[str, str] = Field(default_factory=dict)
    generated_by: str
    timestamp: datetime = Field(default_factory=datetime.now)
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetanalysisState(TypedDict):
    """
    Complete state for the metanalyst agent system.
    
    This state maintains all information throughout the meta-analysis process,
    from initial human request to final report generation.
    """
    
    # Core identification
    metanalysis_id: str
    
    # Communication history
    human_messages: Annotated[List[HumanMessage], add]
    agents_messages: Annotated[List[AgentMessage], add]
    
    # URL processing
    urls_to_process: Annotated[List[URLToProcess], add]
    processed_urls: Annotated[List[ProcessedURL], add]
    
    # Search and retrieval
    search_queries: Annotated[List[SearchQuery], add]
    
    # Analysis data
    objective_metrics: Annotated[List[ObjectiveMetric], add]
    insights: Annotated[List[Insight], add]
    
    # Feedback and iteration
    feedbacks: Annotated[List[Feedback], add]
    
    # Report generation
    report_drafts: Annotated[List[ReportDraft], add]
    final_report_not_edited: Optional[str]
    final_report_edited: Optional[str]
    
    # Process control
    current_step: str  # current processing step
    next_agent: Optional[str]  # next agent to be invoked
    process_status: str  # active, paused, completed, error
    
    # PICO framework (Population, Intervention, Comparison, Outcome)
    pico_population: Optional[str]
    pico_intervention: Optional[str] 
    pico_comparison: Optional[str]
    pico_outcome: Optional[str]
    
    # Quality control
    quality_checks: Dict[str, Any]  # quality assessment results
    validation_status: str  # pending, validated, needs_revision
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Metadata
    metadata: Dict[str, Any]


def create_initial_state(metanalysis_id: str, initial_request: str) -> MetanalysisState:
    """
    Create an initial state for a new meta-analysis process.
    
    Args:
        metanalysis_id: Unique identifier for this meta-analysis
        initial_request: The initial human request that started the process
    
    Returns:
        MetanalysisState: Initial state with basic setup
    """
    now = datetime.now()
    
    initial_human_message = HumanMessage(
        content=initial_request,
        message_type="initial_request",
        timestamp=now
    )
    
    return MetanalysisState(
        metanalysis_id=metanalysis_id,
        human_messages=[initial_human_message],
        agents_messages=[],
        urls_to_process=[],
        processed_urls=[],
        search_queries=[],
        objective_metrics=[],
        insights=[],
        feedbacks=[],
        report_drafts=[],
        final_report_not_edited=None,
        final_report_edited=None,
        current_step="initialization",
        next_agent="orchestrator",
        process_status="active",
        pico_population=None,
        pico_intervention=None,
        pico_comparison=None,
        pico_outcome=None,
        quality_checks={},
        validation_status="pending",
        created_at=now,
        updated_at=now,
        metadata={}
    )


def update_state_timestamp(state: MetanalysisState) -> MetanalysisState:
    """Update the state's updated_at timestamp."""
    state["updated_at"] = datetime.now()
    return state


def get_latest_human_message(state: MetanalysisState) -> Optional[HumanMessage]:
    """Get the most recent human message from the state."""
    if state["human_messages"]:
        return state["human_messages"][-1]
    return None


def get_latest_agent_message(state: MetanalysisState) -> Optional[AgentMessage]:
    """Get the most recent agent message from the state."""
    if state["agents_messages"]:
        return state["agents_messages"][-1]
    return None


def add_agent_message(
    state: MetanalysisState, 
    agent_name: str, 
    content: str, 
    message_type: str = "response",
    metadata: Optional[Dict[str, Any]] = None
) -> MetanalysisState:
    """
    Add a new agent message to the state.
    
    Args:
        state: Current state
        agent_name: Name of the agent sending the message
        content: Message content
        message_type: Type of message (response, status, error, etc.)
        metadata: Additional metadata for the message
    
    Returns:
        Updated state
    """
    message = AgentMessage(
        agent_name=agent_name,
        content=content,
        message_type=message_type,
        metadata=metadata or {}
    )
    
    state["agents_messages"].append(message)
    return update_state_timestamp(state)


def add_human_message(
    state: MetanalysisState,
    content: str,
    message_type: str = "feedback",
    metadata: Optional[Dict[str, Any]] = None
) -> MetanalysisState:
    """
    Add a new human message to the state.
    
    Args:
        state: Current state
        content: Message content
        message_type: Type of message (feedback, request, approval, etc.)
        metadata: Additional metadata for the message
    
    Returns:
        Updated state
    """
    message = HumanMessage(
        content=content,
        message_type=message_type,
        metadata=metadata or {}
    )
    
    state["human_messages"].append(message)
    return update_state_timestamp(state)


def set_next_step(
    state: MetanalysisState, 
    step: str, 
    next_agent: Optional[str] = None
) -> MetanalysisState:
    """
    Update the current step and next agent in the process.
    
    Args:
        state: Current state
        step: Next step name
        next_agent: Next agent to be invoked (optional)
    
    Returns:
        Updated state
    """
    state["current_step"] = step
    if next_agent:
        state["next_agent"] = next_agent
    
    return update_state_timestamp(state)


def get_state_summary(state: MetanalysisState) -> Dict[str, Any]:
    """
    Get a summary of the current state for logging/monitoring.
    
    Args:
        state: Current state
    
    Returns:
        Dictionary with state summary
    """
    return {
        "metanalysis_id": state["metanalysis_id"],
        "current_step": state["current_step"],
        "next_agent": state["next_agent"],
        "process_status": state["process_status"],
        "human_messages_count": len(state["human_messages"]),
        "agent_messages_count": len(state["agents_messages"]),
        "urls_to_process": len(state["urls_to_process"]),
        "processed_urls": len(state["processed_urls"]),
        "search_queries_count": len(state["search_queries"]),
        "insights_count": len(state["insights"]),
        "report_drafts_count": len(state["report_drafts"]),
        "has_final_report": state["final_report_not_edited"] is not None,
        "validation_status": state["validation_status"],
        "created_at": state["created_at"].isoformat(),
        "updated_at": state["updated_at"].isoformat()
    }