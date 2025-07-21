"""
Handoff tools for agent-to-agent communication in the metanalyst system.

This module implements the handoff mechanism that allows the orchestrator
to delegate tasks to specialized agents and manage the flow of control
between different agents in the system.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.types import Command
from datetime import datetime

from ..core.state import MetanalysisState, add_agent_message, set_next_step


class HandoffTarget(str, Enum):
    """Available agents for handoff operations."""
    RESEARCHER = "researcher"
    PROCESSOR = "processor"
    RETRIEVER = "retriever"
    WRITER = "writer"
    REVIEWER = "reviewer"
    ANALYST = "analyst"
    EDITOR = "editor"
    HUMAN_APPROVAL = "human_approval"
    ORCHESTRATOR = "orchestrator"


class HandoffPayload(BaseModel):
    """Payload structure for handoff operations."""
    target_agent: HandoffTarget
    task_description: str
    priority: int = Field(default=1, ge=1, le=5)
    context: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[str] = None
    timeout: Optional[int] = None  # seconds
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


def create_handoff_tool(
    target_agent: HandoffTarget,
    tool_name: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Create a handoff tool for delegating tasks to a specific agent.
    
    Args:
        target_agent: The target agent to hand off to
        tool_name: Custom tool name (optional)
        description: Custom tool description (optional)
    
    Returns:
        A configured handoff tool function
    """
    if tool_name is None:
        tool_name = f"handoff_to_{target_agent.value}"
    
    if description is None:
        description = f"Hand off the current task to the {target_agent.value} agent"
    
    @tool(name=tool_name, description=description)
    def handoff_tool(
        task_description: str,
        priority: int = 1,
        context: Optional[Dict[str, Any]] = None,
        expected_output: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Command:
        """
        Execute handoff to the target agent.
        
        Args:
            task_description: Description of the task to be performed
            priority: Task priority (1-5, higher is more urgent)
            context: Additional context information for the task
            expected_output: Description of expected output format
            timeout: Task timeout in seconds
        
        Returns:
            Command object for state transition
        """
        payload = HandoffPayload(
            target_agent=target_agent,
            task_description=task_description,
            priority=priority,
            context=context or {},
            expected_output=expected_output,
            timeout=timeout,
            metadata={
                "handoff_timestamp": datetime.now().isoformat(),
                "source_agent": "orchestrator",
                "handoff_reason": "task_delegation"
            }
        )
        
        # Create state update with handoff information
        state_update = {
            "current_step": f"handoff_to_{target_agent.value}",
            "next_agent": target_agent.value,
            "metadata": {
                "last_handoff": payload.dict(),
                "handoff_history": []  # This will be appended to existing history
            }
        }
        
        return Command(
            goto=target_agent.value,
            update=state_update,
            graph=Command.PARENT
        )
    
    return handoff_tool


# Pre-configured handoff tools for each agent
researcher_handoff = create_handoff_tool(
    HandoffTarget.RESEARCHER,
    description=(
        "Hand off to the researcher agent to search for scientific literature, "
        "generate search queries, and find relevant publications for the meta-analysis"
    )
)

processor_handoff = create_handoff_tool(
    HandoffTarget.PROCESSOR,
    description=(
        "Hand off to the processor agent to extract content from URLs, "
        "process publications, generate embeddings, and structure data"
    )
)

retriever_handoff = create_handoff_tool(
    HandoffTarget.RETRIEVER,
    description=(
        "Hand off to the retriever agent to search for relevant information "
        "using vector similarity and retrieve specific content from processed documents"
    )
)

writer_handoff = create_handoff_tool(
    HandoffTarget.WRITER,
    description=(
        "Hand off to the writer agent to generate structured reports, "
        "analyze retrieved content, and create initial report drafts"
    )
)

reviewer_handoff = create_handoff_tool(
    HandoffTarget.REVIEWER,
    description=(
        "Hand off to the reviewer agent to evaluate report quality, "
        "provide improvement feedback, and validate compliance with medical standards"
    )
)

analyst_handoff = create_handoff_tool(
    HandoffTarget.ANALYST,
    description=(
        "Hand off to the analyst agent to perform statistical analyses, "
        "generate forest plots, calculate meta-analysis metrics, and create visualizations"
    )
)

editor_handoff = create_handoff_tool(
    HandoffTarget.EDITOR,
    description=(
        "Hand off to the editor agent to integrate reports with analyses, "
        "generate final HTML documents, and ensure proper formatting"
    )
)

human_approval_handoff = create_handoff_tool(
    HandoffTarget.HUMAN_APPROVAL,
    description=(
        "Hand off to human approval process for review, feedback, "
        "or decision-making on critical aspects of the meta-analysis"
    )
)


class HandoffManager:
    """
    Manager class for handling handoff operations and tracking agent flow.
    """
    
    def __init__(self):
        self.handoff_history: List[HandoffPayload] = []
        self.active_handoffs: Dict[str, HandoffPayload] = {}
    
    def record_handoff(self, payload: HandoffPayload) -> None:
        """Record a handoff operation in the history."""
        self.handoff_history.append(payload)
        self.active_handoffs[payload.target_agent.value] = payload
    
    def complete_handoff(self, agent_name: str) -> Optional[HandoffPayload]:
        """Mark a handoff as completed and return the payload."""
        return self.active_handoffs.pop(agent_name, None)
    
    def get_handoff_chain(self) -> List[str]:
        """Get the chain of agent handoffs in order."""
        return [h.target_agent.value for h in self.handoff_history]
    
    def get_pending_handoffs(self) -> Dict[str, HandoffPayload]:
        """Get all pending handoff operations."""
        return self.active_handoffs.copy()
    
    def clear_history(self) -> None:
        """Clear the handoff history and active handoffs."""
        self.handoff_history.clear()
        self.active_handoffs.clear()


def create_conditional_handoff_tool(
    conditions: Dict[str, HandoffTarget],
    tool_name: str = "conditional_handoff",
    description: str = "Conditionally hand off to different agents based on the current state"
):
    """
    Create a conditional handoff tool that can route to different agents based on conditions.
    
    Args:
        conditions: Dictionary mapping condition names to target agents
        tool_name: Name of the tool
        description: Tool description
    
    Returns:
        Conditional handoff tool function
    """
    
    @tool(name=tool_name, description=description)
    def conditional_handoff_tool(
        condition: str,
        task_description: str,
        priority: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> Command:
        """
        Execute conditional handoff based on the specified condition.
        
        Args:
            condition: The condition key to determine target agent
            task_description: Description of the task to be performed
            priority: Task priority (1-5)
            context: Additional context information
        
        Returns:
            Command object for state transition
        """
        if condition not in conditions:
            raise ValueError(f"Unknown condition: {condition}. Available: {list(conditions.keys())}")
        
        target_agent = conditions[condition]
        
        payload = HandoffPayload(
            target_agent=target_agent,
            task_description=task_description,
            priority=priority,
            context=context or {},
            metadata={
                "handoff_timestamp": datetime.now().isoformat(),
                "source_agent": "orchestrator",
                "handoff_reason": "conditional_routing",
                "condition": condition
            }
        )
        
        state_update = {
            "current_step": f"conditional_handoff_to_{target_agent.value}",
            "next_agent": target_agent.value,
            "metadata": {
                "last_handoff": payload.dict(),
                "routing_condition": condition
            }
        }
        
        return Command(
            goto=target_agent.value,
            update=state_update,
            graph=Command.PARENT
        )
    
    return conditional_handoff_tool


def get_all_handoff_tools() -> List:
    """
    Get all available handoff tools for the orchestrator.
    
    Returns:
        List of all handoff tool functions
    """
    return [
        researcher_handoff,
        processor_handoff,
        retriever_handoff,
        writer_handoff,
        reviewer_handoff,
        analyst_handoff,
        editor_handoff,
        human_approval_handoff
    ]


def create_workflow_handoff_tools() -> Dict[str, Any]:
    """
    Create workflow-specific handoff tools for common meta-analysis patterns.
    
    Returns:
        Dictionary of workflow handoff tools
    """
    
    # Literature search workflow
    literature_search_handoff = create_conditional_handoff_tool(
        conditions={
            "search_needed": HandoffTarget.RESEARCHER,
            "process_urls": HandoffTarget.PROCESSOR,
            "search_complete": HandoffTarget.RETRIEVER
        },
        tool_name="literature_search_workflow",
        description="Handle literature search workflow routing"
    )
    
    # Analysis workflow
    analysis_workflow_handoff = create_conditional_handoff_tool(
        conditions={
            "statistical_analysis": HandoffTarget.ANALYST,
            "content_analysis": HandoffTarget.WRITER,
            "quality_review": HandoffTarget.REVIEWER
        },
        tool_name="analysis_workflow",
        description="Handle analysis workflow routing"
    )
    
    # Report generation workflow
    report_workflow_handoff = create_conditional_handoff_tool(
        conditions={
            "draft_report": HandoffTarget.WRITER,
            "review_report": HandoffTarget.REVIEWER,
            "finalize_report": HandoffTarget.EDITOR,
            "human_review": HandoffTarget.HUMAN_APPROVAL
        },
        tool_name="report_workflow",
        description="Handle report generation workflow routing"
    )
    
    return {
        "literature_search": literature_search_handoff,
        "analysis_workflow": analysis_workflow_handoff,
        "report_workflow": report_workflow_handoff
    }


# Global handoff manager instance
handoff_manager = HandoffManager()