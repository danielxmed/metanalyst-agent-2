"""
Orchestrator Agent for the metanalyst system.

This module implements the central orchestrator agent that coordinates
all other agents in the system, makes decisions about next steps,
and manages the overall meta-analysis workflow.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode

from .state import (
    MetanalysisState, 
    create_initial_state,
    add_agent_message,
    set_next_step,
    get_latest_human_message,
    get_state_summary
)
from .config import MetanalystConfig, get_llm_instance, get_postgres_checkpointer
from ..agents.handoffs import (
    get_all_handoff_tools,
    create_workflow_handoff_tools,
    handoff_manager,
    HandoffTarget
)

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Central orchestrator agent that coordinates the entire meta-analysis process.
    
    The orchestrator analyzes the current state and decides which specialized agent
    should be invoked next, managing the flow of the meta-analysis from start to finish.
    """
    
    def __init__(self, config: MetanalystConfig):
        """
        Initialize the orchestrator agent.
        
        Args:
            config: Configuration object for the metanalyst system
        """
        self.config = config
        self.llm = get_llm_instance(config.llm)
        self.checkpointer = get_postgres_checkpointer(config.database)
        
        # Initialize tools
        self.handoff_tools = get_all_handoff_tools()
        self.workflow_tools = create_workflow_handoff_tools()
        self.all_tools = self.handoff_tools + list(self.workflow_tools.values())
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.all_tools + [
            self._define_pico_tool(),
            self._request_human_feedback_tool(),
            self._complete_metanalysis_tool(),
            self._pause_for_review_tool()
        ])
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _define_pico_tool(self):
        """Tool for defining PICO framework elements."""
        @tool
        def define_pico(
            population: str,
            intervention: str,
            comparison: str,
            outcome: str
        ) -> Dict[str, str]:
            """
            Define the PICO framework for the meta-analysis.
            
            Args:
                population: Target population for the study
                intervention: Intervention being studied
                comparison: Comparison/control group
                outcome: Primary outcome being measured
            
            Returns:
                Dictionary with PICO elements
            """
            return {
                "pico_population": population,
                "pico_intervention": intervention,
                "pico_comparison": comparison,
                "pico_outcome": outcome,
                "current_step": "pico_defined"
            }
        
        return define_pico
    
    def _request_human_feedback_tool(self):
        """Tool for requesting human feedback."""
        @tool
        def request_human_feedback(
            request_type: str,
            question: str,
            context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Request feedback from a human user.
            
            Args:
                request_type: Type of feedback needed (approval, guidance, correction)
                question: Specific question for the human
                context: Additional context information
            
            Returns:
                Human feedback response
            """
            feedback_request = {
                "request_type": request_type,
                "question": question,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Use interrupt to pause execution and wait for human input
            human_response = interrupt(feedback_request)
            
            return {
                "human_feedback": human_response,
                "current_step": "human_feedback_received"
            }
        
        return request_human_feedback
    
    def _complete_metanalysis_tool(self):
        """Tool for marking the meta-analysis as complete."""
        @tool
        def complete_metanalysis(
            completion_reason: str,
            final_status: str = "completed"
        ) -> Dict[str, str]:
            """
            Mark the meta-analysis as complete.
            
            Args:
                completion_reason: Reason for completion
                final_status: Final status (completed, cancelled, error)
            
            Returns:
                Completion status
            """
            return {
                "process_status": final_status,
                "current_step": "completed",
                "completion_reason": completion_reason,
                "completed_at": datetime.now().isoformat()
            }
        
        return complete_metanalysis
    
    def _pause_for_review_tool(self):
        """Tool for pausing the process for review."""
        @tool
        def pause_for_review(
            pause_reason: str,
            review_instructions: str
        ) -> Dict[str, str]:
            """
            Pause the process for human review.
            
            Args:
                pause_reason: Reason for pausing
                review_instructions: Instructions for the reviewer
            
            Returns:
                Pause status
            """
            return {
                "process_status": "paused",
                "current_step": "paused_for_review",
                "pause_reason": pause_reason,
                "review_instructions": review_instructions,
                "paused_at": datetime.now().isoformat()
            }
        
        return pause_for_review
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the orchestrator."""
        return """You are the Orchestrator Agent for an automated meta-analysis system. Your role is to:

1. ANALYZE the current state of the meta-analysis process
2. DECIDE which specialized agent should handle the next step
3. COORDINATE the workflow from initial request to final report

AVAILABLE AGENTS:
- RESEARCHER: Search scientific literature, generate queries, find publications
- PROCESSOR: Extract content from URLs, process documents, create embeddings  
- RETRIEVER: Search processed documents using vector similarity
- WRITER: Generate structured reports and analyze content
- REVIEWER: Evaluate quality, provide feedback, validate compliance
- ANALYST: Perform statistical analysis, create visualizations
- EDITOR: Integrate reports with analyses, finalize documents
- HUMAN_APPROVAL: Request human review and feedback

WORKFLOW STAGES:
1. INITIALIZATION: Define PICO framework, understand requirements
2. LITERATURE_SEARCH: Find and collect relevant publications
3. PROCESSING: Extract and structure content from sources
4. ANALYSIS: Perform statistical and content analysis
5. REPORTING: Generate and refine reports
6. REVIEW: Quality control and validation
7. FINALIZATION: Complete final report

DECISION PRINCIPLES:
- Always consider the current state and what has been completed
- Hand off tasks to the most appropriate specialized agent
- Request human feedback for critical decisions
- Ensure quality at each step before proceeding
- Track progress and maintain workflow continuity

Use the available handoff tools to delegate tasks. Always provide clear task descriptions and context."""

    def _create_graph(self) -> StateGraph:
        """Create the orchestrator graph."""
        graph_builder = StateGraph(MetanalysisState)
        
        # Add orchestrator node
        graph_builder.add_node("orchestrator", self._orchestrator_node)
        
        # Add tool node for handling tool calls
        tool_node = ToolNode(self.all_tools + [
            self._define_pico_tool(),
            self._request_human_feedback_tool(),
            self._complete_metanalysis_tool(),
            self._pause_for_review_tool()
        ])
        graph_builder.add_node("tools", tool_node)
        
        # Add conditional edges
        graph_builder.add_conditional_edges(
            "orchestrator",
            self._should_use_tools,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Tools always return to orchestrator
        graph_builder.add_edge("tools", "orchestrator")
        
        # Set entry point
        graph_builder.add_edge(START, "orchestrator")
        
        # Compile with checkpointer
        return graph_builder.compile(checkpointer=self.checkpointer)
    
    def _orchestrator_node(self, state: MetanalysisState) -> Dict[str, Any]:
        """
        Main orchestrator node that analyzes state and decides next actions.
        
        Args:
            state: Current metanalysis state
        
        Returns:
            Updated state with orchestrator decisions
        """
        try:
            # Log current state
            state_summary = get_state_summary(state)
            logger.info(f"Orchestrator analyzing state: {state_summary}")
            
            # Get latest human message
            latest_human_msg = get_latest_human_message(state)
            
            # Create context message
            context_msg = self._create_context_message(state)
            
            # Prepare messages for LLM
            messages = [
                SystemMessage(content=self._create_system_prompt()),
                context_msg
            ]
            
            # Add latest human message if available
            if latest_human_msg:
                messages.append(HumanMessage(
                    content=f"Human message ({latest_human_msg.message_type}): {latest_human_msg.content}"
                ))
            
            # Get LLM response
            response = self.llm_with_tools.invoke(messages)
            
            # Add orchestrator message to state
            updated_state = add_agent_message(
                state,
                agent_name="orchestrator",
                content=response.content,
                message_type="decision",
                metadata={
                    "tool_calls": len(response.tool_calls) if hasattr(response, 'tool_calls') else 0,
                    "decision_timestamp": datetime.now().isoformat()
                }
            )
            
            return {"agents_messages": updated_state["agents_messages"]}
            
        except Exception as e:
            logger.error(f"Error in orchestrator node: {e}")
            error_state = add_agent_message(
                state,
                agent_name="orchestrator",
                content=f"Error occurred: {str(e)}",
                message_type="error"
            )
            return {"agents_messages": error_state["agents_messages"]}
    
    def _create_context_message(self, state: MetanalysisState) -> HumanMessage:
        """Create a context message with current state information."""
        
        # Build context information
        context_parts = [
            f"METANALYSIS ID: {state['metanalysis_id']}",
            f"CURRENT STEP: {state['current_step']}",
            f"PROCESS STATUS: {state['process_status']}",
            f"NEXT AGENT: {state.get('next_agent', 'None')}"
        ]
        
        # Add PICO information if available
        if any([state.get('pico_population'), state.get('pico_intervention'), 
                state.get('pico_comparison'), state.get('pico_outcome')]):
            context_parts.append("PICO FRAMEWORK:")
            context_parts.append(f"  Population: {state.get('pico_population', 'Not defined')}")
            context_parts.append(f"  Intervention: {state.get('pico_intervention', 'Not defined')}")
            context_parts.append(f"  Comparison: {state.get('pico_comparison', 'Not defined')}")
            context_parts.append(f"  Outcome: {state.get('pico_outcome', 'Not defined')}")
        
        # Add progress information
        context_parts.append(f"PROGRESS INDICATORS:")
        context_parts.append(f"  Human messages: {len(state['human_messages'])}")
        context_parts.append(f"  Agent messages: {len(state['agents_messages'])}")
        context_parts.append(f"  URLs to process: {len(state['urls_to_process'])}")
        context_parts.append(f"  Processed URLs: {len(state['processed_urls'])}")
        context_parts.append(f"  Search queries: {len(state['search_queries'])}")
        context_parts.append(f"  Insights: {len(state['insights'])}")
        context_parts.append(f"  Report drafts: {len(state['report_drafts'])}")
        context_parts.append(f"  Validation status: {state['validation_status']}")
        
        # Add recent messages context
        if state['agents_messages']:
            context_parts.append("RECENT AGENT ACTIVITY:")
            for msg in state['agents_messages'][-3:]:  # Last 3 messages
                context_parts.append(f"  {msg.agent_name}: {msg.message_type} - {msg.content[:100]}...")
        
        context = "\n".join(context_parts)
        
        return HumanMessage(content=f"CURRENT STATE CONTEXT:\n{context}\n\nBased on this state, what should be the next action?")
    
    def _should_use_tools(self, state: MetanalysisState) -> str:
        """Determine if tools should be used based on the last message."""
        if state["agents_messages"]:
            last_message = state["agents_messages"][-1]
            # Check if the last message was from orchestrator and has tool calls
            if (last_message.agent_name == "orchestrator" and 
                hasattr(last_message, 'tool_calls') and 
                last_message.tool_calls):
                return "tools"
        
        # Check if process is complete
        if state.get("process_status") == "completed":
            return "end"
        
        return "tools"  # Default to tools for now
    
    def start_metanalysis(
        self, 
        initial_request: str, 
        metanalysis_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Start a new meta-analysis process.
        
        Args:
            initial_request: The initial human request
            metanalysis_id: Optional custom ID for the meta-analysis
            thread_id: Optional custom thread ID
        
        Returns:
            Tuple of (metanalysis_id, thread_id)
        """
        # Generate IDs if not provided
        if metanalysis_id is None:
            metanalysis_id = str(uuid.uuid4())
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        # Create initial state
        initial_state = create_initial_state(metanalysis_id, initial_request)
        
        # Configure for execution
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Starting meta-analysis {metanalysis_id} with thread {thread_id}")
        
        # Start the graph execution
        try:
            result = self.graph.invoke(initial_state, config)
            logger.info(f"Meta-analysis {metanalysis_id} started successfully")
            return metanalysis_id, thread_id
        except Exception as e:
            logger.error(f"Failed to start meta-analysis {metanalysis_id}: {e}")
            raise
    
    def continue_metanalysis(self, thread_id: str, user_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Continue an existing meta-analysis process.
        
        Args:
            thread_id: Thread ID of the existing process
            user_input: Optional user input to add to the process
        
        Returns:
            Current state of the meta-analysis
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get current state
            current_state = self.graph.get_state(config)
            
            if user_input:
                # Add user input to state
                from .state import add_human_message
                updated_state = add_human_message(
                    current_state.values,
                    content=user_input,
                    message_type="feedback"
                )
                
                # Continue execution with updated state
                result = self.graph.invoke(updated_state, config)
            else:
                # Continue from current state
                result = self.graph.invoke(None, config)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to continue meta-analysis {thread_id}: {e}")
            raise
    
    def get_metanalysis_status(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current status of a meta-analysis.
        
        Args:
            thread_id: Thread ID of the process
        
        Returns:
            Status information
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.graph.get_state(config)
            return get_state_summary(state.values)
        except Exception as e:
            logger.error(f"Failed to get status for {thread_id}: {e}")
            raise
    
    def pause_metanalysis(self, thread_id: str, reason: str = "Manual pause") -> bool:
        """
        Pause a meta-analysis process.
        
        Args:
            thread_id: Thread ID of the process
            reason: Reason for pausing
        
        Returns:
            True if successfully paused
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get current state and update status
            current_state = self.graph.get_state(config)
            updated_state = set_next_step(
                current_state.values,
                "paused",
                None
            )
            updated_state["process_status"] = "paused"
            updated_state["metadata"]["pause_reason"] = reason
            updated_state["metadata"]["paused_at"] = datetime.now().isoformat()
            
            # Update the state
            self.graph.update_state(config, updated_state)
            
            logger.info(f"Meta-analysis {thread_id} paused: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause meta-analysis {thread_id}: {e}")
            return False