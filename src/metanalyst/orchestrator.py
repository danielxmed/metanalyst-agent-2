"""
Central Orchestrator Agent for the Metanalyst system.

This module implements the hub-and-spoke architecture where the orchestrator
maintains global state and decision logic, invoking specialized agents as tools.
"""

from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
import uuid
import structlog

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import MetanalysisState, add_agent_message, update_state_timestamp
from .config import MetanalystConfig, LLMProvider
from .llm_factory import create_llm
from .agents import AgentRegistry

logger = structlog.get_logger()


class OrchestratorDecision(BaseModel):
    """Structure for orchestrator decisions."""
    
    next_agent: str = Field(description="Name of the next agent to invoke")
    reasoning: str = Field(description="Reasoning for this decision")
    priority: int = Field(default=1, description="Priority level (1-5)")
    requires_human_approval: bool = Field(default=False)
    estimated_duration: Optional[int] = Field(default=None, description="Estimated duration in seconds")


class MetanalystOrchestrator:
    """
    Central orchestrator agent that manages the entire meta-analysis process.
    
    The orchestrator analyzes the current state and decides which specialized
    agent should be invoked next, maintaining complete control over the workflow.
    """
    
    def __init__(self, config: MetanalystConfig):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self.llm = create_llm(config.agents.orchestrator)
        self.agent_registry = AgentRegistry(config)
        self.logger = logger.bind(component="orchestrator")
        
        # Create the main graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the main orchestrator graph with all agents as nodes."""
        graph = StateGraph(MetanalysisState)
        
        # Add orchestrator node
        graph.add_node("orchestrator", self._orchestrator_node)
        
        # Add all specialized agent nodes
        for agent_name, agent in self.agent_registry.get_all_agents().items():
            graph.add_node(agent_name, agent.node_function)
        
        # Add human interaction node
        graph.add_node("human_interaction", self._human_interaction_node)
        
        # Add completion node
        graph.add_node("completion", self._completion_node)
        
        # Set entry point
        graph.add_edge(START, "orchestrator")
        
        # Add conditional edges from orchestrator to all other nodes
        graph.add_conditional_edges(
            "orchestrator",
            self._route_decision,
            {
                "researcher": "researcher",
                "processor": "processor", 
                "retriever": "retriever",
                "writer": "writer",
                "reviewer": "reviewer",
                "analyst": "analyst",
                "editor": "editor",
                "human_interaction": "human_interaction",
                "completion": "completion",
                "continue": "orchestrator"
            }
        )
        
        # All agents return to orchestrator for next decision
        for agent_name in self.agent_registry.get_all_agents().keys():
            graph.add_edge(agent_name, "orchestrator")
        
        graph.add_edge("human_interaction", "orchestrator")
        graph.add_edge("completion", END)
        
        return graph
    
    def compile(self, checkpointer: Optional[PostgresSaver] = None) -> StateGraph:
        """Compile the graph with optional checkpointer."""
        if checkpointer is None:
            # Create default PostgreSQL checkpointer
            checkpointer = PostgresSaver.from_conn_string(
                self.config.database.connection_string
            )
            checkpointer.setup()
        
        return self.graph.compile(checkpointer=checkpointer)
    
    def _orchestrator_node(self, state: MetanalysisState) -> Dict[str, Any]:
        """
        Main orchestrator decision-making node.
        
        Analyzes current state and decides which agent to invoke next.
        """
        self.logger.info(
            "Orchestrator analyzing state",
            metanalysis_id=state["metanalysis_id"],
            current_step=state["current_step"],
            status=state["status"]
        )
        
        # Check if we're in an error state
        if state["status"] == "error":
            return self._handle_error_state(state)
        
        # Check if we're completed
        if state["status"] == "completed":
            return {"next_agent": "completion"}
        
        # Analyze current state and make decision
        decision = self._make_decision(state)
        
        # Log the decision
        add_agent_message(
            state,
            "orchestrator",
            "decision",
            {
                "next_agent": decision.next_agent,
                "reasoning": decision.reasoning,
                "priority": decision.priority,
                "requires_human_approval": decision.requires_human_approval
            }
        )
        
        self.logger.info(
            "Orchestrator decision made",
            next_agent=decision.next_agent,
            reasoning=decision.reasoning,
            requires_approval=decision.requires_human_approval
        )
        
        # Update state with decision
        updated_state = {
            "next_agent": decision.next_agent,
            "current_step": f"routing_to_{decision.next_agent}",
            "status": "in_progress"
        }
        
        # Check if human approval is required
        if decision.requires_human_approval:
            updated_state["status"] = "awaiting_approval"
            updated_state["next_agent"] = "human_interaction"
        
        return updated_state
    
    def _make_decision(self, state: MetanalysisState) -> OrchestratorDecision:
        """
        Core decision-making logic for the orchestrator.
        
        This is where the orchestrator analyzes the state and determines
        what should happen next in the meta-analysis process.
        """
        # Prepare context for LLM decision
        context = self._prepare_decision_context(state)
        
        # Use LLM to make structured decision
        decision_prompt = self._create_decision_prompt(context)
        
        # Get structured decision from LLM
        structured_llm = self.llm.with_structured_output(OrchestratorDecision)
        decision = structured_llm.invoke([
            SystemMessage(content=self._get_orchestrator_system_prompt()),
            HumanMessage(content=decision_prompt)
        ])
        
        return decision
    
    def _prepare_decision_context(self, state: MetanalysisState) -> Dict[str, Any]:
        """Prepare context information for decision making."""
        return {
            "metanalysis_id": state["metanalysis_id"],
            "current_step": state["current_step"],
            "status": state["status"],
            "pico": state["pico"],
            "urls_to_process": len(state["urls_to_process"]),
            "processed_urls": len(state["processed_urls"]),
            "search_queries": len(state["search_queries"]),
            "has_insights": bool(state["insights"]),
            "has_drafts": bool(state["report_drafts"]),
            "has_final_report": bool(state["final_report_not_edited"]),
            "error_count": len(state["error_messages"]),
            "recent_agent_messages": state["agents_messages"][-5:] if state["agents_messages"] else [],
            "pending_human_messages": [
                msg for msg in state["human_messages"] 
                if msg.get("requires_response", False)
            ]
        }
    
    def _create_decision_prompt(self, context: Dict[str, Any]) -> str:
        """Create the decision prompt for the LLM."""
        return f"""
        Analyze the current state of the meta-analysis and decide what should happen next.
        
        Current Context:
        - Meta-analysis ID: {context['metanalysis_id']}
        - Current Step: {context['current_step']}
        - Status: {context['status']}
        - PICO Framework: {context['pico']}
        - URLs to process: {context['urls_to_process']}
        - Processed URLs: {context['processed_urls']}
        - Search queries executed: {context['search_queries']}
        - Has insights: {context['has_insights']}
        - Has report drafts: {context['has_drafts']}
        - Has final report: {context['has_final_report']}
        - Error count: {context['error_count']}
        - Pending human responses: {len(context['pending_human_messages'])}
        
        Recent agent activity:
        {context['recent_agent_messages']}
        
        Available agents and their purposes:
        - researcher: Search for scientific literature and collect URLs
        - processor: Extract and process content from URLs, create embeddings
        - retriever: Search and retrieve relevant information from processed content
        - writer: Generate structured reports and analysis
        - reviewer: Review and validate content quality
        - analyst: Perform statistical analysis and create visualizations
        - editor: Final editing and formatting of reports
        - human_interaction: Get human feedback or approval
        - completion: Finalize the meta-analysis
        
        Based on the current state, determine the next agent to invoke and provide clear reasoning.
        """
    
    def _get_orchestrator_system_prompt(self) -> str:
        """Get the system prompt for the orchestrator."""
        return """
        You are the central orchestrator of a meta-analysis agent system. Your role is to:
        
        1. Analyze the current state of the meta-analysis process
        2. Decide which specialized agent should be invoked next
        3. Ensure the process follows a logical sequence
        4. Identify when human intervention is needed
        5. Maintain quality and progress toward completion
        
        Decision-making principles:
        - Always start with PICO definition if not complete
        - Literature search (researcher) comes before processing
        - Content must be processed before retrieval
        - Analysis requires processed and retrieved content
        - Writing comes after sufficient analysis
        - Review is essential before final editing
        - Human approval is required for critical decisions
        
        You must provide structured decisions with clear reasoning.
        """
    
    def _route_decision(self, state: MetanalysisState) -> str:
        """Route to the next node based on orchestrator decision."""
        next_agent = state.get("next_agent", "completion")
        
        # Map agent names to valid node names
        valid_agents = {
            "researcher", "processor", "retriever", "writer", 
            "reviewer", "analyst", "editor", "human_interaction", 
            "completion", "continue"
        }
        
        if next_agent in valid_agents:
            return next_agent
        else:
            self.logger.warning(
                "Invalid agent decision, defaulting to completion",
                invalid_agent=next_agent
            )
            return "completion"
    
    def _human_interaction_node(self, state: MetanalysisState) -> Dict[str, Any]:
        """Handle human interaction requirements."""
        self.logger.info(
            "Human interaction required",
            metanalysis_id=state["metanalysis_id"]
        )
        
        # This would typically pause for human input
        # For now, we'll simulate approval
        add_agent_message(
            state,
            "human_interaction",
            "notification",
            {"message": "Human interaction node triggered - awaiting input"}
        )
        
        return {
            "status": "awaiting_human_input",
            "current_step": "human_interaction"
        }
    
    def _completion_node(self, state: MetanalysisState) -> Dict[str, Any]:
        """Handle completion of the meta-analysis."""
        self.logger.info(
            "Meta-analysis completion",
            metanalysis_id=state["metanalysis_id"]
        )
        
        add_agent_message(
            state,
            "orchestrator",
            "completion",
            {"message": "Meta-analysis process completed successfully"}
        )
        
        return {
            "status": "completed",
            "completed_at": datetime.now(),
            "current_step": "completed"
        }
    
    def _handle_error_state(self, state: MetanalysisState) -> Dict[str, Any]:
        """Handle error states and recovery."""
        error_count = len(state["error_messages"])
        
        if error_count >= self.config.max_retries:
            self.logger.error(
                "Maximum retries exceeded, requiring human intervention",
                metanalysis_id=state["metanalysis_id"],
                error_count=error_count
            )
            return {"next_agent": "human_interaction"}
        
        # Attempt recovery
        self.logger.info(
            "Attempting error recovery",
            metanalysis_id=state["metanalysis_id"],
            error_count=error_count
        )
        
        return {
            "status": "recovering",
            "next_agent": "orchestrator",
            "current_step": "error_recovery"
        }
    
    async def run_metanalysis(
        self,
        human_request: str,
        metanalysis_id: Optional[str] = None,
        pico: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete meta-analysis from start to finish.
        
        Args:
            human_request: The initial human request
            metanalysis_id: Optional ID, will be generated if not provided
            pico: Optional PICO framework
            
        Returns:
            Final state of the meta-analysis
        """
        if not metanalysis_id:
            metanalysis_id = str(uuid.uuid4())
        
        self.logger.info(
            "Starting meta-analysis",
            metanalysis_id=metanalysis_id,
            request=human_request
        )
        
        # Create initial state
        from .state import create_initial_state
        initial_state = create_initial_state(
            metanalysis_id=metanalysis_id,
            human_request=human_request,
            pico=pico
        )
        
        # Compile graph with checkpointer
        compiled_graph = self.compile()
        
        # Run the graph
        config = {"configurable": {"thread_id": metanalysis_id}}
        
        try:
            final_state = await compiled_graph.ainvoke(initial_state, config)
            
            self.logger.info(
                "Meta-analysis completed",
                metanalysis_id=metanalysis_id,
                status=final_state["status"]
            )
            
            return final_state
            
        except Exception as e:
            self.logger.error(
                "Meta-analysis failed",
                metanalysis_id=metanalysis_id,
                error=str(e)
            )
            raise