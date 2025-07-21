"""
Base Agent class for all specialized agents in the Metanalyst system.

This module defines the common interface and functionality that all
specialized agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import structlog
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import MetanalysisState, add_agent_message, update_state_timestamp
from ..config import LLMConfig


class BaseAgent(ABC):
    """
    Base class for all specialized agents in the Metanalyst system.
    
    Each agent is designed to be invoked as a tool by the central orchestrator,
    following the agents-as-a-tool pattern.
    """
    
    def __init__(self, name: str, llm: BaseChatModel, config: LLMConfig):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for this agent
            llm: Language model instance for this agent
            config: LLM configuration for this agent
        """
        self.name = name
        self.llm = llm
        self.config = config
        self.logger = structlog.get_logger().bind(agent=name)
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            System prompt string defining the agent's role and capabilities
        """
        pass
    
    @abstractmethod
    def execute(self, state: MetanalysisState) -> Dict[str, Any]:
        """
        Execute the agent's core functionality.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            Dictionary with state updates
        """
        pass
    
    def node_function(self, state: MetanalysisState) -> Dict[str, Any]:
        """
        Main node function that can be used in LangGraph.
        
        This is the function that gets called when the orchestrator
        routes to this agent.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            Updated state
        """
        self.logger.info(
            "Agent execution started",
            metanalysis_id=state["metanalysis_id"],
            current_step=state["current_step"]
        )
        
        try:
            # Record agent activation
            add_agent_message(
                state,
                self.name,
                "activation",
                {
                    "message": f"{self.name} agent activated",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Execute the agent's core functionality
            result = self.execute(state)
            
            # Record successful completion
            add_agent_message(
                state,
                self.name,
                "completion",
                {
                    "message": f"{self.name} agent completed successfully",
                    "result_summary": self._summarize_result(result),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.info(
                "Agent execution completed successfully",
                metanalysis_id=state["metanalysis_id"],
                result_keys=list(result.keys())
            )
            
            # Update timestamp and return result
            update_state_timestamp(state)
            return result
            
        except Exception as e:
            # Record error
            error_msg = f"Error in {self.name} agent: {str(e)}"
            state["error_messages"].append(error_msg)
            
            add_agent_message(
                state,
                self.name,
                "error",
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.error(
                "Agent execution failed",
                metanalysis_id=state["metanalysis_id"],
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Update state to error status
            return {
                "status": "error",
                "current_step": f"error_in_{self.name}",
                "next_agent": "orchestrator"  # Return to orchestrator for error handling
            }
    
    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """
        Create a summary of the agent's result for logging.
        
        Args:
            result: Result dictionary from agent execution
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        for key, value in result.items():
            if isinstance(value, list):
                summary_parts.append(f"{key}: {len(value)} items")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: {len(value)} keys")
            elif isinstance(value, str) and len(value) > 100:
                summary_parts.append(f"{key}: {len(value)} characters")
            else:
                summary_parts.append(f"{key}: {type(value).__name__}")
        
        return ", ".join(summary_parts)
    
    def create_llm_prompt(self, context: str, task_description: str) -> List[Any]:
        """
        Create a prompt for the LLM with system and user messages.
        
        Args:
            context: Context information for the task
            task_description: Specific task to perform
            
        Returns:
            List of messages for LLM
        """
        return [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"""
            Context: {context}
            
            Task: {task_description}
            
            Please provide a detailed response following your role as {self.name}.
            """)
        ]
    
    def validate_state_requirements(self, state: MetanalysisState, required_fields: List[str]) -> bool:
        """
        Validate that the state contains required fields for this agent.
        
        Args:
            state: Current metanalysis state
            required_fields: List of required field names
            
        Returns:
            True if all required fields are present and valid
        """
        for field in required_fields:
            if field not in state:
                self.logger.warning(
                    "Missing required field in state",
                    field=field,
                    metanalysis_id=state.get("metanalysis_id")
                )
                return False
            
            # Check if field is empty when it shouldn't be
            value = state[field]
            if value is None or (isinstance(value, (list, dict, str)) and len(value) == 0):
                self.logger.warning(
                    "Required field is empty",
                    field=field,
                    metanalysis_id=state.get("metanalysis_id")
                )
                return False
        
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about this agent's capabilities.
        
        Returns:
            Dictionary describing agent capabilities
        """
        return {
            "name": self.name,
            "description": self.__doc__ or f"{self.name} agent",
            "llm_provider": self.config.provider,
            "llm_model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
    
    def should_execute(self, state: MetanalysisState) -> bool:
        """
        Determine if this agent should execute based on current state.
        
        This is a default implementation that can be overridden by subclasses
        to implement more sophisticated execution logic.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            True if agent should execute, False otherwise
        """
        # Default: execute if not in error state
        return state.get("status") != "error"
    
    def estimate_execution_time(self, state: MetanalysisState) -> int:
        """
        Estimate execution time for this agent in seconds.
        
        This can be overridden by subclasses to provide more accurate estimates
        based on the current state and workload.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            Estimated execution time in seconds
        """
        # Default estimate: 30 seconds
        return 30