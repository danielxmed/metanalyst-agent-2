"""
Agent Registry for managing all specialized agents in the Metanalyst system.

This module provides a centralized registry for creating and managing
all the specialized worker agents.
"""

from typing import Dict, Any, Optional, List
import structlog

from ..config import MetanalystConfig
from ..llm_factory import create_llm
from .base import BaseAgent
from .researcher import ResearcherAgent
from .processor import ProcessorAgent
from .retriever import RetrieverAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent
from .analyst import AnalystAgent
from .editor import EditorAgent

logger = structlog.get_logger()


class AgentRegistry:
    """
    Registry for managing all specialized agents in the Metanalyst system.
    
    The registry creates and manages instances of all specialized agents,
    providing them to the orchestrator for the hub-and-spoke architecture.
    """
    
    def __init__(self, config: MetanalystConfig):
        """
        Initialize the agent registry with configuration.
        
        Args:
            config: Metanalyst configuration containing agent settings
        """
        self.config = config
        self.logger = logger.bind(component="agent_registry")
        self._agents: Dict[str, BaseAgent] = {}
        
        # Initialize all agents
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """Initialize all specialized agents."""
        self.logger.info("Initializing specialized agents")
        
        try:
            # Create LLMs for each agent
            agent_configs = {
                "researcher": self.config.agents.researcher,
                "processor": self.config.agents.processor,
                "retriever": self.config.agents.retriever,
                "writer": self.config.agents.writer,
                "reviewer": self.config.agents.reviewer,
                "analyst": self.config.agents.analyst,
                "editor": self.config.agents.editor
            }
            
            # Initialize each agent with its specific configuration
            self._agents["researcher"] = ResearcherAgent(
                name="researcher",
                llm=create_llm(agent_configs["researcher"]),
                config=agent_configs["researcher"],
                tavily_config=self.config.tavily
            )
            
            self._agents["processor"] = ProcessorAgent(
                name="processor",
                llm=create_llm(agent_configs["processor"]),
                config=agent_configs["processor"],
                vectorstore_config=self.config.vectorstore
            )
            
            self._agents["retriever"] = RetrieverAgent(
                name="retriever",
                llm=create_llm(agent_configs["retriever"]),
                config=agent_configs["retriever"],
                vectorstore_config=self.config.vectorstore
            )
            
            self._agents["writer"] = WriterAgent(
                name="writer",
                llm=create_llm(agent_configs["writer"]),
                config=agent_configs["writer"]
            )
            
            self._agents["reviewer"] = ReviewerAgent(
                name="reviewer",
                llm=create_llm(agent_configs["reviewer"]),
                config=agent_configs["reviewer"]
            )
            
            self._agents["analyst"] = AnalystAgent(
                name="analyst",
                llm=create_llm(agent_configs["analyst"]),
                config=agent_configs["analyst"]
            )
            
            self._agents["editor"] = EditorAgent(
                name="editor",
                llm=create_llm(agent_configs["editor"]),
                config=agent_configs["editor"]
            )
            
            self.logger.info(
                "All agents initialized successfully",
                agent_count=len(self._agents),
                agent_names=list(self._agents.keys())
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize agents",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get a specific agent by name.
        
        Args:
            name: Name of the agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """
        Get all registered agents.
        
        Returns:
            Dictionary mapping agent names to agent instances
        """
        return self._agents.copy()
    
    def get_agent_names(self) -> List[str]:
        """
        Get list of all agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get capabilities of all agents.
        
        Returns:
            Dictionary mapping agent names to their capabilities
        """
        return {
            name: agent.get_capabilities()
            for name, agent in self._agents.items()
        }
    
    def validate_agents(self) -> List[str]:
        """
        Validate all agents and return any issues found.
        
        Returns:
            List of validation issues (empty if all valid)
        """
        issues = []
        
        for name, agent in self._agents.items():
            try:
                # Check if agent has required methods
                if not hasattr(agent, 'execute'):
                    issues.append(f"Agent {name} missing execute method")
                
                if not hasattr(agent, 'get_system_prompt'):
                    issues.append(f"Agent {name} missing get_system_prompt method")
                
                if not hasattr(agent, 'node_function'):
                    issues.append(f"Agent {name} missing node_function method")
                
                # Check if agent's LLM is properly configured
                if not agent.llm:
                    issues.append(f"Agent {name} has no LLM configured")
                
                # Test system prompt
                try:
                    prompt = agent.get_system_prompt()
                    if not isinstance(prompt, str) or len(prompt) == 0:
                        issues.append(f"Agent {name} has invalid system prompt")
                except Exception as e:
                    issues.append(f"Agent {name} system prompt error: {e}")
                
            except Exception as e:
                issues.append(f"Agent {name} validation error: {e}")
        
        return issues
    
    def get_execution_order(self) -> List[str]:
        """
        Get recommended execution order for agents in a typical workflow.
        
        Returns:
            List of agent names in recommended execution order
        """
        return [
            "researcher",    # Search for literature
            "processor",     # Process and extract content
            "retriever",     # Retrieve relevant information
            "analyst",       # Perform statistical analysis
            "writer",        # Generate reports
            "reviewer",      # Review quality
            "editor"         # Final editing
        ]
    
    def get_dependencies(self) -> Dict[str, List[str]]:
        """
        Get dependencies between agents.
        
        Returns:
            Dictionary mapping agent names to their dependencies
        """
        return {
            "researcher": [],  # No dependencies
            "processor": ["researcher"],  # Needs URLs from researcher
            "retriever": ["processor"],   # Needs processed content
            "analyst": ["retriever"],     # Needs retrieved information
            "writer": ["analyst", "retriever"],  # Needs analysis and information
            "reviewer": ["writer"],       # Needs draft to review
            "editor": ["reviewer"]        # Needs reviewed content
        }
    
    def can_execute_parallel(self, agent_names: List[str]) -> bool:
        """
        Check if the given agents can execute in parallel.
        
        Args:
            agent_names: List of agent names to check
            
        Returns:
            True if agents can execute in parallel, False otherwise
        """
        dependencies = self.get_dependencies()
        
        # Check if any agent depends on another in the list
        for agent in agent_names:
            agent_deps = dependencies.get(agent, [])
            for dep in agent_deps:
                if dep in agent_names:
                    return False
        
        return True
    
    def get_next_agents(self, completed_agents: List[str]) -> List[str]:
        """
        Get list of agents that can execute next based on completed agents.
        
        Args:
            completed_agents: List of agents that have completed execution
            
        Returns:
            List of agent names that can execute next
        """
        dependencies = self.get_dependencies()
        all_agents = set(self.get_agent_names())
        remaining_agents = all_agents - set(completed_agents)
        
        next_agents = []
        
        for agent in remaining_agents:
            agent_deps = dependencies.get(agent, [])
            # Check if all dependencies are satisfied
            if all(dep in completed_agents for dep in agent_deps):
                next_agents.append(agent)
        
        return next_agents
    
    def restart_agent(self, agent_name: str) -> bool:
        """
        Restart a specific agent (recreate its instance).
        
        Args:
            agent_name: Name of the agent to restart
            
        Returns:
            True if restart successful, False otherwise
        """
        try:
            if agent_name not in self._agents:
                self.logger.warning("Cannot restart unknown agent", agent=agent_name)
                return False
            
            self.logger.info("Restarting agent", agent=agent_name)
            
            # Get the agent's configuration
            agent_config = getattr(self.config.agents, agent_name)
            
            # Recreate the agent based on its type
            if agent_name == "researcher":
                self._agents[agent_name] = ResearcherAgent(
                    name=agent_name,
                    llm=create_llm(agent_config),
                    config=agent_config,
                    tavily_config=self.config.tavily
                )
            elif agent_name == "processor":
                self._agents[agent_name] = ProcessorAgent(
                    name=agent_name,
                    llm=create_llm(agent_config),
                    config=agent_config,
                    vectorstore_config=self.config.vectorstore
                )
            # Add other agent types as needed...
            
            self.logger.info("Agent restarted successfully", agent=agent_name)
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to restart agent",
                agent=agent_name,
                error=str(e)
            )
            return False