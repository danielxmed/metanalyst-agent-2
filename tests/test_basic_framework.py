"""
Basic framework test for the Metanalyst Agent system.

This test demonstrates the core functionality and validates
the framework structure without requiring external APIs.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from src.metanalyst.config import MetanalystConfig, LLMProvider
from src.metanalyst.state import create_initial_state, MetanalysisState
from src.metanalyst.orchestrator import MetanalystOrchestrator
from src.metanalyst.agents.registry import AgentRegistry


class TestMetanalystFramework:
    """Test suite for the basic framework functionality."""
    
    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = MetanalystConfig()
        
        assert config.system_name == "Metanalyst Agent"
        assert config.agents.orchestrator.provider == LLMProvider.ANTHROPIC
        assert config.agents.orchestrator.model_name == "claude-3-5-sonnet-latest"
        assert config.tavily.max_results == 10
        assert config.database.host == "localhost"
    
    def test_state_creation(self):
        """Test initial state creation."""
        metanalysis_id = "test-meta-001"
        human_request = "Analyze the effectiveness of drug X for condition Y"
        pico = {
            "population": "Adults with condition Y",
            "intervention": "Drug X",
            "comparison": "Placebo",
            "outcome": "Symptom improvement"
        }
        
        state = create_initial_state(
            metanalysis_id=metanalysis_id,
            human_request=human_request,
            pico=pico
        )
        
        assert state["metanalysis_id"] == metanalysis_id
        assert state["pico"] == pico
        assert state["status"] == "initialized"
        assert len(state["human_messages"]) == 1
        assert state["human_messages"][0]["content"] == human_request
        assert isinstance(state["created_at"], datetime)
    
    @patch('src.metanalyst.llm_factory.create_llm')
    def test_agent_registry_initialization(self, mock_create_llm):
        """Test agent registry initialization."""
        # Mock LLM creation
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        config = MetanalystConfig()
        # Set dummy API keys to avoid validation errors
        config.agents.orchestrator.api_key = "test-key"
        config.agents.researcher.api_key = "test-key"
        config.agents.processor.api_key = "test-key"
        config.agents.retriever.api_key = "test-key"
        config.agents.writer.api_key = "test-key"
        config.agents.reviewer.api_key = "test-key"
        config.agents.analyst.api_key = "test-key"
        config.agents.editor.api_key = "test-key"
        
        registry = AgentRegistry(config)
        
        # Check that all agents are initialized
        expected_agents = [
            "researcher", "processor", "retriever", "writer",
            "reviewer", "analyst", "editor"
        ]
        
        for agent_name in expected_agents:
            agent = registry.get_agent(agent_name)
            assert agent is not None
            assert agent.name == agent_name
            assert agent.llm == mock_llm
    
    @patch('src.metanalyst.llm_factory.create_llm')
    def test_orchestrator_initialization(self, mock_create_llm):
        """Test orchestrator initialization."""
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        config = MetanalystConfig()
        # Set dummy API keys
        config.agents.orchestrator.api_key = "test-key"
        config.agents.researcher.api_key = "test-key"
        config.agents.processor.api_key = "test-key"
        config.agents.retriever.api_key = "test-key"
        config.agents.writer.api_key = "test-key"
        config.agents.reviewer.api_key = "test-key"
        config.agents.analyst.api_key = "test-key"
        config.agents.editor.api_key = "test-key"
        
        orchestrator = MetanalystOrchestrator(config)
        
        assert orchestrator.config == config
        assert orchestrator.llm == mock_llm
        assert orchestrator.agent_registry is not None
        assert orchestrator.graph is not None
    
    def test_agent_dependencies(self):
        """Test agent dependency mapping."""
        config = MetanalystConfig()
        # Set dummy API keys
        for agent_name in ["orchestrator", "researcher", "processor", "retriever", 
                          "writer", "reviewer", "analyst", "editor"]:
            getattr(config.agents, agent_name).api_key = "test-key"
        
        registry = AgentRegistry(config)
        dependencies = registry.get_dependencies()
        
        # Test some key dependencies
        assert dependencies["researcher"] == []  # No dependencies
        assert "researcher" in dependencies["processor"]  # Processor needs researcher
        assert "processor" in dependencies["retriever"]  # Retriever needs processor
        assert "retriever" in dependencies["analyst"]  # Analyst needs retriever
    
    def test_execution_order(self):
        """Test recommended execution order."""
        config = MetanalystConfig()
        # Set dummy API keys
        for agent_name in ["orchestrator", "researcher", "processor", "retriever", 
                          "writer", "reviewer", "analyst", "editor"]:
            getattr(config.agents, agent_name).api_key = "test-key"
        
        registry = AgentRegistry(config)
        order = registry.get_execution_order()
        
        expected_order = [
            "researcher", "processor", "retriever", "analyst",
            "writer", "reviewer", "editor"
        ]
        
        assert order == expected_order
    
    def test_parallel_execution_check(self):
        """Test parallel execution capability check."""
        config = MetanalystConfig()
        # Set dummy API keys
        for agent_name in ["orchestrator", "researcher", "processor", "retriever", 
                          "writer", "reviewer", "analyst", "editor"]:
            getattr(config.agents, agent_name).api_key = "test-key"
        
        registry = AgentRegistry(config)
        
        # These should be able to run in parallel (no dependencies between them)
        assert registry.can_execute_parallel(["researcher"]) == True
        
        # These should NOT be able to run in parallel (processor depends on researcher)
        assert registry.can_execute_parallel(["researcher", "processor"]) == False
    
    def test_next_agents_calculation(self):
        """Test calculation of next available agents."""
        config = MetanalystConfig()
        # Set dummy API keys
        for agent_name in ["orchestrator", "researcher", "processor", "retriever", 
                          "writer", "reviewer", "analyst", "editor"]:
            getattr(config.agents, agent_name).api_key = "test-key"
        
        registry = AgentRegistry(config)
        
        # Initially, only researcher can run (no dependencies)
        next_agents = registry.get_next_agents([])
        assert "researcher" in next_agents
        assert "processor" not in next_agents  # Depends on researcher
        
        # After researcher completes, processor can run
        next_agents = registry.get_next_agents(["researcher"])
        assert "processor" in next_agents
        assert "retriever" not in next_agents  # Depends on processor
    
    @patch('src.metanalyst.llm_factory.create_llm')
    def test_config_llm_provider_update(self, mock_create_llm):
        """Test updating LLM provider for all agents."""
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        config = MetanalystConfig()
        
        # Initially using Anthropic
        assert config.agents.orchestrator.provider == LLMProvider.ANTHROPIC
        
        # Update to OpenAI
        config.update_llm_provider(LLMProvider.OPENAI, "gpt-4-turbo-preview")
        
        # Check that all agents are updated
        assert config.agents.orchestrator.provider == LLMProvider.OPENAI
        assert config.agents.researcher.provider == LLMProvider.OPENAI
        assert config.agents.processor.provider == LLMProvider.OPENAI


if __name__ == "__main__":
    # Run basic framework validation
    test_suite = TestMetanalystFramework()
    
    print("🧪 Testing Metanalyst Framework...")
    
    try:
        test_suite.test_config_creation()
        print("✅ Configuration creation: PASSED")
        
        test_suite.test_state_creation()
        print("✅ State creation: PASSED")
        
        # Note: Other tests require mocking and are better run with pytest
        print("✅ Basic framework validation: PASSED")
        
        print("\n🎉 All basic tests passed! Framework is ready for development.")
        print("\n📝 Next steps:")
        print("   1. Set up environment variables (.env file)")
        print("   2. Install dependencies (pip install -r requirements.txt)")
        print("   3. Set up PostgreSQL database")
        print("   4. Implement full agent functionality")
        print("   5. Add comprehensive testing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise