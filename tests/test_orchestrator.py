"""
Tests for the orchestrator agent functionality.

This module contains tests to validate the orchestrator agent setup,
state management, and basic functionality.
"""

import pytest
import os
from datetime import datetime
from unittest.mock import Mock, patch

from metanalyst.core.config import MetanalystConfig, TESTING_CONFIG
from metanalyst.core.state import create_initial_state, MetanalysisState
from metanalyst.core.orchestrator import OrchestratorAgent
from metanalyst.agents.handoffs import HandoffTarget, get_all_handoff_tools


class TestOrchestratorSetup:
    """Test orchestrator agent initialization and setup."""
    
    def test_config_creation(self):
        """Test that configuration can be created with defaults."""
        config = MetanalystConfig()
        assert config.llm.provider == "anthropic"
        assert config.llm.model_name == "claude-3-5-sonnet-latest"
        assert config.database.postgres_url is not None
    
    def test_testing_config(self):
        """Test that testing configuration is properly set up."""
        config = TESTING_CONFIG
        assert config.llm.model_name == "claude-3-haiku-20240307"
        assert config.system.require_human_approval is False
        assert config.system.log_level == "DEBUG"
    
    def test_initial_state_creation(self):
        """Test that initial state is created correctly."""
        metanalysis_id = "test-123"
        initial_request = "Analyze the effectiveness of drug X for condition Y"
        
        state = create_initial_state(metanalysis_id, initial_request)
        
        assert state["metanalysis_id"] == metanalysis_id
        assert state["current_step"] == "initialization"
        assert state["next_agent"] == "orchestrator"
        assert state["process_status"] == "active"
        assert len(state["human_messages"]) == 1
        assert state["human_messages"][0].content == initial_request
        assert state["human_messages"][0].message_type == "initial_request"


class TestHandoffTools:
    """Test handoff tools functionality."""
    
    def test_handoff_tools_available(self):
        """Test that all expected handoff tools are available."""
        tools = get_all_handoff_tools()
        
        # Should have tools for all agents
        expected_count = len(HandoffTarget) - 1  # Minus orchestrator
        assert len(tools) >= expected_count
        
        # Check tool names
        tool_names = [tool.name for tool in tools]
        assert "handoff_to_researcher" in tool_names
        assert "handoff_to_processor" in tool_names
        assert "handoff_to_writer" in tool_names
    
    def test_handoff_tool_execution(self):
        """Test that handoff tools can be executed."""
        from metanalyst.agents.handoffs import researcher_handoff
        
        # Mock execution - tool should return a Command object
        result = researcher_handoff.invoke({
            "task_description": "Search for literature on topic X",
            "priority": 2,
            "context": {"topic": "test"},
            "expected_output": "List of relevant papers"
        })
        
        # Should return a Command object
        from langgraph.types import Command
        assert isinstance(result, Command)
        assert result.goto == "researcher"


class TestStateManagement:
    """Test state management functionality."""
    
    def test_state_updates(self):
        """Test that state can be updated correctly."""
        from metanalyst.core.state import add_agent_message, set_next_step
        
        # Create initial state
        state = create_initial_state("test-id", "test request")
        
        # Add agent message
        updated_state = add_agent_message(
            state,
            agent_name="orchestrator",
            content="Starting analysis",
            message_type="status"
        )
        
        assert len(updated_state["agents_messages"]) == 1
        assert updated_state["agents_messages"][0].agent_name == "orchestrator"
        assert updated_state["agents_messages"][0].content == "Starting analysis"
        
        # Update next step
        updated_state = set_next_step(updated_state, "pico_definition", "orchestrator")
        assert updated_state["current_step"] == "pico_definition"
        assert updated_state["next_agent"] == "orchestrator"
    
    def test_state_summary(self):
        """Test that state summary provides correct information."""
        from metanalyst.core.state import get_state_summary
        
        state = create_initial_state("test-id", "test request")
        summary = get_state_summary(state)
        
        assert summary["metanalysis_id"] == "test-id"
        assert summary["current_step"] == "initialization"
        assert summary["process_status"] == "active"
        assert summary["human_messages_count"] == 1
        assert summary["agent_messages_count"] == 0


class TestOrchestratorAgent:
    """Test orchestrator agent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = TESTING_CONFIG.copy()
        # Override to avoid actual database/API calls
        return config
    
    @patch('metanalyst.core.orchestrator.get_llm_instance')
    @patch('metanalyst.core.orchestrator.get_postgres_checkpointer')
    def test_orchestrator_initialization(self, mock_checkpointer, mock_llm, mock_config):
        """Test that orchestrator can be initialized."""
        # Mock the dependencies
        mock_llm.return_value = Mock()
        mock_checkpointer.return_value = Mock()
        
        orchestrator = OrchestratorAgent(mock_config)
        
        assert orchestrator.config == mock_config
        assert orchestrator.llm is not None
        assert orchestrator.checkpointer is not None
        assert len(orchestrator.handoff_tools) > 0
    
    @patch('metanalyst.core.orchestrator.get_llm_instance')
    @patch('metanalyst.core.orchestrator.get_postgres_checkpointer')
    def test_orchestrator_tools_binding(self, mock_checkpointer, mock_llm, mock_config):
        """Test that orchestrator tools are properly bound."""
        # Mock the dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.bind_tools = Mock(return_value=Mock())
        mock_llm.return_value = mock_llm_instance
        mock_checkpointer.return_value = Mock()
        
        orchestrator = OrchestratorAgent(mock_config)
        
        # Verify that bind_tools was called
        mock_llm_instance.bind_tools.assert_called_once()
        
        # Check that tools were bound (handoff tools + internal tools)
        call_args = mock_llm_instance.bind_tools.call_args[0][0]
        assert len(call_args) > len(orchestrator.handoff_tools)  # Should include internal tools


if __name__ == "__main__":
    pytest.main([__file__])