#!/usr/bin/env python3
"""
Simple validation script for the Metanalyst Agent framework.

This script validates the basic structure and imports without requiring
external dependencies like pytest or API keys.
"""

import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from metanalyst.config import MetanalystConfig, LLMProvider
        print("✅ Config module imported successfully")
        
        from metanalyst.state import MetanalysisState, create_initial_state
        print("✅ State module imported successfully")
        
        from metanalyst.agents.base import BaseAgent
        print("✅ Base agent imported successfully")
        
        # Test agent imports (these might fail due to missing dependencies)
        try:
            from metanalyst.agents.researcher import ResearcherAgent
            print("✅ Researcher agent imported successfully")
        except ImportError as e:
            print(f"⚠️  Researcher agent import failed (expected): {e}")
        
        print("✅ Core imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration creation and validation."""
    print("\n🧪 Testing configuration...")
    
    try:
        from metanalyst.config import MetanalystConfig, LLMProvider
        
        # Test default configuration
        config = MetanalystConfig()
        
        assert config.system_name == "Metanalyst Agent"
        assert config.agents.orchestrator.provider == LLMProvider.ANTHROPIC
        assert config.agents.orchestrator.model_name == "claude-3-5-sonnet-latest"
        assert config.tavily.max_results == 10
        assert config.database.host == "localhost"
        
        print("✅ Default configuration created successfully")
        
        # Test LLM provider switching
        config.update_llm_provider(LLMProvider.OPENAI, "gpt-4-turbo-preview")
        assert config.agents.orchestrator.provider == LLMProvider.OPENAI
        assert config.agents.researcher.provider == LLMProvider.OPENAI
        
        print("✅ LLM provider switching works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_state():
    """Test state creation and manipulation."""
    print("\n🧪 Testing state management...")
    
    try:
        from metanalyst.state import create_initial_state, add_agent_message
        
        # Test initial state creation
        metanalysis_id = "test-meta-001"
        human_request = "Test meta-analysis request"
        pico = {
            "population": "Test population",
            "intervention": "Test intervention",
            "comparison": "Test comparison",
            "outcome": "Test outcome"
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
        assert isinstance(state["created_at"], datetime)
        
        print("✅ Initial state created successfully")
        
        # Test adding agent message
        add_agent_message(
            state,
            "test_agent",
            "test_message",
            {"test": "content"}
        )
        
        assert len(state["agents_messages"]) == 1
        assert state["agents_messages"][0]["agent_name"] == "test_agent"
        
        print("✅ Agent message addition works")
        
        return True
        
    except Exception as e:
        print(f"❌ State test failed: {e}")
        return False

def test_structure():
    """Test overall framework structure."""
    print("\n🧪 Testing framework structure...")
    
    try:
        # Check that all required directories exist
        required_dirs = [
            "src/metanalyst",
            "src/metanalyst/agents",
            "tests",
            "docs"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"❌ Missing directory: {dir_path}")
                return False
        
        print("✅ All required directories exist")
        
        # Check that all required files exist
        required_files = [
            "src/metanalyst/__init__.py",
            "src/metanalyst/config.py",
            "src/metanalyst/state.py",
            "src/metanalyst/orchestrator.py",
            "src/metanalyst/llm_factory.py",
            "src/metanalyst/agents/__init__.py",
            "src/metanalyst/agents/base.py",
            "src/metanalyst/agents/registry.py",
            "requirements.txt",
            ".env.example"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Missing file: {file_path}")
                return False
        
        print("✅ All required files exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧬 Metanalyst Agent Framework Validation")
    print("=" * 50)
    
    tests = [
        ("Framework Structure", test_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("State Management", test_state),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Framework structure is valid.")
        print("\n📝 Next steps:")
        print("   1. Set up environment variables (.env file)")
        print("   2. Install dependencies (pip install -r requirements.txt)")
        print("   3. Set up PostgreSQL database")
        print("   4. Run: python example_usage.py")
        print("   5. Implement full agent functionality")
        
        print("\n🏗️  Framework Status:")
        print("   ✅ State management: Complete")
        print("   ✅ Configuration system: Complete")
        print("   ✅ Agent registry: Complete")
        print("   ✅ Orchestrator structure: Complete")
        print("   ✅ LLM factory: Complete")
        print("   ⚠️  Agent implementations: Stubs only")
        print("   ⚠️  Database integration: Needs setup")
        print("   ⚠️  Vector store: Needs implementation")
        
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)