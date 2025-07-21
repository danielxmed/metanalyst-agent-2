"""
Basic usage example for the metanalyst agent system.

This script demonstrates how to initialize and use the orchestrator agent
for automated meta-analysis generation.
"""

import os
import logging
from pathlib import Path

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from metanalyst.core.config import MetanalystConfig, DEVELOPMENT_CONFIG
from metanalyst.core.orchestrator import OrchestratorAgent
from metanalyst.core.state import get_state_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function demonstrating basic usage of the metanalyst system.
    """
    
    print("🔬 Metanalyst Agent System - Basic Usage Example")
    print("=" * 50)
    
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("✅ Loaded environment variables from .env file")
    else:
        print("⚠️  No .env file found. Using default configuration.")
        print("   Please copy .env.example to .env and configure your API keys.")
    
    # Check for required environment variables
    required_vars = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("   Please set these variables in your .env file or environment.")
        return
    
    try:
        # Initialize configuration
        print("\n🔧 Initializing configuration...")
        config = DEVELOPMENT_CONFIG
        print(f"   LLM Provider: {config.llm.provider}")
        print(f"   LLM Model: {config.llm.model_name}")
        print(f"   Database URL: {config.database.postgres_url}")
        
        # Initialize orchestrator agent
        print("\n🤖 Initializing orchestrator agent...")
        orchestrator = OrchestratorAgent(config)
        print("   ✅ Orchestrator agent initialized successfully")
        
        # Display available handoff tools
        print(f"\n🔄 Available handoff tools: {len(orchestrator.handoff_tools)}")
        for tool in orchestrator.handoff_tools:
            print(f"   - {tool.name}: {tool.description}")
        
        # Example meta-analysis request
        example_request = """
        I need to conduct a meta-analysis on the effectiveness of cognitive behavioral therapy (CBT) 
        for treating depression in adults. Please help me:
        
        1. Define the PICO framework
        2. Search for relevant literature
        3. Extract and analyze data
        4. Generate a comprehensive report
        
        Population: Adults (18+ years) with major depressive disorder
        Intervention: Cognitive Behavioral Therapy
        Comparison: Control groups (waitlist, placebo, or treatment as usual)
        Outcome: Depression severity scores (e.g., Beck Depression Inventory, Hamilton Rating Scale)
        """
        
        print(f"\n📝 Example meta-analysis request:")
        print(example_request.strip())
        
        # Start the meta-analysis process
        print(f"\n🚀 Starting meta-analysis process...")
        
        try:
            metanalysis_id, thread_id = orchestrator.start_metanalysis(
                initial_request=example_request.strip(),
                metanalysis_id="example-cbt-depression-2024",
                thread_id="example-thread-001"
            )
            
            print(f"   ✅ Meta-analysis started successfully!")
            print(f"   📋 Meta-analysis ID: {metanalysis_id}")
            print(f"   🧵 Thread ID: {thread_id}")
            
            # Get initial status
            status = orchestrator.get_metanalysis_status(thread_id)
            print(f"\n📊 Initial status:")
            for key, value in status.items():
                print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"   ❌ Failed to start meta-analysis: {e}")
            print(f"   This might be due to missing database connection or API keys.")
            return
        
        print(f"\n🎉 Example completed successfully!")
        print(f"   The orchestrator agent is now ready to coordinate the meta-analysis process.")
        print(f"   In a real scenario, it would:")
        print(f"   1. Analyze the request and define PICO framework")
        print(f"   2. Hand off to researcher agent for literature search")
        print(f"   3. Coordinate with processor agent for data extraction")
        print(f"   4. Manage the entire workflow until completion")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        return


def test_configuration():
    """Test the configuration system."""
    print("\n🧪 Testing configuration system...")
    
    # Test default configuration
    default_config = MetanalystConfig()
    print(f"   Default LLM provider: {default_config.llm.provider}")
    print(f"   Default model: {default_config.llm.model_name}")
    
    # Test development configuration
    dev_config = DEVELOPMENT_CONFIG
    print(f"   Development config - Human approval required: {dev_config.system.require_human_approval}")
    print(f"   Development config - Log level: {dev_config.system.log_level}")
    
    print("   ✅ Configuration system working correctly")


def test_handoff_tools():
    """Test the handoff tools system."""
    print("\n🧪 Testing handoff tools...")
    
    from metanalyst.agents.handoffs import (
        get_all_handoff_tools,
        researcher_handoff,
        HandoffTarget
    )
    
    # Test getting all tools
    tools = get_all_handoff_tools()
    print(f"   Available tools: {len(tools)}")
    
    # Test individual tool
    try:
        result = researcher_handoff.invoke({
            "task_description": "Test search for literature",
            "priority": 1,
            "context": {"test": True}
        })
        print(f"   ✅ Handoff tool execution successful")
        print(f"   Result type: {type(result).__name__}")
    except Exception as e:
        print(f"   ❌ Handoff tool test failed: {e}")


if __name__ == "__main__":
    # Run tests first
    test_configuration()
    test_handoff_tools()
    
    # Then run main example
    main()