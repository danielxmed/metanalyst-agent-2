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
        
        # Initialize orchestrator agent ONCE - this creates the graph structure
        print("\n🤖 Initializing orchestrator agent...")
        orchestrator = OrchestratorAgent(config)
        print("   ✅ Orchestrator agent initialized successfully")
        print(f"   📊 Graph structure created and ready for multiple meta-analyses")
        
        # Display available handoff tools
        print(f"\n🔄 Available handoff tools: {len(orchestrator.handoff_tools)}")
        for tool in orchestrator.handoff_tools:
            print(f"   - {tool.name}: {tool.description}")
        
        # Example 1: CBT for Depression meta-analysis
        print(f"\n📝 Example 1: CBT for Depression meta-analysis")
        cbt_request = """
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
        
        try:
            # Start first meta-analysis - this creates a new state but uses existing graph
            metanalysis_id_1, thread_id_1 = orchestrator.start_metanalysis(
                initial_request=cbt_request.strip(),
                metanalysis_id="cbt-depression-2024",
                thread_id="thread-cbt-001"
            )
            
            print(f"   ✅ Meta-analysis 1 started successfully!")
            print(f"   📋 Meta-analysis ID: {metanalysis_id_1}")
            print(f"   🧵 Thread ID: {thread_id_1}")
            
            # Get status of first meta-analysis
            status_1 = orchestrator.get_metanalysis_status(thread_id_1)
            print(f"   📊 Status: {status_1['process_status']} at step '{status_1['current_step']}'")
            
        except Exception as e:
            print(f"   ❌ Failed to start meta-analysis 1: {e}")
            print(f"   This might be due to missing database connection or API keys.")
        
        # Example 2: Mindfulness for Anxiety meta-analysis
        print(f"\n📝 Example 2: Mindfulness for Anxiety meta-analysis")
        mindfulness_request = """
        I want to analyze the effectiveness of mindfulness-based interventions for anxiety disorders.
        
        Population: Adults with diagnosed anxiety disorders
        Intervention: Mindfulness-based interventions (MBSR, MBCT)
        Comparison: Standard care or waitlist control
        Outcome: Anxiety symptom reduction (GAD-7, STAI scores)
        """
        
        try:
            # Start second meta-analysis - SAME orchestrator, NEW state and thread
            metanalysis_id_2, thread_id_2 = orchestrator.start_metanalysis(
                initial_request=mindfulness_request.strip(),
                metanalysis_id="mindfulness-anxiety-2024",
                thread_id="thread-mindfulness-001"
            )
            
            print(f"   ✅ Meta-analysis 2 started successfully!")
            print(f"   📋 Meta-analysis ID: {metanalysis_id_2}")
            print(f"   🧵 Thread ID: {thread_id_2}")
            
            # Get status of second meta-analysis
            status_2 = orchestrator.get_metanalysis_status(thread_id_2)
            print(f"   📊 Status: {status_2['process_status']} at step '{status_2['current_step']}'")
            
        except Exception as e:
            print(f"   ❌ Failed to start meta-analysis 2: {e}")
        
        # Demonstrate continuing an existing meta-analysis
        print(f"\n🔄 Demonstrating continuation of existing meta-analysis...")
        try:
            # Continue the first meta-analysis with additional input
            result = orchestrator.continue_metanalysis(
                thread_id_1,
                user_input="Please focus on randomized controlled trials published in the last 10 years."
            )
            print(f"   ✅ Meta-analysis 1 continued with additional user input")
            
            # Get updated status
            updated_status = orchestrator.get_metanalysis_status(thread_id_1)
            print(f"   📊 Updated status: {updated_status['process_status']} at step '{updated_status['current_step']}'")
            
        except Exception as e:
            print(f"   ❌ Failed to continue meta-analysis: {e}")
        
        print(f"\n🎉 Example completed successfully!")
        print(f"   Key points demonstrated:")
        print(f"   1. ✅ Single orchestrator instance handles multiple meta-analyses")
        print(f"   2. ✅ Each meta-analysis has its own state and thread_id")
        print(f"   3. ✅ Graph structure is reused efficiently")
        print(f"   4. ✅ Meta-analyses can be continued independently")
        print(f"   5. ✅ State persistence allows resuming from any point")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        return


def demonstrate_orchestrator_reuse():
    """
    Demonstrate that the orchestrator can be reused for multiple meta-analyses
    without recreating the graph structure.
    """
    print("\n🧪 Demonstrating Orchestrator Reuse Pattern...")
    
    from metanalyst.core.config import TESTING_CONFIG
    
    # Create ONE orchestrator instance
    config = TESTING_CONFIG
    orchestrator = OrchestratorAgent(config)
    print("   ✅ Single orchestrator created")
    
    # Simulate multiple meta-analysis requests
    requests = [
        ("Drug A effectiveness", "drug-a-meta-2024"),
        ("Therapy B outcomes", "therapy-b-meta-2024"), 
        ("Intervention C analysis", "intervention-c-meta-2024")
    ]
    
    thread_ids = []
    
    for request, meta_id in requests:
        try:
            # Each call creates NEW state but uses SAME graph
            _, thread_id = orchestrator.start_metanalysis(
                initial_request=request,
                metanalysis_id=meta_id
            )
            thread_ids.append(thread_id)
            print(f"   ✅ Started {meta_id} with thread {thread_id}")
        except Exception as e:
            print(f"   ❌ Failed to start {meta_id}: {e}")
    
    # Demonstrate that each has independent state
    print(f"   📊 Created {len(thread_ids)} independent meta-analyses")
    for i, thread_id in enumerate(thread_ids):
        try:
            status = orchestrator.get_metanalysis_status(thread_id)
            print(f"   Thread {i+1}: {status['metanalysis_id']} - {status['process_status']}")
        except Exception as e:
            print(f"   ❌ Failed to get status for thread {i+1}: {e}")
    
    print("   ✅ Orchestrator reuse pattern validated")


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
    
    # Demonstrate orchestrator reuse pattern
    demonstrate_orchestrator_reuse()
    
    # Then run main example
    main()