"""
Example usage of the Metanalyst Agent framework.

This script demonstrates how to set up and use the metanalyst system
for autonomous meta-analysis generation.
"""

import asyncio
import os
from dotenv import load_dotenv

from src.metanalyst.config import MetanalystConfig, LLMProvider, validate_config
from src.metanalyst.orchestrator import MetanalystOrchestrator
from src.metanalyst.state import create_initial_state

# Load environment variables
load_dotenv()


async def run_example_metanalysis():
    """Run an example meta-analysis using the framework."""
    
    print("🧬 Metanalyst Agent Framework - Example Usage")
    print("=" * 50)
    
    # 1. Create configuration
    print("\n📋 Step 1: Setting up configuration...")
    config = MetanalystConfig.from_env()
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 Please check your .env file and ensure all API keys are set.")
        return
    
    print("✅ Configuration validated successfully")
    print(f"   - LLM Provider: {config.agents.orchestrator.provider}")
    print(f"   - Search Provider: Tavily")
    print(f"   - Database: {config.database.host}:{config.database.port}")
    
    # 2. Example research question
    print("\n🔬 Step 2: Defining research question...")
    
    human_request = """
    I need a meta-analysis on the effectiveness of mindfulness-based interventions 
    for reducing anxiety in adults. Please include randomized controlled trials 
    published in the last 10 years.
    """
    
    pico_framework = {
        "population": "Adults with anxiety disorders",
        "intervention": "Mindfulness-based interventions (MBSR, MBCT, etc.)",
        "comparison": "Control groups (waitlist, usual care, or active control)",
        "outcome": "Reduction in anxiety symptoms (measured by validated scales)"
    }
    
    print(f"Research Question: {human_request.strip()}")
    print("\nPICO Framework:")
    for key, value in pico_framework.items():
        print(f"   - {key.title()}: {value}")
    
    # 3. Initialize orchestrator
    print("\n🎯 Step 3: Initializing orchestrator...")
    try:
        orchestrator = MetanalystOrchestrator(config)
        print("✅ Orchestrator initialized successfully")
        print(f"   - {len(orchestrator.agent_registry.get_agent_names())} specialized agents loaded")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return
    
    # 4. Create initial state
    print("\n📊 Step 4: Creating initial state...")
    metanalysis_id = "example-mindfulness-anxiety-2024"
    
    initial_state = create_initial_state(
        metanalysis_id=metanalysis_id,
        human_request=human_request,
        pico=pico_framework
    )
    
    print(f"✅ Initial state created")
    print(f"   - Meta-analysis ID: {metanalysis_id}")
    print(f"   - Status: {initial_state['status']}")
    print(f"   - Created: {initial_state['created_at']}")
    
    # 5. Show framework capabilities
    print("\n🛠️  Step 5: Framework capabilities...")
    
    agent_capabilities = orchestrator.agent_registry.get_agent_capabilities()
    print("Available agents:")
    for agent_name, capabilities in agent_capabilities.items():
        print(f"   - {agent_name}: {capabilities['description']}")
    
    execution_order = orchestrator.agent_registry.get_execution_order()
    print(f"\nRecommended execution order: {' → '.join(execution_order)}")
    
    dependencies = orchestrator.agent_registry.get_dependencies()
    print("\nAgent dependencies:")
    for agent, deps in dependencies.items():
        if deps:
            print(f"   - {agent} depends on: {', '.join(deps)}")
        else:
            print(f"   - {agent}: no dependencies")
    
    # 6. Demonstrate configuration flexibility
    print("\n⚙️  Step 6: Configuration flexibility...")
    
    print("Current LLM configuration:")
    print(f"   - Provider: {config.agents.orchestrator.provider}")
    print(f"   - Model: {config.agents.orchestrator.model_name}")
    print(f"   - Temperature: {config.agents.orchestrator.temperature}")
    
    print("\nEasy LLM provider switching:")
    print("   - config.update_llm_provider(LLMProvider.OPENAI)")
    print("   - config.update_llm_provider(LLMProvider.GOOGLE)")
    print("   - config.update_llm_provider(LLMProvider.AZURE)")
    
    # 7. Show next steps for actual execution
    print("\n🚀 Step 7: Ready for execution!")
    print("""
To run the actual meta-analysis, you would call:
    
    final_state = await orchestrator.run_metanalysis(
        human_request=human_request,
        metanalysis_id=metanalysis_id,
        pico=pico_framework
    )

This would:
1. Start with the orchestrator analyzing the request
2. Route to the researcher agent for literature search
3. Process found URLs with the processor agent
4. Retrieve relevant information with the retriever agent
5. Perform statistical analysis with the analyst agent
6. Generate reports with the writer agent
7. Review quality with the reviewer agent
8. Finalize with the editor agent

Each step is checkpointed in PostgreSQL for:
- Human-in-the-loop interventions
- Error recovery and retries
- Process monitoring and debugging
- Time travel and state inspection
""")
    
    print("\n✨ Framework setup complete! Ready for meta-analysis automation.")


def demonstrate_config_switching():
    """Demonstrate easy LLM provider switching."""
    
    print("\n🔄 Demonstrating LLM Provider Switching")
    print("=" * 40)
    
    config = MetanalystConfig()
    
    providers = [
        (LLMProvider.ANTHROPIC, "claude-3-5-sonnet-latest"),
        (LLMProvider.OPENAI, "gpt-4-turbo-preview"),
        (LLMProvider.GOOGLE, "gemini-pro"),
    ]
    
    for provider, model in providers:
        config.update_llm_provider(provider, model)
        print(f"✅ Switched to {provider}: {model}")
        print(f"   - Orchestrator: {config.agents.orchestrator.model_name}")
        print(f"   - Researcher: {config.agents.researcher.model_name}")
        print(f"   - All agents updated consistently")
        print()


if __name__ == "__main__":
    print("🧬 Metanalyst Agent Framework")
    print("Autonomous Meta-Analysis Generation System")
    print("By Nobrega Medtech")
    print()
    
    # Check if we're in a proper environment
    if not os.path.exists(".env") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: No .env file found and no API keys in environment")
        print("   Copy .env.example to .env and add your API keys to run with real services")
        print("   Running in demo mode...")
        print()
    
    # Run the example
    try:
        asyncio.run(run_example_metanalysis())
        demonstrate_config_switching()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your configuration and try again.")