#!/usr/bin/env python3
"""
Complete pipeline test: researcher_agent -> processor_agent with process_urls tool
"""

import os
import sys
import importlib.util
from dotenv import load_dotenv
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv(os.path.join(project_root, '.env'))

# Import state
from state.state import MetaAnalysisState

# Import agents directly
spec_researcher = importlib.util.spec_from_file_location("researcher", os.path.join(project_root, "agents", "researcher.py"))
researcher_module = importlib.util.module_from_spec(spec_researcher)
spec_researcher.loader.exec_module(researcher_module)

spec_processor = importlib.util.spec_from_file_location("processor", os.path.join(project_root, "agents", "processor.py"))
processor_module = importlib.util.module_from_spec(spec_processor)
spec_processor.loader.exec_module(processor_module)

# Get the agents
researcher_agent = researcher_module.researcher_agent
processor_agent = processor_module.processor_agent

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def print_state_summary(state, stage_name):
    """Print a summary of the current state"""
    print(f"\n" + "="*60)
    print(f"📊 STATE SUMMARY - {stage_name}")
    print(f"="*60)
    
    print(f"🔤 User Request: {state.get('user_request', 'Not set')}")
    print(f"🔬 Current Iteration: {state.get('current_iteration', 0)}")
    print(f"📝 Messages Count: {len(state.get('messages', []))}")
    
    # PICO elements
    pico = state.get('meta_analysis_pico', {})
    if pico:
        print(f"🎯 PICO Elements:")
        print(f"   - Population: {pico.get('population', 'Not set')}")
        print(f"   - Intervention: {pico.get('intervention', 'Not set')}")
        print(f"   - Comparison: {pico.get('comparison', 'Not set')}")
        print(f"   - Outcome: {pico.get('outcome', 'Not set')}")
    
    # Search and URL data
    print(f"🔍 Previous Search Queries: {len(state.get('previous_search_queries', []))}")
    if state.get('previous_search_queries'):
        for i, query in enumerate(state.get('previous_search_queries', [])[:3]):
            print(f"   {i+1}. {query}")
        if len(state.get('previous_search_queries', [])) > 3:
            print(f"   ... and {len(state.get('previous_search_queries', [])) - 3} more")
    
    print(f"🌐 URLs to Process: {len(state.get('urls_to_process', []))}")
    if state.get('urls_to_process'):
        for i, url in enumerate(state.get('urls_to_process', [])[:3]):
            print(f"   {i+1}. {url}")
        if len(state.get('urls_to_process', [])) > 3:
            print(f"   ... and {len(state.get('urls_to_process', [])) - 3} more")
    
    print(f"✅ Processed URLs: {len(state.get('processed_urls', []))}")
    if state.get('processed_urls'):
        for i, url in enumerate(state.get('processed_urls', [])[:3]):
            print(f"   {i+1}. {url}")
        if len(state.get('processed_urls', [])) > 3:
            print(f"   ... and {len(state.get('processed_urls', [])) - 3} more")
    
    print(f"📚 Retrieved Chunks: {len(state.get('retrieved_chunks', []))}")
    print(f"📊 Analysis Results: {len(state.get('analysis_results', []))}")
    
    # Print last message content if available
    messages = state.get('messages', [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'content'):
            content = last_msg.content
        elif isinstance(last_msg, dict):
            content = last_msg.get('content', str(last_msg))
        else:
            content = str(last_msg)
        
        print(f"💬 Last Message: {content[:100]}{'...' if len(str(content)) > 100 else ''}")

def create_simulated_state():
    """Create a simulated initial state for testing"""
    return {
        "current_iteration": 1,
        "messages": [
            {"role": "user", "content": "I need a meta-analysis on diabetes treatment with metformin"},
            {"role": "assistant", "content": "I'll help you conduct a meta-analysis on diabetes treatment with metformin. Let me start by searching for relevant studies."}
        ],
        "remaining_steps": 20,
        "user_request": "Conduct a meta-analysis on diabetes treatment with metformin efficacy",
        "meta_analysis_pico": {
            "population": "Adults with type 2 diabetes mellitus",
            "intervention": "Metformin therapy",
            "comparison": "Placebo or other antidiabetic medications",
            "outcome": "Glycemic control (HbA1c reduction), cardiovascular outcomes, and mortality"
        },
        "previous_search_queries": [
            "type 2 diabetes metformin treatment efficacy",
            "metformin diabetes mellitus randomized controlled trials",
            "diabetes metformin HbA1c glycemic control"
        ],
        "previous_retrieve_queries": [],
        "urls_to_process": [],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft": None,
        "current_draft_iteration": 0,
        "reviewer_feedbacks": [],
        "final_draft": None
    }

def simulated_state_node(state):
    """Node that returns the simulated state - acts as our starting point"""
    print_state_summary(state, "INITIAL SIMULATED STATE")
    return state

def print_researcher_output(state):
    """Print researcher agent output"""
    print_state_summary(state, "AFTER RESEARCHER AGENT")
    return state

def print_processor_output(state):
    """Print processor agent output"""
    print_state_summary(state, "AFTER PROCESSOR AGENT")
    return state

def create_test_graph():
    """Create the test graph with the specified flow"""
    
    # Create the state graph
    builder = StateGraph(MetaAnalysisState)
    
    # Add nodes
    builder.add_node("simulated_state", simulated_state_node)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("print_researcher", print_researcher_output)
    builder.add_node("processor", processor_agent)
    builder.add_node("print_processor", print_processor_output)
    
    # Add edges: start -> simulated_state -> researcher -> processor -> end
    builder.add_edge(START, "simulated_state")
    builder.add_edge("simulated_state", "researcher")
    builder.add_edge("researcher", "print_researcher")
    builder.add_edge("print_researcher", "processor")
    builder.add_edge("processor", "print_processor")
    builder.add_edge("print_processor", END)
    
    # Create memory checkpointer
    memory = MemorySaver()
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory)
    
    return graph

def run_complete_pipeline_test():
    """Run the complete pipeline test"""
    print("🧪 COMPLETE PIPELINE TEST: researcher_agent -> processor_agent")
    print("="*80)
    
    # Create initial simulated state
    initial_state = create_simulated_state()
    
    # Create the test graph
    graph = create_test_graph()
    
    # Configuration for the run
    config = {
        "configurable": {"thread_id": "test_pipeline_001"},
        "recursion_limit": 25
    }
    
    try:
        print("🚀 Starting pipeline execution...")
        
        # Run the graph
        final_state = graph.invoke(initial_state, config=config)
        
        print("\n" + "🎉" * 20)
        print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        # Final summary
        print_state_summary(final_state, "FINAL RESULTS")
        
        # Check if URLs were processed
        urls_processed = len(final_state.get('processed_urls', []))
        urls_initial = len(initial_state.get('urls_to_process', []))
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   - Initial URLs to process: {urls_initial}")
        print(f"   - URLs successfully processed: {urls_processed}")
        print(f"   - Processing success rate: {(urls_processed/urls_initial*100) if urls_initial > 0 else 0:.1f}%")
        
        # Check data directories
        data_dir = Path(project_root) / "data"
        json_dir = data_dir / "full_json_referenced"
        chunks_dir = data_dir / "chunks"
        
        if json_dir.exists():
            json_files = list(json_dir.glob("*.json"))
            print(f"   - Referenced JSON files created: {len(json_files)}")
        
        if chunks_dir.exists():
            chunk_files = list(chunks_dir.glob("*.json"))
            print(f"   - Chunk files created: {len(chunk_files)}")
        
        vectorstore_path = data_dir / "publications_vectorstore"
        if vectorstore_path.exists():
            print(f"   - Vectorstore created: ✅ YES")
        else:
            print(f"   - Vectorstore created: ❌ NO")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ ERROR during pipeline execution:")
        print(f"   {type(e).__name__}: {str(e)}")
        
        # Print traceback for debugging
        import traceback
        print(f"\n🔍 Full traceback:")
        traceback.print_exc()
        
        return None

if __name__ == "__main__":
    print("🔬 TESTING COMPLETE METANALYST PIPELINE")
    print("=" * 80)
    print("This test will run:")
    print("1. 📋 Create simulated state with NEJM URLs")
    print("2. 🔍 Run researcher_agent (might add more URLs)")
    print("3. ⚙️  Run processor_agent (process_urls tool)")
    print("4. 📊 Show state changes at each step")
    print("=" * 80)
    
    input("Press Enter to start the test...")
    
    final_state = run_complete_pipeline_test()
    
    if final_state:
        print(f"\n✅ Test completed! Check the /data directory for generated files.")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")
