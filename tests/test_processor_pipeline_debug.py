#!/usr/bin/env python3
"""
Debug test to understand why processor_agent fails in the complete pipeline
"""

import os
import sys
import importlib.util
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables
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

researcher_agent = researcher_module.researcher_agent
processor_agent = processor_module.processor_agent

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

def debug_pipeline_flow():
    """Debug the pipeline flow step by step"""
    print("🔬 PIPELINE FLOW DEBUG")
    print("="*60)
    
    # Step 1: Create initial state
    print("📝 Step 1: Creating initial state...")
    initial_state = create_simulated_state()
    print(f"   URLs to process: {len(initial_state['urls_to_process'])}")
    print(f"   Messages: {len(initial_state['messages'])}")
    
    # Step 2: Run researcher agent
    print("\n🔍 Step 2: Running researcher agent...")
    try:
        researcher_result = researcher_agent.invoke(initial_state)
        print(f"   URLs to process after researcher: {len(researcher_result.get('urls_to_process', []))}")
        print(f"   Messages after researcher: {len(researcher_result.get('messages', []))}")
        
        # Show last few URLs
        urls = researcher_result.get('urls_to_process', [])
        if urls:
            print(f"   Sample URLs:")
            for i, url in enumerate(urls[:3]):
                print(f"     {i+1}. {url}")
        
        # Step 3: Run processor agent on the result
        print(f"\n⚙️ Step 3: Running processor agent on researcher result...")
        print(f"   Input URLs to process: {len(researcher_result.get('urls_to_process', []))}")
        
        # Debug: Let's inspect the exact state being passed
        print(f"   State keys: {list(researcher_result.keys())}")
        print(f"   State type: {type(researcher_result)}")
        
        processor_result = processor_agent.invoke(researcher_result)
        print(f"   URLs to process after processor: {len(processor_result.get('urls_to_process', []))}")
        print(f"   Processed URLs after processor: {len(processor_result.get('processed_urls', []))}")
        print(f"   Messages after processor: {len(processor_result.get('messages', []))}")
        
        # Check the last message from processor
        messages = processor_result.get('messages', [])
        if messages:
            last_msg = messages[-1]
            print(f"   Last message type: {type(last_msg)}")
            print(f"   Last message content preview: {str(last_msg)[:200]}...")
        else:
            print("   ❌ No messages from processor!")
        
        return processor_result
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_pipeline_flow()
