#!/usr/bin/env python3
"""
Debug test for processor_agent to understand state access
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

# Import state and prompt
from state.state import MetaAnalysisState
from prompts.processor_prompt import processor_prompt

def debug_processor_prompt():
    """Test the processor prompt directly"""
    print("🔍 PROCESSOR PROMPT DEBUG")
    print("="*50)
    
    # Test state structure
    test_state = {
        "current_iteration": 1,
        "messages": [],
        "remaining_steps": 20,
        "user_request": "Test",
        "meta_analysis_pico": None,
        "previous_search_queries": [],
        "previous_retrieve_queries": [],
        "urls_to_process": [
            "https://www.nejm.org/doi/full/10.1056/NEJM190402251500802",
            "https://www.nejm.org/doi/full/10.1056/NEJM189609031351004"
        ],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft": None,
        "current_draft_iteration": 0,
        "reviewer_feedbacks": [],
        "final_draft": None
    }
    
    print("🧪 Test State:")
    print(f"   urls_to_process: {test_state['urls_to_process']}")
    print(f"   processed_urls: {test_state['processed_urls']}")
    
    print("\n📝 Processor Prompt:")
    print(processor_prompt)
    
    # Test if the state matches what the schema expects
    print("\n🔍 State Schema Analysis:")
    print("Expected keys from MetaAnalysisState schema:")
    import inspect
    annotations = MetaAnalysisState.__annotations__
    for key, value in annotations.items():
        state_value = test_state.get(key, "MISSING")
        print(f"   {key}: {value} -> {type(state_value)} = {state_value}")

if __name__ == "__main__":
    debug_processor_prompt()
