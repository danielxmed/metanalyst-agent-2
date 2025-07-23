#!/usr/bin/env python3
"""
Isolated test for processor_agent to debug why it's not calling process_urls tool
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

# Import processor agent directly
spec_processor = importlib.util.spec_from_file_location("processor", os.path.join(project_root, "agents", "processor.py"))
processor_module = importlib.util.module_from_spec(spec_processor)
spec_processor.loader.exec_module(processor_module)

processor_agent = processor_module.processor_agent

def test_processor_isolated():
    """Test processor_agent in isolation"""
    print("🧪 ISOLATED PROCESSOR AGENT TEST")
    print("="*50)
    
    # Create test state with URLs to process (complete state)
    test_state = {
        "current_iteration": 1,
        "messages": [{"role": "user", "content": "Please process the URLs"}],
        "remaining_steps": 20,
        "user_request": "Test processor agent with URLs",
        "meta_analysis_pico": {
            "population": "Test population",
            "intervention": "Test intervention",
            "comparison": "Test comparison",
            "outcome": "Test outcome"
        },
        "previous_search_queries": [],
        "previous_retrieve_queries": [],
        "urls_to_process": [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Metformin",
            "https://www.nejm.org/doi/full/10.1056/NEJM190402251500802"
        ],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft": None,
        "current_draft_iteration": 0,
        "reviewer_feedbacks": [],
        "final_draft": None
    }
    
    print(f"📥 INPUT STATE:")
    print(f"   URLs to process: {len(test_state['urls_to_process'])}")
    for i, url in enumerate(test_state['urls_to_process']):
        print(f"   {i+1}. {url}")
    print(f"   Processed URLs: {len(test_state['processed_urls'])}")
    
    print("\n🚀 Invoking processor_agent...")
    
    try:
        # Invoke processor agent
        result = processor_agent.invoke(test_state)
        
        print(f"\n📤 OUTPUT STATE:")
        print(f"   URLs to process: {len(result.get('urls_to_process', []))}")
        print(f"   Processed URLs: {len(result.get('processed_urls', []))}")
        print(f"   Messages: {len(result.get('messages', []))}")
        
        # Print all messages
        messages = result.get('messages', [])
        for i, msg in enumerate(messages):
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', str(msg))
            else:
                content = str(msg)
            print(f"   Message {i+1}: {content[:200]}{'...' if len(str(content)) > 200 else ''}")
        
        # Check if URLs were moved to processed
        if len(result.get('processed_urls', [])) > 0:
            print("\n✅ SUCCESS: URLs were processed!")
        else:
            print("\n❌ FAILURE: No URLs were processed.")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_processor_isolated()
