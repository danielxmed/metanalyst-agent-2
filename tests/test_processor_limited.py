#!/usr/bin/env python3
"""
Test processor with limited URLs to avoid API overload
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

def test_processor_with_large_state():
    """Test processor_agent with a large state that mimics real pipeline"""
    print("🧪 PROCESSOR AGENT TEST - LARGE STATE SIMULATION")
    print("="*60)
    
    # Create a state similar to what comes from researcher_agent
    # but with limited URLs to avoid API overload
    test_state = {
        "current_iteration": 1,
        "messages": [
            {"role": "user", "content": "Conduct a meta-analysis on diabetes treatment with metformin"},
            {"role": "assistant", "content": "I'll help you conduct a meta-analysis"},
            # Simulate multiple messages from researcher
        ] + [{"role": "assistant", "content": f"Search iteration {i}"} for i in range(10)],
        "remaining_steps": 20,
        "user_request": "Conduct a meta-analysis on diabetes treatment with metformin efficacy",
        "meta_analysis_pico": {
            "population": "Adults with type 2 diabetes mellitus",
            "intervention": "Metformin therapy",
            "comparison": "Placebo or other antidiabetic medications",
            "outcome": "Glycemic control (HbA1c reduction), cardiovascular outcomes, and mortality"
        },
        "previous_search_queries": [f"search query {i}" for i in range(20)],
        "previous_retrieve_queries": [],
        "urls_to_process": [
            # Mix of accessible and inaccessible URLs (limited to 5)
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Metformin",
            "https://pdfs.semanticscholar.org/5837/d4edcf288c0074a2fc756116fa04bfc64d33.pdf",
            "https://search.ebscohost.com/login.aspx?direct=true&profile=ehost",
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
    print(f"   Messages count: {len(test_state['messages'])}")
    print(f"   Search queries count: {len(test_state['previous_search_queries'])}")
    print(f"   URLs to process: {len(test_state['urls_to_process'])}")
    for i, url in enumerate(test_state['urls_to_process']):
        print(f"   {i+1}. {url[:60]}...")
    
    # Calculate approximate context size
    import json
    state_json = json.dumps(test_state, default=str)
    approx_chars = len(state_json)
    approx_tokens = approx_chars // 4  # rough estimate
    print(f"   Estimated context tokens: ~{approx_tokens:,}")
    
    print("\n🚀 Invoking processor_agent with large state...")
    
    try:
        # Invoke processor agent
        result = processor_agent.invoke(test_state)
        
        print(f"\n📤 OUTPUT STATE:")
        print(f"   URLs to process: {len(result.get('urls_to_process', []))}")
        print(f"   Processed URLs: {len(result.get('processed_urls', []))}")
        print(f"   Messages: {len(result.get('messages', []))}")
        
        # Print last few messages
        messages = result.get('messages', [])
        for i, msg in enumerate(messages[-3:], len(messages)-2):
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', str(msg))
            else:
                content = str(msg)
            print(f"   Message {i}: {content[:100]}{'...' if len(str(content)) > 100 else ''}")
        
        # Check results
        processed_count = len(result.get('processed_urls', []))
        if processed_count > 0:
            print(f"\n✅ SUCCESS: {processed_count} URLs were processed!")
        else:
            print(f"\n⚠️  No URLs processed (might be due to API issues or inaccessible URLs)")
        
        return result
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_type}: {error_msg}")
        
        if "529" in error_msg or "overloaded" in error_msg.lower():
            print("🔍 DIAGNOSIS: API Overload - context too large for processor_agent")
            print("💡 SOLUTION NEEDED: Implement batching or context reduction")
        elif "timeout" in error_msg.lower():
            print("🔍 DIAGNOSIS: Timeout - processing too slow")
        else:
            print("🔍 DIAGNOSIS: Other error")
            import traceback
            traceback.print_exc()
        
        return None

if __name__ == "__main__":
    test_processor_with_large_state()
