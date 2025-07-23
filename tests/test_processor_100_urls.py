#!/usr/bin/env python3
"""
Test processor_agent with exactly 100 URLs to reproduce pipeline issue
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

# Import agents
spec_processor = importlib.util.spec_from_file_location("processor", os.path.join(project_root, "agents", "processor.py"))
processor_module = importlib.util.module_from_spec(spec_processor)
spec_processor.loader.exec_module(processor_module)

processor_agent = processor_module.processor_agent

def test_processor_with_100_urls():
    """Test processor with exactly 100 URLs like in the pipeline"""
    print("🧪 TESTING WITH 100 URLs (PIPELINE REPRODUCTION)")
    print("="*60)
    
    # Generate 100 test URLs (similar to what researcher generates)
    test_urls = []
    base_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Metformin",
        "https://www.nejm.org/doi/full/10.1056/NEJM190402251500802",
        "https://journals.plos.org/plosmedicine/article/info%3Adoi%2F10.1371%2Fjournal.pmed.1001204",
        "https://www.sciencedirect.com/science/article/pii/S0002934397002544"
    ]
    
    # Create 100 URLs by repeating and adding suffixes
    for i in range(100):
        base_url = base_urls[i % len(base_urls)]
        # Add query parameters to make URLs unique
        test_url = f"{base_url}?test_param={i}"
        test_urls.append(test_url)
    
    # Create test state similar to pipeline
    test_state = {
        "current_iteration": 1,
        "messages": [
            {"role": "user", "content": "I need a meta-analysis on diabetes treatment with metformin"},
            {"role": "assistant", "content": "I'll help you conduct a meta-analysis on diabetes treatment with metformin."}
        ],
        "remaining_steps": 20,
        "user_request": "Conduct a meta-analysis on diabetes treatment with metformin efficacy",
        "meta_analysis_pico": {
            "population": "Adults with type 2 diabetes mellitus",
            "intervention": "Metformin therapy",
            "comparison": "Placebo or other antidiabetic medications",
            "outcome": "Glycemic control (HbA1c reduction), cardiovascular outcomes, and mortality"
        },
        "previous_search_queries": [f"diabetes query {i}" for i in range(20)],
        "previous_retrieve_queries": [],
        "urls_to_process": test_urls,
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft": None,
        "current_draft_iteration": 0,
        "reviewer_feedbacks": [],
        "final_draft": None
    }
    
    print(f"📥 INPUT:")
    print(f"   URLs to process: {len(test_state['urls_to_process'])}")
    print(f"   Sample URLs:")
    for i, url in enumerate(test_state['urls_to_process'][:3]):
        print(f"     {i+1}. {url}")
    print(f"   Messages in context: {len(test_state['messages'])}")
    print(f"   Previous queries: {len(test_state['previous_search_queries'])}")
    
    try:
        print(f"\n🚀 Invoking processor_agent with 100 URLs...")
        result = processor_agent.invoke(test_state)
        
        processed_count = len(result.get('processed_urls', []))
        remaining_count = len(result.get('urls_to_process', []))
        messages_count = len(result.get('messages', []))
        
        print(f"\n📤 OUTPUT:")
        print(f"   Processed URLs: {processed_count}")
        print(f"   Remaining URLs: {remaining_count}")
        print(f"   Total messages: {messages_count}")
        
        # Check last message
        messages = result.get('messages', [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                content = last_msg.content
                if content:
                    print(f"   Last message preview: {str(content)[:200]}...")
                    print(f"   ✅ Last message has content")
                else:
                    print(f"   ❌ Last message is EMPTY!")
                    print(f"   Last message type: {type(last_msg)}")
                    print(f"   Last message attributes: {dir(last_msg)}")
            else:
                print(f"   Last message type: {type(last_msg)}")
        else:
            print(f"   ❌ No messages returned!")
        
        # Success check
        if processed_count > 0:
            print(f"\n   ✅ SUCCESS: {processed_count} URLs processed out of 100")
        else:
            print(f"\n   ❌ FAILURE: No URLs processed out of 100")
            
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔬 PROCESSOR AGENT 100 URLs TEST")
    print("="*60)
    print("This test reproduces the exact conditions from the complete pipeline")
    print("where processor_agent fails to process 100 URLs.\n")
    
    test_processor_with_100_urls()
    
    print("\n✅ Test completed!")
