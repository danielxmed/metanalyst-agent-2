#!/usr/bin/env python3
"""
Test processor_agent with limited URLs to see if quantity is the issue
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

def test_processor_with_different_url_counts():
    """Test processor with different numbers of URLs"""
    
    # Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Metformin",
        "https://www.nejm.org/doi/full/10.1056/NEJM190402251500802",
        "https://journals.plos.org/plosmedicine/article/info%3Adoi%2F10.1371%2Fjournal.pmed.1001204",
        "https://www.sciencedirect.com/science/article/pii/S0002934397002544"
    ]
    
    # Test with different numbers of URLs
    for url_count in [1, 2, 3, 5]:
        print(f"\n🧪 TESTING WITH {url_count} URLs")
        print("="*50)
        
        # Create test state
        test_state = {
            "current_iteration": 1,
            "messages": [{"role": "user", "content": f"Please process {url_count} URLs"}],
            "remaining_steps": 20,
            "user_request": f"Test processor agent with {url_count} URLs",
            "meta_analysis_pico": {
                "population": "Test population",
                "intervention": "Test intervention", 
                "comparison": "Test comparison",
                "outcome": "Test outcome"
            },
            "previous_search_queries": [],
            "previous_retrieve_queries": [],
            "urls_to_process": test_urls[:url_count],
            "processed_urls": [],
            "retrieved_chunks": [],
            "analysis_results": [],
            "current_draft": None,
            "current_draft_iteration": 0,
            "reviewer_feedbacks": [],
            "final_draft": None
        }
        
        print(f"📥 INPUT: {len(test_state['urls_to_process'])} URLs to process")
        
        try:
            result = processor_agent.invoke(test_state)
            
            processed_count = len(result.get('processed_urls', []))
            remaining_count = len(result.get('urls_to_process', []))
            messages_count = len(result.get('messages', []))
            
            print(f"📤 OUTPUT:")
            print(f"   Processed URLs: {processed_count}")
            print(f"   Remaining URLs: {remaining_count}")
            print(f"   Messages: {messages_count}")
            
            # Check last message
            messages = result.get('messages', [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    content = last_msg.content
                    if content:
                        print(f"   Last message preview: {str(content)[:100]}...")
                    else:
                        print(f"   ❌ Last message is EMPTY!")
                else:
                    print(f"   Last message type: {type(last_msg)}")
            
            # Success check
            if processed_count > 0:
                print(f"   ✅ SUCCESS: {processed_count} URLs processed")
            else:
                print(f"   ❌ FAILURE: No URLs processed")
                
        except Exception as e:
            print(f"   ❌ ERROR: {type(e).__name__}: {str(e)}")

def test_processor_with_large_context():
    """Test processor with many messages (like in pipeline)"""
    print(f"\n🧪 TESTING WITH LARGE MESSAGE CONTEXT")
    print("="*50)
    
    # Simulate the kind of large message context from pipeline
    large_messages = [{"role": "user", "content": "Initial request"}]
    
    # Add many messages to simulate researcher output
    for i in range(20):
        large_messages.append({
            "role": "assistant", 
            "content": f"Research step {i+1}: Found studies on diabetes and metformin treatment..."
        })
    
    test_state = {
        "current_iteration": 1,
        "messages": large_messages,
        "remaining_steps": 20,
        "user_request": "Test with large context",
        "meta_analysis_pico": {
            "population": "Test population",
            "intervention": "Test intervention",
            "comparison": "Test comparison", 
            "outcome": "Test outcome"
        },
        "previous_search_queries": [f"query_{i}" for i in range(20)],
        "previous_retrieve_queries": [],
        "urls_to_process": [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Metformin"
        ],
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
    print(f"   Messages in context: {len(test_state['messages'])}")
    print(f"   Previous queries: {len(test_state['previous_search_queries'])}")
    
    try:
        result = processor_agent.invoke(test_state)
        
        processed_count = len(result.get('processed_urls', []))
        print(f"📤 OUTPUT: {processed_count} URLs processed")
        
        messages = result.get('messages', [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content') and last_msg.content:
                print(f"   ✅ Last message has content")
            else:
                print(f"   ❌ Last message is empty!")
        
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    print("🔬 PROCESSOR AGENT URL QUANTITY TEST")
    print("="*60)
    
    test_processor_with_different_url_counts()
    test_processor_with_large_context()
    
    print("\n✅ Tests completed!")
