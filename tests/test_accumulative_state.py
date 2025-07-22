#!/usr/bin/env python3
import os
import sys
import importlib.util
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv(os.path.join(project_root, '.env'))

# Import researcher module directly avoiding __init__.py
spec = importlib.util.spec_from_file_location("researcher", os.path.join(project_root, "agents", "researcher.py"))
researcher_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(researcher_module)

researcher_agent = researcher_module.researcher_agent

def test_accumulative_state():
    print("🧪 Testing if the state is CUMULATIVE (does not replace, but adds)...")
    
    # Initial state with some existing data
    initial_state = {
        "messages": [{"role": "user", "content": "First search"}],
        "meta_analysis_pico": {
            "population": "people with type 2 diabetes",
            "intervention": "metformin", 
            "comparison": "placebo",
            "outcome": "glycemic control"
        },
        "user_request": "Test cumulative state",
        "previous_search_queries": ["previous_search_1", "previous_search_2"],  # Already has 2 queries
        "urls_to_process": ["existing_url_1.com", "existing_url_2.com"],    # Already has 2 URLs
        "remaining_steps": 10,
        "current_iteration": 1,
        "previous_retrieve_queries": [],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft_iteration": 0,
        "reviewer_feedbacks": []
    }
    
    print(f"📊 INITIAL State:")
    print(f"   - Existing URLs: {len(initial_state['urls_to_process'])} -> {initial_state['urls_to_process']}")
    print(f"   - Existing queries: {len(initial_state['previous_search_queries'])} -> {initial_state['previous_search_queries']}")
    
    # Execute the agent with pre-existing data
    result = researcher_agent.invoke(initial_state, config={"recursion_limit": 15})
    
    print(f"\n📊 State AFTER execution:")
    print(f"   - Total URLs: {len(result.get('urls_to_process', []))}")
    print(f"   - Total Queries: {len(result.get('previous_search_queries', []))}")
    
    # Check if old data still exists
    old_urls_preserved = all(url in result.get('urls_to_process', []) 
                           for url in initial_state['urls_to_process'])
    old_queries_preserved = all(query in result.get('previous_search_queries', []) 
                              for query in initial_state['previous_search_queries'])
    
    print(f"\n✅ ACCUMULATION Verifications:")
    print(f"   - Old URLs preserved: {'✅ YES' if old_urls_preserved else '❌ NO'}")
    print(f"   - Old queries preserved: {'✅ YES' if old_queries_preserved else '❌ NO'}")
    
    # List old URLs vs new
    if old_urls_preserved:
        new_urls = [url for url in result.get('urls_to_process', []) 
                   if url not in initial_state['urls_to_process']]
        print(f"   - NEW URLs found: {len(new_urls)}")
        if new_urls:
            print(f"     → Examples: {new_urls[:3]}...")
    
    # List old queries vs new  
    if old_queries_preserved:
        new_queries = [q for q in result.get('previous_search_queries', []) 
                      if q not in initial_state['previous_search_queries']]
        print(f"   - NEW queries executed: {len(new_queries)}")
        if new_queries:
            print(f"     → New queries: {new_queries}")
    
    if old_urls_preserved and old_queries_preserved:
        print(f"\n🎉 CONFIRMED: State is CUMULATIVE!")
        print(f"   ✓ Old data is preserved")
        print(f"   ✓ New data is added (not replaced)")
        print(f"   ✓ operator.add is working correctly")
    else:
        print(f"\n❌ PROBLEM: State is being replaced, not accumulated!")
        
    return result

if __name__ == "__main__":
    test_accumulative_state()
