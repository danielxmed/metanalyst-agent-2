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

# Test to verify if the state is being updated
def test_state_update():
    print("🧪 Testing if the state is being updated by tools...")
    
    # Initial state
    initial_state = {
        "messages": [
            {
                "role": "user", 
                "content": "Test diabetes search"
            }
        ],
        "meta_analysis_pico": {
            "population": "people with type 2 diabetes",
            "intervention": "metformin", 
            "comparison": "placebo",
            "outcome": "glycemic control"
        },
        "user_request": "Test diabetes search",
        "previous_search_queries": [],
        "urls_to_process": []  # Initially empty field
    }
    
    print(f"📊 Initial state:")
    print(f"   - URLs to process: {len(initial_state.get('urls_to_process', []))}")
    print(f"   - Previous queries: {len(initial_state.get('previous_search_queries', []))}")
    print(f"   - Messages: {len(initial_state.get('messages', []))}")
    
    # Execute the agent
    result = researcher_agent.invoke(initial_state)
    
    print(f"\n📊 State after execution:")
    print(f"   - URLs to process: {len(result.get('urls_to_process', []))}")
    print(f"   - Previous queries: {len(result.get('previous_search_queries', []))}")
    print(f"   - Messages: {len(result.get('messages', []))}")
    
    # Check if the state was updated
    urls_updated = len(result.get('urls_to_process', [])) > len(initial_state.get('urls_to_process', []))
    queries_updated = len(result.get('previous_search_queries', [])) > len(initial_state.get('previous_search_queries', []))
    messages_updated = len(result.get('messages', [])) > len(initial_state.get('messages', []))
    
    print(f"\n✅ Verifications:")
    print(f"   - URLs were added: {'✅ YES' if urls_updated else '❌ NO'}")
    print(f"   - Queries were added: {'✅ YES' if queries_updated else '❌ NO'}")
    print(f"   - Messages were added: {'✅ YES' if messages_updated else '❌ NO'}")
    
    if urls_updated and queries_updated and messages_updated:
        print(f"\n🎉 SUCCESS: State was correctly updated by the tools!")
        print(f"   - {len(result.get('urls_to_process', []))} URLs found")
        print(f"   - {len(result.get('previous_search_queries', []))} queries executed")
    else:
        print(f"\n❌ FAILURE: State was not updated as expected")
        
    return result

if __name__ == "__main__":
    test_state_update()
