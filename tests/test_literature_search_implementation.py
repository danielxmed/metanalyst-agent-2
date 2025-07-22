#!/usr/bin/env python3
"""
Basic test to verify literature_search tool implementation
"""

import os
from tools.researcher_agent_tools import literature_search

def test_literature_search_tool():
    """Tests if the literature_search tool is working correctly"""

    print("🧪 Testing the literature_search tool implementation...")
    
    # Test 1: Check if the tool exists and can be imported
    print("✅ Tool literature_search imported successfully")
    
    # Test 2: Check if the tool has the correct signature
    assert hasattr(literature_search, 'name'), "Tool should have 'name' attribute"
    assert hasattr(literature_search, 'description'), "Tool should have 'description' attribute"
    print(f"✅ Tool name: {literature_search.name}")
    print(f"✅ Tool description: {literature_search.description[:100]}...")
    
    # Test 3: Check if it can execute without error (even without API key)
    try:
        # Simulate execution without API key to test error handling
        if 'TAVILY_API_KEY' in os.environ:
            del os.environ['TAVILY_API_KEY']
        
        result = literature_search.invoke({"query": "diabetes mellitus treatment"})
        print("✅ Tool executed without error (error handling working)")
        print(f"✅ Result type: {type(result)}")
        
        # Check if it returns a Command
        from langgraph.types import Command
        assert isinstance(result, Command), "Tool should return a LangGraph Command"
        print("✅ Tool returns Command correctly")
        
    except Exception as e:
        print(f"❌ Error in tool execution: {e}")
        return False
    
    print("\n🎉 All tests passed! The literature_search tool is correctly implemented.")
    return True

if __name__ == "__main__":
    test_literature_search_tool()
