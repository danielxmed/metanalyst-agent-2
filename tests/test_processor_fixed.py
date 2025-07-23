#!/usr/bin/env python3
"""
Test to verify the processor_agent loop fix
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

# Import processor agent directly
spec_processor = importlib.util.spec_from_file_location("processor", os.path.join(project_root, "agents", "processor.py"))
processor_module = importlib.util.module_from_spec(spec_processor)
spec_processor.loader.exec_module(processor_module)

# Get the processor agent
processor_agent = processor_module.processor_agent

def test_processor_fixed():
    """Test processor with URLs to verify no loop"""
    print("🧪 TESTING PROCESSOR AGENT - LOOP FIX")
    print("="*60)
    
    # Create test state with a few URLs
    test_state = {
        "current_iteration": 1,
        "messages": [
            {"role": "user", "content": "Process these URLs"},
            {"role": "assistant", "content": "I'll process the URLs now"}
        ],
        "remaining_steps": 20,
        "user_request": "Process diabetes research URLs",
        "urls_to_process": [
            "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "https://pubmed.ncbi.nlm.nih.gov/23456789/",
            "https://pubmed.ncbi.nlm.nih.gov/34567890/"
        ],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": []
    }
    
    print(f"📊 Initial state:")
    print(f"   URLs to process: {len(test_state['urls_to_process'])}")
    print(f"   Processed URLs: {len(test_state['processed_urls'])}")
    
    # Test configuration
    config = {
        "configurable": {"thread_id": "test_processor_fixed_001"},
        "recursion_limit": 10  # Lower limit to catch loops early
    }
    
    try:
        print(f"\n🚀 Running processor agent...")
        
        # Run the processor agent
        final_state = processor_agent.invoke(test_state, config=config)
        
        print(f"\n✅ PROCESSOR COMPLETED!")
        print(f"📊 Final state:")
        print(f"   URLs to process: {len(final_state.get('urls_to_process', []))}")
        print(f"   Processed URLs: {len(final_state.get('processed_urls', []))}")
        
        # Verify the fix worked
        urls_remaining = len(final_state.get('urls_to_process', []))
        urls_processed = len(final_state.get('processed_urls', []))
        
        if urls_remaining == 0:
            print(f"✅ SUCCESS: All URLs processed, no loop detected!")
        else:
            print(f"⚠️  WARNING: {urls_remaining} URLs still remaining")
        
        print(f"✅ Total URLs processed: {urls_processed}")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ ERROR during processor execution:")
        print(f"   {type(e).__name__}: {str(e)}")
        
        # Print traceback for debugging
        import traceback
        print(f"\n🔍 Full traceback:")
        traceback.print_exc()
        
        return None

if __name__ == "__main__":
    print("🔬 TESTING PROCESSOR AGENT - VERIFYING LOOP FIX")
    print("=" * 80)
    print("This test will:")
    print("1. 📋 Create state with 3 URLs")
    print("2. ⚙️  Run processor_agent")
    print("3. 📊 Verify no loop and all URLs processed")
    print("=" * 80)
    
    input("Press Enter to start the test...")
    
    final_state = test_processor_fixed()
    
    if final_state:
        print(f"\n✅ Test completed successfully!")
    else:
        print(f"\n❌ Test failed.")
