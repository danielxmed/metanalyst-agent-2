#!/usr/bin/env python3
"""
Detailed debug test for processor_agent to understand why it's not calling process_urls
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

def test_processor_detailed():
    """Test processor with detailed state inspection"""
    print("🔍 DETAILED PROCESSOR DEBUG")
    print("="*60)
    
    # Create test state with URLs (complete state structure)
    test_state = {
        "current_iteration": 1,
        "messages": [
            {"role": "user", "content": "Process these research URLs for the meta-analysis"},
            {"role": "assistant", "content": "I'll process the URLs now"}
        ],
        "remaining_steps": 20,
        "user_request": "Process diabetes research URLs",
        "meta_analysis_pico": {
            "population": "Adults with type 2 diabetes mellitus",
            "intervention": "Metformin therapy",
            "comparison": "Placebo or other antidiabetic medications",
            "outcome": "Glycemic control (HbA1c reduction), cardiovascular outcomes"
        },
        "previous_search_queries": [
            "type 2 diabetes metformin treatment efficacy",
            "metformin diabetes mellitus randomized controlled trials"
        ],
        "previous_retrieve_queries": [],
        "urls_to_process": [
            "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "https://pubmed.ncbi.nlm.nih.gov/23456789/"
        ],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft": None,
        "current_draft_iteration": 0,
        "reviewer_feedbacks": [],
        "final_draft": None
    }
    
    print(f"📊 Initial state inspection:")
    print(f"   urls_to_process: {test_state['urls_to_process']}")
    print(f"   processed_urls: {test_state['processed_urls']}")
    print(f"   len(urls_to_process): {len(test_state['urls_to_process'])}")
    
    # Test configuration with very detailed logging
    config = {
        "configurable": {"thread_id": "debug_processor_001"},
        "recursion_limit": 15
    }
    
    print(f"\n🤖 Agent info:")
    print(f"   Agent: {processor_agent}")
    print(f"   Agent type: {type(processor_agent)}")
    
    try:
        print(f"\n🚀 Starting processor agent...")
        print(f"=" * 40)
        
        # Run the processor agent step by step
        final_state = None
        step_count = 0
        
        for step in processor_agent.stream(test_state, config=config):
            step_count += 1
            print(f"\n📝 STEP {step_count}:")
            print(f"   Step keys: {list(step.keys())}")
            
            for key, value in step.items():
                print(f"   {key}: {type(value)}")
                
                # Handle different step types
                if key == "processor" or key == "agent":
                    if hasattr(value, 'get'):
                        messages = value.get('messages', [])
                        print(f"     Messages count: {len(messages)}")
                        if messages:
                            last_msg = messages[-1]
                            print(f"     Last message: {str(last_msg)[:150]}...")
                            
                        urls_to_process = value.get('urls_to_process', [])
                        processed_urls = value.get('processed_urls', [])
                        print(f"     URLs to process: {len(urls_to_process)}")
                        print(f"     Processed URLs: {len(processed_urls)}")
                        
                        final_state = value
                    else:
                        print(f"     Value: {str(value)[:200]}...")
                        
                elif key == "tools":
                    if hasattr(value, 'get'):
                        messages = value.get('messages', [])
                        print(f"     Tool messages count: {len(messages)}")
                        if messages:
                            last_msg = messages[-1]
                            print(f"     Tool last message: {str(last_msg)[:150]}...")
                            
                        urls_to_process = value.get('urls_to_process', [])
                        processed_urls = value.get('processed_urls', [])
                        print(f"     Tool URLs to process: {len(urls_to_process)}")
                        print(f"     Tool Processed URLs: {len(processed_urls)}")
                        
                        final_state = value
                    else:
                        print(f"     Tool value: {str(value)[:200]}...")
                else:
                    print(f"     Content: {str(value)[:200]}...")
                        
            if step_count > 10:  # Safety limit
                print(f"⚠️  Stopping after {step_count} steps to prevent infinite loop")
                break
        
        print(f"\n" + "="*40)
        print(f"✅ PROCESSOR FINISHED after {step_count} steps")
        
        if final_state:
            print(f"\n📊 Final state analysis:")
            print(f"   URLs to process: {len(final_state.get('urls_to_process', []))}")
            print(f"   Processed URLs: {len(final_state.get('processed_urls', []))}")
            print(f"   Messages: {len(final_state.get('messages', []))}")
            
            # Check if any tool calls were made
            messages = final_state.get('messages', [])
            tool_calls = 0
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls += len(msg.tool_calls)
                elif isinstance(msg, dict) and 'tool_calls' in msg:
                    tool_calls += len(msg['tool_calls'])
            
            print(f"   Tool calls made: {tool_calls}")
            
            # Show last few messages
            if messages:
                print(f"\n📜 Last 3 messages:")
                for i, msg in enumerate(messages[-3:]):
                    print(f"   {i+1}. {msg}")
        
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
    print("🔬 DETAILED PROCESSOR AGENT DEBUG")
    print("=" * 80)
    print("This test will run the processor with step-by-step analysis")
    print("=" * 80)
    
    final_state = test_processor_detailed()
    
    if final_state:
        print(f"\n✅ Debug completed!")
    else:
        print(f"\n❌ Debug failed.")
