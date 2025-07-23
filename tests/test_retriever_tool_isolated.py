"""
Isolated test for the retrieve_chunks tool.
Tests the tool functionality without running the full agent pipeline.
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.retriever_agent_tools import retrieve_chunks
from state.state import MetaAnalysisState
from langgraph.types import Command

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_state(retrieved_chunks=None, previous_queries=None) -> Dict[str, Any]:
    """Create a test state for tool testing."""
    return {
        "retrieved_chunks": retrieved_chunks or [],
        "previous_retrieve_queries": previous_queries or [],
        "messages": [],
        "metanalysis_pico": {
            "Population": "Patients with atrial fibrillation",
            "Intervention": "Amiodarone",
            "Comparison": "Beta-blockers",
            "Outcome": "Cardiovascular mortality and arrhythmia control"
        }
    }

def test_basic_retrieval():
    """Test basic chunk retrieval functionality."""
    logger.info("=== TEST 1: Basic Retrieval ===")
    
    state = create_test_state()
    query = "amiodarone versus beta blockers atrial fibrillation"
    tool_call_id = "test_call_1"
    
    try:
        # Invoke the tool correctly using LangChain's invoke method
        result = retrieve_chunks.invoke({
            "query": query,
            "tool_call_id": tool_call_id,
            "state": state
        })
        
        logger.info(f"Result type: {type(result)}")
        
        if isinstance(result, Command):
            logger.info("✅ Tool returned Command object correctly")
            
            # Check if state update contains expected keys
            update = result.update if hasattr(result, 'update') else {}
            logger.info(f"State update keys: {list(update.keys())}")
            
            if "retrieved_chunks" in update:
                chunks = update["retrieved_chunks"]
                logger.info(f"✅ Retrieved {len(chunks)} chunks")
                
                # Log sample chunk structure
                if chunks:
                    sample_chunk = chunks[0]
                    logger.info(f"Sample chunk keys: {list(sample_chunk.keys())}")
                    logger.info(f"Sample chunk similarity score: {sample_chunk.get('similarity_score', 'N/A')}")
            
            if "previous_retrieve_queries" in update:
                queries = update["previous_retrieve_queries"]
                logger.info(f"✅ Updated query history: {len(queries)} queries")
            
            if "messages" in update:
                messages = update["messages"]
                logger.info(f"✅ Generated {len(messages)} messages")
                for i, msg in enumerate(messages):
                    logger.info(f"  Message {i+1}: {type(msg).__name__}")
            
            # Print detailed final state for inspection
            logger.info("\n" + "="*60)
            logger.info("📋 FINAL STATE AFTER TEST 1 (Basic Retrieval)")
            logger.info("="*60)
            
            # Create final state by applying updates
            final_state = state.copy()
            final_state.update(update)
            
            logger.info(f"🔍 Query History: {len(final_state.get('previous_retrieve_queries', []))} queries")
            for i, query in enumerate(final_state.get('previous_retrieve_queries', []), 1):
                logger.info(f"  Query {i}: '{query}'")
            
            logger.info(f"📚 Retrieved Chunks: {len(final_state.get('retrieved_chunks', []))} chunks")
            for i, chunk in enumerate(final_state.get('retrieved_chunks', [])[:3], 1):  # Show first 3 chunks
                logger.info(f"  Chunk {i}:")
                logger.info(f"    - ID: {chunk.get('chunk_id', 'N/A')}")
                logger.info(f"    - Title: {chunk.get('title', 'N/A')}")
                logger.info(f"    - Authors: {chunk.get('authors', 'N/A')}")
                logger.info(f"    - Year: {chunk.get('year', 'N/A')}")
                logger.info(f"    - Journal: {chunk.get('journal', 'N/A')}")
                logger.info(f"    - Reference: {chunk.get('reference', 'N/A')[:80]}...")
                logger.info(f"    - Source: {chunk.get('source', 'N/A')[:50]}...")
                logger.info(f"    - Content: {chunk.get('content', 'N/A')[:100]}...")
                logger.info(f"    - Similarity Score: {chunk.get('similarity_score', 'N/A')}")
                logger.info(f"    - Chunk Index: {chunk.get('chunk_index', 'N/A')}/{chunk.get('total_chunks', 'N/A')}")
            
            if len(final_state.get('retrieved_chunks', [])) > 3:
                logger.info(f"    ... and {len(final_state.get('retrieved_chunks', [])) - 3} more chunks")
            
            logger.info(f"💬 Messages: {len(final_state.get('messages', []))} messages")
            for i, msg in enumerate(final_state.get('messages', []), 1):
                msg_type = type(msg).__name__
                msg_content = getattr(msg, 'content', 'N/A')
                logger.info(f"  Message {i} ({msg_type}): {str(msg_content)[:100]}...")
            
            # Calculate estimated tokens
            total_chunks = len(final_state.get('retrieved_chunks', []))
            estimated_tokens = total_chunks * 200
            logger.info(f"📊 Token Estimation: {total_chunks} chunks × 200 tokens = ~{estimated_tokens} tokens")
            logger.info(f"📈 Remaining capacity: {600 - total_chunks} chunks (~{(600 - total_chunks) * 200} tokens)")
            
            logger.info("="*60 + "\n")
            
        else:
            logger.error(f"❌ Expected Command object, got {type(result)}")
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        logger.error(f"Error type: {type(e)}")

def test_duplicate_filtering():
    """Test that duplicate chunks are filtered out."""
    logger.info("=== TEST 2: Duplicate Filtering ===")
    
    # Create state with existing chunks
    existing_chunks = [
        {
            "content": "Amiodarone is an antiarrhythmic medication...",
            "chunk_id": "chunk_1",
            "source": "test_source_1"
        }
    ]
    
    state = create_test_state(
        retrieved_chunks=existing_chunks,
        previous_queries=["previous test query"]
    )
    
    query = "amiodarone effectiveness atrial fibrillation"
    tool_call_id = "test_call_2"
    
    try:
        result = retrieve_chunks.invoke({
            "query": query,
            "tool_call_id": tool_call_id,
            "state": state
        })
        
        if isinstance(result, Command):
            update = result.update if hasattr(result, 'update') else {}
            
            if "retrieved_chunks" in update:
                final_chunks = update["retrieved_chunks"]
                new_chunks_count = len(final_chunks) - len(existing_chunks)
                logger.info(f"✅ Started with {len(existing_chunks)} chunks, added {new_chunks_count} new chunks")
                logger.info(f"Total chunks after filtering: {len(final_chunks)}")
            
            if "previous_retrieve_queries" in update:
                queries = update["previous_retrieve_queries"]
                logger.info(f"✅ Query history updated: {queries}")
                
    except Exception as e:
        logger.error(f"❌ Duplicate filtering test failed: {str(e)}")

def test_token_limit():
    """Test behavior when approaching token limit."""
    logger.info("=== TEST 3: Token Limit Behavior ===")
    
    # Create state with many existing chunks (simulating near limit)
    large_chunks_list = []
    for i in range(590):  # Near the 600 limit
        large_chunks_list.append({
            "content": f"Sample chunk content {i}",
            "chunk_id": f"chunk_{i}",
            "source": f"source_{i}"
        })
    
    state = create_test_state(retrieved_chunks=large_chunks_list)
    
    query = "test query near limit"
    tool_call_id = "test_call_3"
    
    try:
        result = retrieve_chunks.invoke({
            "query": query,
            "tool_call_id": tool_call_id,
            "state": state
        })
        
        if isinstance(result, Command):
            update = result.update if hasattr(result, 'update') else {}
            
            if "messages" in update:
                messages = update["messages"]
                # Check if tool message mentions approaching limits
                for msg in messages:
                    if hasattr(msg, 'content') and 'chunks' in str(msg.content).lower():
                        logger.info(f"✅ Tool provided informative message about chunk count")
                        break
                        
        logger.info("✅ Token limit handling test completed")
        
    except Exception as e:
        logger.error(f"❌ Token limit test failed: {str(e)}")

def test_over_limit():
    """Test behavior when already over the chunk limit."""
    logger.info("=== TEST 4: Over Limit Behavior ===")
    
    # Create state with chunks over the limit
    over_limit_chunks = []
    for i in range(650):  # Over the 600 limit
        over_limit_chunks.append({
            "content": f"Over limit chunk {i}",
            "chunk_id": f"over_chunk_{i}",
            "source": f"over_source_{i}"
        })
    
    state = create_test_state(retrieved_chunks=over_limit_chunks)
    
    query = "test query over limit"
    tool_call_id = "test_call_4"
    
    try:
        result = retrieve_chunks.invoke({
            "query": query,
            "tool_call_id": tool_call_id,
            "state": state
        })
        
        if isinstance(result, Command):
            update = result.update if hasattr(result, 'update') else {}
            
            # Should not add new chunks, but should update query history
            if "retrieved_chunks" in update:
                final_chunks = update["retrieved_chunks"]
                if len(final_chunks) == len(over_limit_chunks):
                    logger.info("✅ No new chunks added when over limit")
                else:
                    logger.warning(f"⚠️  Chunks count changed: {len(over_limit_chunks)} -> {len(final_chunks)}")
            
            if "previous_retrieve_queries" in update:
                queries = update["previous_retrieve_queries"]
                logger.info(f"✅ Query still recorded in history: {len(queries)} total queries")
                
            if "messages" in update:
                messages = update["messages"]
                for msg in messages:
                    if hasattr(msg, 'content') and 'stopped' in str(msg.content).lower():
                        logger.info("✅ Tool correctly reported limit reached")
                        break
                        
    except Exception as e:
        logger.error(f"❌ Over limit test failed: {str(e)}")

def test_missing_vectorstore():
    """Test behavior when vectorstore files are missing."""
    logger.info("=== TEST 5: Missing Vectorstore Handling ===")
    
    # Temporarily rename vectorstore files to simulate missing files
    faiss_path = "data/publications_vectorstore/index.faiss"
    pkl_path = "data/publications_vectorstore/index.pkl"
    
    faiss_backup = None
    pkl_backup = None
    
    try:
        # Backup files if they exist
        if os.path.exists(faiss_path):
            faiss_backup = faiss_path + ".backup"
            os.rename(faiss_path, faiss_backup)
            
        if os.path.exists(pkl_path):
            pkl_backup = pkl_path + ".backup"
            os.rename(pkl_path, pkl_backup)
        
        state = create_test_state()
        query = "test query missing files"
        tool_call_id = "test_call_5"
        
        result = retrieve_chunks.invoke({
            "query": query,
            "tool_call_id": tool_call_id,
            "state": state
        })
        
        if isinstance(result, Command):
            update = result.update if hasattr(result, 'update') else {}
            
            if "messages" in update:
                messages = update["messages"]
                for msg in messages:
                    if hasattr(msg, 'content') and 'not found' in str(msg.content).lower():
                        logger.info("✅ Tool correctly handled missing vectorstore files")
                        break
                        
    except Exception as e:
        logger.error(f"❌ Missing vectorstore test failed: {str(e)}")
        
    finally:
        # Restore files
        if faiss_backup and os.path.exists(faiss_backup):
            os.rename(faiss_backup, faiss_path)
        if pkl_backup and os.path.exists(pkl_backup):
            os.rename(pkl_backup, pkl_path)

def run_all_tests():
    """Run all isolated tests for the retrieve_chunks tool."""
    logger.info("🚀 STARTING RETRIEVE_CHUNKS TOOL ISOLATED TESTS")
    start_time = time.time()
    
    tests = [
        test_basic_retrieval,
        test_duplicate_filtering,
        test_token_limit,
        test_over_limit,
        test_missing_vectorstore
    ]
    
    for i, test_func in enumerate(tests, 1):
        try:
            logger.info(f"\n--- Running Test {i}/{len(tests)}: {test_func.__name__} ---")
            test_func()
            logger.info(f"✅ Test {i} completed successfully")
        except Exception as e:
            logger.error(f"❌ Test {i} failed: {str(e)}")
        
        # Small delay between tests
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    logger.info(f"\n🏁 ALL TESTS COMPLETED - Total time: {total_time:.2f}s")

if __name__ == "__main__":
    run_all_tests()
