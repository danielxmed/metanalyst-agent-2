from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from typing import Annotated, List, Dict, Any
import os
import pickle
import numpy as np
import json
from datetime import datetime


@tool
def retrieve_chunks(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],  
    state: Annotated[dict, InjectedState],
    top_k: int = 50
) -> Command:
    """
    Performs a semantic search on a local vector database for gathering chunks of
    referenced medical literature in order to make a meta-analysis.
    
    Now saves chunks as individual JSON files in data/retrieved_chunks directory
    instead of storing them in state to avoid context overload.
    
    Args:
        query (str): Natural language query to search for relevant chunks
        top_k (int): Number of top similar chunks to retrieve (default: 50)
    
    Returns:
        Command: Updates state with chunk count and query history
    """
    
    try:
        # Import required libraries
        import faiss
        from langchain_openai import OpenAIEmbeddings
        
        # Check current chunk count from state
        current_chunks_count = state.get("retrieved_chunks_count", 0)
        
        # Check if we have exceeded token limit (120,000 tokens ~ 600 chunks)
        if current_chunks_count >= 600:
            limit_message = ToolMessage(
                content=f"🛑 Retrieval stopped: Already have {current_chunks_count} chunks (limit: 600 for ~120,000 tokens).\n"
                       f"Query attempted: '{query}' (not executed)",
                tool_call_id=tool_call_id
            )
            
            # Add query to history even if not executed
            # Let the custom reducer handle the query limiting
            return Command(
                update={
                    "previous_retrieve_queries": [query],  # Add single query, let reducer handle limiting
                    "messages": [limit_message]
                }
            )
        
        # Ensure retrieved_chunks directory exists
        chunks_dir = "data/retrieved_chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Paths to vector store files
        faiss_index_path = "data/publications_vectorstore/index.faiss"
        metadata_path = "data/publications_vectorstore/index.pkl"
        
        # Check if vector store files exist
        if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
            error_message = ToolMessage(
                content=f"Error: Vector store files not found at {faiss_index_path} or {metadata_path}",
                tool_call_id=tool_call_id
            )
            return Command(
                update={"messages": [error_message]}
            )
        
        # Load FAISS index
        index = faiss.read_index(faiss_index_path)
        
        # Load metadata (docstore and index mapping)
        with open(metadata_path, 'rb') as f:
            docstore, index_to_docstore_id = pickle.load(f)
        
        # Initialize embeddings model (same as used for vectorization)
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Generate embedding for the query
        query_embedding = embeddings_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Perform similarity search using cosine similarity
        scores, indices = index.search(query_vector, top_k)
        
        # Load existing chunk IDs and content to avoid duplicates
        existing_chunk_ids = set()
        existing_content = set()
        
        for filename in os.listdir(chunks_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(chunks_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_chunk = json.load(f)
                        existing_chunk_ids.add(existing_chunk.get("chunk_id", ""))
                        existing_content.add(existing_chunk.get("content", ""))
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Retrieve chunks based on search results
        retrieved_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(index_to_docstore_id):  # Ensure index is valid
                # Get document ID from index mapping
                doc_id = index_to_docstore_id[idx]
                
                # Get document from docstore
                if doc_id in docstore._dict:
                    document = docstore._dict[doc_id]
                    
                    # Create chunk dictionary with metadata
                    chunk = {
                        "content": document.page_content,
                        "source": document.metadata.get("url", "unknown"),
                        "chunk_id": doc_id,
                        "similarity_score": float(1 - score),  # Convert distance to similarity
                        "query": query,
                        "rank": i + 1,
                        "title": document.metadata.get("title", ""),
                        "authors": document.metadata.get("authors", ""),
                        "year": document.metadata.get("year", ""),
                        "journal": document.metadata.get("journal", ""),
                        "reference": document.metadata.get("reference", ""),
                        "doi": document.metadata.get("doi", ""),
                        "chunk_index": document.metadata.get("chunk_index", 0),
                        "total_chunks": document.metadata.get("total_chunks", 1),
                        "retrieved_at": datetime.now().isoformat()
                    }
                    retrieved_chunks.append(chunk)
        
        # Filter out chunks that are already retrieved
        new_chunks = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            if chunk_id not in existing_chunk_ids and content not in existing_content:
                new_chunks.append(chunk)
        
        # Save new chunks to individual JSON files
        saved_chunks = 0
        for chunk in new_chunks:
            # Create filename with timestamp and chunk_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"chunk_{timestamp}_{chunk.get('chunk_id', 'unknown')}.json"
            filepath = os.path.join(chunks_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
                saved_chunks += 1
            except Exception as e:
                print(f"Warning: Failed to save chunk to {filepath}: {e}")
        
        # Update chunk count
        new_chunks_count = current_chunks_count + saved_chunks
        
        # Estimate tokens (approximately 200 tokens per chunk)
        estimated_tokens = saved_chunks * 200
        
        # Create informative tool message
        search_message = ToolMessage(
            content=f"🔍 Retriever Agent executed semantic search.\n"
                   f"Query: '{query}'\n"
                   f"Chunks found: {len(retrieved_chunks)}\n"
                   f"New unique chunks: {saved_chunks}\n" 
                   f"Estimated tokens added: {estimated_tokens}\n"
                   f"Total chunks available: {new_chunks_count}\n"
                   f"Chunks saved to: data/retrieved_chunks/",
            tool_call_id=tool_call_id
        )
        
        # Create friendly AI message
        friendly_message = AIMessage(
            content=f"Completed semantic search for '{query}'. "
                   f"Retrieved {saved_chunks} new relevant chunks from medical literature. "
                   f"Average similarity score: {np.mean([c['similarity_score'] for c in new_chunks]):.3f} "
                   f"(total chunks: {new_chunks_count}). "
                   f"Chunks saved to data/retrieved_chunks/ directory. "
                   f"Ready for more queries or analysis.",
            name="retriever"
        )
        
        # Update state with chunk count and query history
        # Let the custom reducer handle the query limiting
        return Command(
            update={
                "retrieved_chunks_count": new_chunks_count,
                "previous_retrieve_queries": [query],  # Add single query, let reducer handle limiting
                "messages": [search_message, friendly_message]
            }
        )
        
    except ImportError as e:
        error_message = ToolMessage(
            content=f"Error: Missing required library - {str(e)}. "
                   f"Please ensure faiss-cpu and langchain-openai are installed.",
            tool_call_id=tool_call_id
        )
        return Command(
            update={"messages": [error_message]}
        )
        
    except Exception as e:
        error_message = ToolMessage(
            content=f"Error in chunk retrieval: {str(e)}",
            tool_call_id=tool_call_id
        )
        return Command(
            update={"messages": [error_message]}
        )
