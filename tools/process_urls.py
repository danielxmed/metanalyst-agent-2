from state.state import MetaAnalysisState
from typing import Annotated, Any, Dict, List
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import asyncio
import json
import os
import logging
from pathlib import Path
import hashlib
import concurrent.futures
import shutil
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create data directories
DATA_DIR = Path("data")
FULL_JSON_DIR = DATA_DIR / "full_json_referenced"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = DATA_DIR

for dir_path in [DATA_DIR, FULL_JSON_DIR, CHUNKS_DIR, VECTORSTORE_DIR]:
    dir_path.mkdir(exist_ok=True)

class PublicationReference(BaseModel):
    """Schema for structured LLM output"""
    reference: str = Field(description="Complete APA-style reference")
    title: str = Field(description="Publication title", default="Not available")
    authors: str = Field(description="Authors", default="Not available")
    year: str = Field(description="Publication year", default="Not available")
    journal: str = Field(description="Journal name", default="Not available")
    doi: str = Field(description="DOI", default="Not available")

@tool
def process_urls(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[MetaAnalysisState, InjectedState]
) -> Command:
    """
    Process URLs from state in batch: extract content, reference, chunk, vectorize, and store.
    
    This tool takes the urls_to_process list from the state and processes all URLs in parallel:
    1. Extract content using Tavily MCP API
    2. Generate APA references using LLM in batch
    3. Chunk content (200 tokens per chunk, 20 tokens overlap)
    4. Vectorize and store in local FAISS vectorstore
    5. Clear the urls_to_process list from state
    """
    
    def create_url_hash(url: str) -> str:
        """Create a unique hash for URL to use as filename."""
        return hashlib.md5(url.encode()).hexdigest()

    def extract_urls_with_tavily(urls: List[str]) -> List[Dict[str, Any]]:
        """Extract content from URLs using Tavily Python client."""
        try:
            logger.info(f"Extracting content from {len(urls)} URLs using Tavily Python client")
            
            # Import Tavily client
            try:
                from tavily import TavilyClient
            except ImportError:
                raise ImportError("Tavily Python client not installed. Run: pip install tavily-python")
            
            # Get Tavily API key from environment
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY not found in environment")
            
            # Initialize Tavily client
            tavily_client = TavilyClient(api_key=tavily_api_key)
            
            # Execute extract request
            response = tavily_client.extract(
                urls=urls,
                include_images=False,
                extract_depth="advanced",
                format="text"
            )
            
            extracted_data = []
            
            # Process successful results
            if "results" in response:
                for result in response["results"]:
                    extracted_data.append({
                        "url": result["url"],
                        "content": result.get("raw_content", ""),
                        "status": "success" if result.get("raw_content") else "error"
                    })
            
            # Process failed results
            if "failed_results" in response:
                for failed_result in response["failed_results"]:
                    logger.warning(f"Failed to extract {failed_result['url']}: {failed_result.get('error', 'Unknown error')}")
                    extracted_data.append({
                        "url": failed_result["url"],
                        "content": "",
                        "status": "error",
                        "error": failed_result.get("error", "Unknown error")
                    })
            
            # Ensure all URLs are accounted for
            processed_urls = [d["url"] for d in extracted_data]
            for url in urls:
                if url not in processed_urls:
                    extracted_data.append({
                        "url": url,
                        "content": "",
                        "status": "error",
                        "error": "URL not found in response"
                    })
            
            successful_count = len([d for d in extracted_data if d["status"] == "success"])
            logger.info(f"Successfully extracted content from {successful_count}/{len(urls)} URLs")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting URLs with Tavily: {str(e)}")
            # Return error status for all URLs
            return [{"url": url, "content": "", "status": "error", "error": str(e)} for url in urls]

    def generate_references_batch(extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate APA references for extracted content using batch LLM calls."""
        try:
            # Initialize LLM for referencing
            reference_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
            
            # Create reference prompt
            reference_prompt = PromptTemplate(
                input_variables=["content"],
                template="""
                You are an expert academic researcher. Based on the provided content from a scientific publication, 
                extract the necessary information to create an APA-style reference.

                Content (first 8000 characters):
                {content}

                Please analyze this content and return ONLY a JSON object with the following structure:
                {{
                    "reference": "Complete APA-style reference for this publication",
                    "title": "Title of the publication",
                    "authors": "Authors of the publication", 
                    "year": "Publication year",
                    "journal": "Journal name (if applicable)",
                    "doi": "DOI (if available)"
                }}

                If some information is not available in the content, use "Not available" for those fields.
                Ensure the reference follows proper APA format.
                """
            )
            
            # Create chain with structured output
            chain = reference_prompt | reference_llm.with_structured_output(PublicationReference)
            
            # Prepare inputs for batch processing
            valid_extractions = [d for d in extracted_data if d["status"] == "success" and d["content"]]
            
            if not valid_extractions:
                logger.warning("No valid extractions to reference")
                return []
            
            # Prepare batch inputs (truncate content to 8000 chars)
            batch_inputs = [{"content": d["content"][:8000]} for d in valid_extractions]
            
            logger.info(f"Generating references for {len(batch_inputs)} publications")
            
            # Execute batch processing
            batch_results = chain.batch(batch_inputs, config={"max_concurrency": 5})
            
            # Combine results with original data
            referenced_data = []
            for i, extraction in enumerate(valid_extractions):
                if i < len(batch_results):
                    ref_result = batch_results[i]
                    referenced_data.append({
                        "reference": ref_result.reference,
                        "url": extraction["url"],
                        "content": extraction["content"],
                        "metadata": {
                            "title": ref_result.title,
                            "authors": ref_result.authors,
                            "year": ref_result.year,
                            "journal": ref_result.journal,
                            "doi": ref_result.doi
                        }
                    })
                else:
                    referenced_data.append({
                        "reference": "Error generating reference",
                        "url": extraction["url"],
                        "content": extraction["content"],
                        "metadata": {}
                    })
            
            logger.info(f"Generated references for {len(referenced_data)} publications")
            return referenced_data
            
        except Exception as e:
            logger.error(f"Error generating references: {str(e)}")
            # Return basic referenced data without proper references
            return [{
                "reference": f"Error generating reference: {str(e)}",
                "url": d["url"],
                "content": d["content"],
                "metadata": {}
            } for d in extracted_data if d["status"] == "success"]

    def save_referenced_jsons(referenced_data: List[Dict[str, Any]]) -> List[str]:
        """Save referenced publications as JSON files."""
        try:
            saved_files = []
            for data in referenced_data:
                url_hash = create_url_hash(data["url"])
                filename = f"ref_{url_hash}.json"
                filepath = FULL_JSON_DIR / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(str(filepath))
            
            logger.info(f"Saved {len(saved_files)} referenced JSON files")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving referenced JSONs: {str(e)}")
            raise

    def chunk_publications(referenced_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk all publications with 200 tokens per chunk and 20 tokens overlap."""
        try:
            # Initialize text splitter (approx 200 tokens = 800 chars, 20 tokens = 80 chars)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            )
            
            all_chunks = []
            
            for pub_data in referenced_data:
                content = pub_data["content"]
                if not content:
                    continue
                
                # Split the content
                chunks = text_splitter.split_text(content)
                
                # Create chunk objects
                for i, chunk in enumerate(chunks):
                    chunk_obj = {
                        "reference": pub_data["reference"],
                        "url": pub_data["url"],
                        "content": chunk,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "metadata": pub_data["metadata"]
                    }
                    all_chunks.append(chunk_obj)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(referenced_data)} publications")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error chunking publications: {str(e)}")
            raise

    def save_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
        """Save chunks as JSON files."""
        try:
            saved_files = []
            for chunk in chunks:
                url_hash = create_url_hash(chunk["url"])
                chunk_filename = f"chunk_{url_hash}_{chunk['chunk_index']}.json"
                chunk_filepath = CHUNKS_DIR / chunk_filename
                
                with open(chunk_filepath, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, indent=2, ensure_ascii=False)
                
                saved_files.append(str(chunk_filepath))
            
            logger.info(f"Saved {len(saved_files)} chunk files")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving chunks: {str(e)}")
            raise

    def vectorize_and_store(chunks: List[Dict[str, Any]]) -> str:
        """Vectorize chunks and store in FAISS vectorstore."""
        try:
            if not chunks:
                raise ValueError("No chunks to vectorize")
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Prepare texts and metadata
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [
                {
                    "reference": chunk["reference"],
                    "url": chunk["url"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    **chunk.get("metadata", {})
                }
                for chunk in chunks
            ]
            
            # Create or update vectorstore
            vectorstore_path = VECTORSTORE_DIR / "publications_vectorstore"
            
            if vectorstore_path.exists():
                # Load existing vectorstore and add new documents
                try:
                    vectorstore = FAISS.load_local(
                        str(vectorstore_path), 
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    vectorstore.add_texts(texts, metadatas=metadatas)
                    logger.info(f"Added {len(texts)} new documents to existing vectorstore")
                except Exception as e:
                    logger.warning(f"Could not load existing vectorstore: {e}. Creating new one.")
                    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
                    logger.info(f"Created new vectorstore with {len(texts)} documents")
            else:
                # Create new vectorstore
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
                logger.info(f"Created new vectorstore with {len(texts)} documents")
            
            # Save vectorstore
            vectorstore.save_local(str(vectorstore_path))
            logger.info(f"Saved vectorstore to {vectorstore_path}")
            
            return str(vectorstore_path)
            
        except Exception as e:
            logger.error(f"Error vectorizing and storing: {str(e)}")
            raise

    def cleanup_temp_directories():
        """Clean up temporary directories after vectorstore creation."""
        try:
            # Clear full_json_referenced directory
            if FULL_JSON_DIR.exists():
                shutil.rmtree(FULL_JSON_DIR)
                FULL_JSON_DIR.mkdir(exist_ok=True)
                logger.info(f"Cleared {FULL_JSON_DIR} directory")
            
            # Clear chunks directory
            if CHUNKS_DIR.exists():
                shutil.rmtree(CHUNKS_DIR)
                CHUNKS_DIR.mkdir(exist_ok=True)
                logger.info(f"Cleared {CHUNKS_DIR} directory")
                
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directories: {str(e)}")

    # Main execution logic
    urls_to_process = state.get("urls_to_process", [])
    processed_urls_existing = state.get("processed_urls", [])
    
    if not urls_to_process:
        logger.info("No URLs to process")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No URLs to process",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # DEDUPLICATION:
    # 1) remove duplicatas mantendo ordem original
    # 2) descartar URLs já processadas
    seen, deduped = set(), []
    for u in urls_to_process:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    urls_to_process_filtered = [url for url in deduped if url not in processed_urls_existing]
    
    if not urls_to_process_filtered:
        logger.info("All URLs already processed")
        return Command(
            update={
                "urls_to_process": [],  # Clear the list since all are already processed
                "messages": [
                    ToolMessage(
                        content="All URLs have already been processed",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # PROCESSAMENTO EM LOTE COM ROTAÇÃO ATÉ 200 URLS
    MAX_BATCH_SIZE = 20          # requisição segura à Tavily/LLM
    MAX_TOTAL_PER_INVOCATION = 200  # limite global por chamada da ferramenta

    # Limita o total a processar nesta invocação
    urls_to_handle_now = urls_to_process_filtered[:MAX_TOTAL_PER_INVOCATION]
    remaining_urls = urls_to_process_filtered[MAX_TOTAL_PER_INVOCATION:]

    try:
        processed_urls: List[str] = []
        total_chunks = 0
        saved_json_files: List[str] = []
        saved_chunk_files: List[str] = []
        vectorstore_path = ""

        # Itera em blocos de 20 até esgotar ou atingir 200
        for start in range(0, len(urls_to_handle_now), MAX_BATCH_SIZE):
            urls_batch = urls_to_handle_now[start:start + MAX_BATCH_SIZE]
            logger.info(f"Processing batch of {len(urls_batch)} URLs "
                        f"(batch {start//MAX_BATCH_SIZE + 1}/"
                        f"{(len(urls_to_handle_now)+MAX_BATCH_SIZE-1)//MAX_BATCH_SIZE})")

            # Step 1: Extract content
            extracted_data = extract_urls_with_tavily(urls_batch)
            successful_extractions = [d for d in extracted_data if d["status"] == "success"]
            if not successful_extractions:
                logger.warning("Batch skipped: no successful extractions")
                continue

            # Step 2: Generate references
            referenced_data = generate_references_batch(successful_extractions)
            if not referenced_data:
                logger.warning("Batch skipped: failed to generate references")
                continue

            # Step 3: Save referenced JSONs
            saved_json_files.extend(save_referenced_jsons(referenced_data))

            # Step 4: Chunk publications
            chunks = chunk_publications(referenced_data)

            # Step 5: Save chunks
            saved_chunk_files.extend(save_chunks(chunks))

            # Step 6: Vectorize and store
            vectorstore_path = vectorize_and_store(chunks)

            # Acumula estatísticas
            processed_urls.extend([d["url"] for d in referenced_data])
            total_chunks += len(chunks)

            # Limpeza temporária a cada lote
            cleanup_temp_directories()

        logger.info(f"Successfully processed {len(processed_urls)} URLs in this invocation, "
                    f"created {total_chunks} chunks")

        # Marcar TODAS as URLs manuseadas neste lote como processadas,
        # independentemente de falha ou sucesso, para evitar loops no supervisor
        processed_urls_updated = list(set(processed_urls_existing + urls_to_handle_now))
        
        # Somente mantemos em urls_to_process aquelas que sobraram fora
        # do limite MAX_TOTAL_PER_INVOCATION (remaining_urls)
        urls_to_process_updated = remaining_urls

        # Update state: remove processed URLs and keep only unprocessed/failed ones
        batch_status = f"Batch {len(processed_urls)}/{len(urls_to_handle_now)} successful"
        remaining_status = f", {len(urls_to_process_updated)} URLs remaining" if urls_to_process_updated else ", all URLs processed"
        
        return Command(
            update={
                "urls_to_process": urls_to_process_updated,  # Remove processed URLs, keep only unprocessed/failed
                "processed_urls": processed_urls_updated,  # Deduplicated processed URLs
                "messages": [
                    ToolMessage(
                        content=f"Successfully processed {len(processed_urls)}/{len(urls_to_handle_now)} URLs ({batch_status}){remaining_status}. "
                                f"Generated {total_chunks} chunks and stored in vectorstore at {vectorstore_path}. "
                                f"Saved {len(saved_json_files)} referenced publications and {len(saved_chunk_files)} chunks.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Error in process_urls tool: {str(e)}")
        
        # On failure, update state to avoid infinite loops
        processed_urls_updated = list(set(processed_urls_existing + urls_to_handle_now))
        urls_to_process_updated = remaining_urls
        
        return Command(
            update={
                "urls_to_process": urls_to_process_updated,
                "processed_urls": processed_urls_updated,
                "messages": [
                    ToolMessage(
                        content=f"Error processing URLs: {str(e)}. "
                                f"Marked {len(urls_to_handle_now)} URLs as processed to avoid loop.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
