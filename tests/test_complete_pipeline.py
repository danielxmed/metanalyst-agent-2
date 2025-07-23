from agents.supervisor import supervisor_agent
from state.state import MetaAnalysisState
import logging
import time

# Configure logging for debug
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("=== STARTING METANALYST PIPELINE TEST ===")

chunk_count = 0
start_time = time.time()

try:
    for chunk in supervisor_agent.stream(
        MetaAnalysisState(
            user_request = "Amiodarone vs beta-blockers for atrial fibrillation",
            messages = [
                {
                    "role": "user",
                    "content": "Amiodarone vs beta-blockers for atrial fibrillation"
                }
            ],
            urls_to_process = [],
            processed_urls = [],
            current_iteration = 1,
            remaining_steps = 10,
            meta_analysis_pico = {},
            previous_search_queries = [],
            previous_retrieve_queries = [],
            retrieved_chunks_count = 0,
            analysis_results = [],
            current_draft = "",
            current_draft_iteration = 1,
            reviewer_feedbacks = [],
            final_draft = ""
        )
    ):
        chunk_count += 1
        elapsed_time = time.time() - start_time
        
        logger.info(f"=== CHUNK #{chunk_count} (Elapsed: {elapsed_time:.2f}s) ===")
        
        # Extract agent name from the chunk
        if isinstance(chunk, dict):
            for key, value in chunk.items():
                logger.info(f"Agent: {key}")
                if isinstance(value, dict) and 'messages' in value:
                    logger.info(f"Number of messages: {len(value['messages'])}")
                    if value['messages']:
                        last_msg = value['messages'][-1]
                        if hasattr(last_msg, 'content'):
                            logger.info(f"Last message content (first 200 chars): {str(last_msg.content)[:200]}...")
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            logger.info(f"Tool calls: {[tc['name'] for tc in last_msg.tool_calls]}")
                
                # Log state info for debugging
                if 'urls_to_process' in value:
                    logger.info(f"URLs to process: {len(value['urls_to_process'])}")
                if 'processed_urls' in value:
                    logger.info(f"Processed URLs: {len(value['processed_urls'])}")
                
                # Log queries info for debugging
                if 'previous_search_queries' in value:
                    logger.info(f"Previous search queries ({len(value['previous_search_queries'])}): {value['previous_search_queries']}")
                if 'previous_retrieve_queries' in value:
                    logger.info(f"Previous retrieve queries ({len(value['previous_retrieve_queries'])}): {value['previous_retrieve_queries']}")
                if 'retrieved_chunks_count' in value:
                    logger.info(f"Retrieved chunks count: {value['retrieved_chunks_count']}")
        
        print(f"\n=== CHUNK #{chunk_count} ===")
        print(chunk)
        print("=" * 50)
        
        # Safety timeout - if one chunk takes more than 10 minutes, break
        if elapsed_time > 600:  # 10 minutes
            logger.error("TIMEOUT: Single chunk taking too long, breaking...")
            break
            
        # If we have too many chunks, something might be wrong
        if chunk_count > 100:
            logger.error("TOO MANY CHUNKS: Possible infinite loop, breaking...")
            break

except Exception as e:
    logger.error(f"ERROR in pipeline: {str(e)}", exc_info=True)
    
finally:
    total_time = time.time() - start_time
    logger.info(f"=== PIPELINE FINISHED - Total chunks: {chunk_count}, Total time: {total_time:.2f}s ===")
