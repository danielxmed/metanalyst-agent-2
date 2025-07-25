"""
URL management utilities for handling URLs in JSON files instead of state.
"""

import json
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

# Paths to URL files
DATA_DIR = Path("data")
URLS_DIR = DATA_DIR / "urls"
URLS_TO_PROCESS_FILE = URLS_DIR / "urls_to_process.json"
PROCESSED_URLS_FILE = URLS_DIR / "processed_urls.json"

def ensure_url_files_exist():
    """Ensure URL files and directory exist."""
    URLS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not URLS_TO_PROCESS_FILE.exists():
        with open(URLS_TO_PROCESS_FILE, 'w') as f:
            json.dump([], f)
    
    if not PROCESSED_URLS_FILE.exists():
        with open(PROCESSED_URLS_FILE, 'w') as f:
            json.dump([], f)

def load_urls_to_process() -> List[str]:
    """Load URLs to process from JSON file."""
    ensure_url_files_exist()
    try:
        with open(URLS_TO_PROCESS_FILE, 'r') as f:
            urls = json.load(f)
            return urls if isinstance(urls, list) else []
    except (json.JSONDecodeError, FileNotFoundError):
        logger.warning("Error loading urls_to_process.json, returning empty list")
        return []

def load_processed_urls() -> List[str]:
    """Load processed URLs from JSON file."""
    ensure_url_files_exist()
    try:
        with open(PROCESSED_URLS_FILE, 'r') as f:
            urls = json.load(f)
            return urls if isinstance(urls, list) else []
    except (json.JSONDecodeError, FileNotFoundError):
        logger.warning("Error loading processed_urls.json, returning empty list")
        return []

def save_urls_to_process(urls: List[str]):
    """Save URLs to process to JSON file."""
    ensure_url_files_exist()
    with open(URLS_TO_PROCESS_FILE, 'w') as f:
        json.dump(urls, f, indent=2)

def save_processed_urls(urls: List[str]):
    """Save processed URLs to JSON file."""
    ensure_url_files_exist()
    with open(PROCESSED_URLS_FILE, 'w') as f:
        json.dump(urls, f, indent=2)

def add_urls_to_process(new_urls: List[str]) -> int:
    """
    Add new URLs to process with deduplication.
    Returns the number of URLs actually added.
    """
    if not new_urls:
        return 0
    
    current_to_process = load_urls_to_process()
    processed_urls = load_processed_urls()
    
    # Create sets for fast lookup
    to_process_set = set(current_to_process)
    processed_set = set(processed_urls)
    
    # Deduplicate and filter out already processed URLs
    urls_to_add = []
    for url in new_urls:
        if url not in to_process_set and url not in processed_set:
            urls_to_add.append(url)
            to_process_set.add(url)  # Update set to avoid duplicates within this batch
    
    if urls_to_add:
        updated_urls = current_to_process + urls_to_add
        save_urls_to_process(updated_urls)
        logger.info(f"Added {len(urls_to_add)} new URLs to process")
    
    return len(urls_to_add)

def move_urls_to_processed(urls_to_move: List[str]) -> int:
    """
    Move URLs from to_process to processed.
    Returns the number of URLs actually moved.
    """
    if not urls_to_move:
        return 0
        
    current_to_process = load_urls_to_process()
    processed_urls = load_processed_urls()
    
    # Remove from to_process
    updated_to_process = [url for url in current_to_process if url not in urls_to_move]
    
    # Add to processed (with deduplication)
    processed_set = set(processed_urls)
    urls_to_add_to_processed = [url for url in urls_to_move if url not in processed_set]
    updated_processed = processed_urls + urls_to_add_to_processed
    
    # Save both files
    save_urls_to_process(updated_to_process)
    save_processed_urls(updated_processed)
    
    moved_count = len(current_to_process) - len(updated_to_process)
    logger.info(f"Moved {moved_count} URLs from to_process to processed")
    
    return moved_count

def get_urls_to_process_count() -> int:
    """Get count of URLs to process."""
    return len(load_urls_to_process())

def get_processed_urls_count() -> int:
    """Get count of processed URLs."""
    return len(load_processed_urls())

def get_batch_urls_to_process(batch_size: int) -> List[str]:
    """Get a batch of URLs to process."""
    urls = load_urls_to_process()
    return urls[:batch_size]