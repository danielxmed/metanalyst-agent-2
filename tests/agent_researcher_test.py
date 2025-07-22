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


result = researcher_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Search for studies on type 2 diabetes and metformin"
            }
        ],
        "meta_analysis_pico": {
            "population": "people with type 2 diabetes",
            "intervention": "metformin",
            "comparison": "placebo",
            "outcome": "glycemic control"
        },
        "user_request": "Search for studies on type 2 diabetes and metformin",
        "previous_search_queries": [],
        "remaining_steps": 15,  # Step limit to avoid infinite recursion
        "current_iteration": 1,
        "urls_to_process": [],
        "previous_retrieve_queries": [],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft_iteration": 0,
        "reviewer_feedbacks": []
    },
    config={"recursion_limit": 20}  # Increase recursion limit
)

print(result)
