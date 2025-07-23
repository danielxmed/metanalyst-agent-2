#!/usr/bin/env python3
"""
Debug test specifically for Tavily API
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv(os.path.join(project_root, '.env'))

def test_tavily_api():
    """Test Tavily API directly"""
    print("🔍 TAVILY API DEBUG TEST")
    print("="*50)
    
    # Check API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    print(f"📋 API Key present: {'✅ YES' if tavily_api_key else '❌ NO'}")
    if tavily_api_key:
        print(f"   Key starts with: {tavily_api_key[:10]}...")
    
    if not tavily_api_key:
        print("❌ TAVILY_API_KEY not found in environment!")
        return
    
    try:
        # Import and test Tavily client
        from tavily import TavilyClient
        print("✅ Successfully imported TavilyClient")
        
        # Initialize client
        tavily_client = TavilyClient(api_key=tavily_api_key)
        print("✅ Successfully initialized TavilyClient")
        
        # Test URLs (simpler URLs first)
        test_urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://www.nejm.org/doi/full/10.1056/NEJM190402251500802"
        ]
        
        print(f"\n🧪 Testing with {len(test_urls)} URLs:")
        for i, url in enumerate(test_urls):
            print(f"   {i+1}. {url}")
        
        # Execute extract request
        print("\n🚀 Executing extract request...")
        response = tavily_client.extract(urls=test_urls, include_images=False)
        
        print(f"\n📤 Response structure:")
        print(f"   Type: {type(response)}")
        print(f"   Keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
        if "results" in response:
            results = response["results"]
            print(f"   Results count: {len(results)}")
            for i, result in enumerate(results):
                print(f"   Result {i+1}:")
                print(f"      URL: {result.get('url', 'Not found')}")
                print(f"      Raw content length: {len(result.get('raw_content', ''))}")
                print(f"      Raw content preview: {result.get('raw_content', '')[:100]}...")
        
        if "failed_results" in response:
            failed_results = response["failed_results"]
            print(f"   Failed results count: {len(failed_results)}")
            for i, failed_result in enumerate(failed_results):
                print(f"   Failed Result {i+1}:")
                print(f"      URL: {failed_result.get('url', 'Not found')}")
                print(f"      Error: {failed_result.get('error', 'No error info')}")
        
        print(f"\n📊 SUCCESS: Tavily API is working!")
        return response
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
    except Exception as e:
        print(f"❌ API Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tavily_api()
