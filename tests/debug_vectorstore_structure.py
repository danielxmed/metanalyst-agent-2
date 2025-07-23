"""
Script to debug the structure of the vectorstore metadata
"""

import pickle
import os

def investigate_pkl_structure():
    pkl_path = "data/publications_vectorstore/index.pkl"
    
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return
    
    print(f"🔍 Investigating structure of {pkl_path}")
    
    try:
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"📊 Type of loaded object: {type(metadata)}")
        print(f"📊 Length/size: {len(metadata) if hasattr(metadata, '__len__') else 'N/A'}")
        
        # Check if it's a list/array
        if hasattr(metadata, '__getitem__') and hasattr(metadata, '__len__'):
            if len(metadata) > 0:
                print(f"📊 First item type: {type(metadata[0])}")
                print(f"📊 First item: {str(metadata[0])[:200]}...")
                
                # Check if first item has attributes we can access
                first_item = metadata[0]
                if hasattr(first_item, '__dict__'):
                    print(f"📊 First item attributes: {list(first_item.__dict__.keys())}")
                elif hasattr(first_item, 'page_content'):
                    print(f"📊 First item has page_content: {first_item.page_content[:100]}...")
                elif hasattr(first_item, 'content'):
                    print(f"📊 First item has content: {first_item.content[:100]}...")
        
        # Check if it's a dictionary  
        elif isinstance(metadata, dict):
            print(f"📊 Dictionary keys: {list(metadata.keys())[:10]}...")
            if metadata:
                first_key = list(metadata.keys())[0]
                print(f"📊 First value type: {type(metadata[first_key])}")
        
        # Check if it's a docstore object
        elif hasattr(metadata, 'docs') or hasattr(metadata, '_docs'):
            print("📊 Appears to be a docstore object")
            if hasattr(metadata, 'docs'):
                print(f"📊 Has 'docs' attribute with {len(metadata.docs)} items")
            if hasattr(metadata, '_docs'):
                print(f"📊 Has '_docs' attribute with {len(metadata._docs)} items")
        
        # If it's a tuple with 2 elements (typical FAISS structure)
        if isinstance(metadata, tuple) and len(metadata) == 2:
            docstore, index_to_docstore_id = metadata
            print(f"📊 Element 1 (docstore): {type(docstore)}")
            print(f"📊 Element 2 (index_to_docstore_id): {type(index_to_docstore_id)}")
            
            # Check docstore structure
            if hasattr(docstore, '_dict'):
                docs_dict = docstore._dict
                print(f"📊 Docstore has {len(docs_dict)} documents")
                if docs_dict:
                    first_key = list(docs_dict.keys())[0]
                    first_doc = docs_dict[first_key]
                    print(f"📊 First doc type: {type(first_doc)}")
                    print(f"📊 First doc attributes: {dir(first_doc)[:10]}...")
                    if hasattr(first_doc, 'page_content'):
                        print(f"📊 First doc content: {first_doc.page_content[:100]}...")
                    if hasattr(first_doc, 'metadata'):
                        print(f"📊 First doc metadata: {first_doc.metadata}")
            
            # Check index mapping
            if hasattr(index_to_docstore_id, '__len__'):
                print(f"📊 Index mapping has {len(index_to_docstore_id)} entries")
                if len(index_to_docstore_id) > 0:
                    if isinstance(index_to_docstore_id, list):
                        print(f"📊 First mapping: 0 -> {index_to_docstore_id[0]}")
                    elif isinstance(index_to_docstore_id, dict):
                        first_key = list(index_to_docstore_id.keys())[0]
                        print(f"📊 First mapping: {first_key} -> {index_to_docstore_id[first_key]}")
        
        print("✅ Investigation completed")
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")

if __name__ == "__main__":
    investigate_pkl_structure()
