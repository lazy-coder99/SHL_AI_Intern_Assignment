import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def ingest_data(json_path: str, index_path: str):
    print(f"Loading data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f, strict=False)

    documents = []
    for item in data:
        # Determine test_type (first character of the first key, defaults to 'O' for Other)
        keys = item.get("keys", [])
        test_type = keys[0][0].upper() if keys and len(keys[0]) > 0 else "O"

        # Combine relevant fields into an LLM-friendly text format
        name = item.get("name", "")
        description = item.get("description", "")
        job_levels = item.get("job_levels_raw", "").strip(", ")
        duration = item.get("duration", "")
        languages = item.get("languages_raw", "").strip(", ")
        
        page_content = f"Assessment Name: {name}\n"
        page_content += f"Category: {', '.join(keys)}\n"
        page_content += f"Description: {description}\n"
        page_content += f"Target Job Levels: {job_levels}\n"
        page_content += f"Duration: {duration}\n"
        page_content += f"Languages: {languages}\n"
        
        metadata = {
            "name": name,
            "url": item.get("link", ""),
            "test_type": test_type,
            "description": description
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"Loaded {len(documents)} assessments. Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print(f"Saving index to {index_path}...")
    vectorstore.save_local(index_path)
    print("Done!")

if __name__ == "__main__":
    ingest_data("shl_product_catalog.json", "faiss_index")
