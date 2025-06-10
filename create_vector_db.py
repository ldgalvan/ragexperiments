import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader

MODELS_TO_TEST = [
    "thenlper/gte-large",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

def create_vector_db_for_model(model_name: str):
    """Create and save FAISS index for a specific embedding model"""
    print(f"\n{'='*50}\nProcessing {model_name}\n{'='*50}")
    
    try:
        # Configure embeddings with GPU optimization
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 512  # For embedding generation only
            }
        )
        
        # Load documents
        loader = JSONLoader(
            file_path="embedded_chunks.json",
            jq_schema=".[] | {page_content: .text, metadata}",
            text_content=False
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        # Create vector store (removed batch_size parameter)
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        # Save with model-specific name
        dir_name = f"faiss_index_{model_name.replace('/', '_')}"
        vector_store.save_local(dir_name)
        print(f"✅ Successfully saved vector DB to {dir_name}")
        return vector_store

    except Exception as e:
        print(f"❌ Error processing {model_name}: {str(e)}")
        return None

def test_all_models(query: str = "What are the key compliance requirements?"):
    """Test retrieval across all created vector DBs"""
    print("\n\n=== TESTING ALL MODELS ===")
    for model_name in MODELS_TO_TEST:
        dir_name = f"faiss_index_{model_name.replace('/', '_')}"
        if not os.path.exists(dir_name):
            print(f"\n⚠️  Skipping {model_name} - no vector DB found")
            continue
            
        try:
            print(f"\nTesting {model_name}:")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            db = FAISS.load_local(
                dir_name,
                embeddings,
                allow_dangerous_deserialization=True  # Add this line
            )
            docs = db.similarity_search(query, k=2)
            
            print(f"Top results for '{query}':")
            for i, doc in enumerate(docs):
                print(f"[{i+1}] {doc.page_content[:150]}...")
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")


if __name__ == "__main__":
    # First install required packages:
    # pip install langchain-community==0.2.1 faiss-gpu==1.8.0 sentence-transformers
    
    # Create vector DBs for all models
    for model in MODELS_TO_TEST:
        create_vector_db_for_model(model)
    
    # Test retrieval across all models
    test_all_models()
