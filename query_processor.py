import os
import json
import torch
import argparse
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# Suppress TensorFlow CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODELS_TO_TEST = [
    "thenlper/gte-large",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

PROMPT_TEMPLATE = """[INST] You are a compliance expert analyzing documents. 
Use these context excerpts to answer the query. Cite sources from metadata.
Be precise with regulatory references (§ symbols, article numbers).

Context:
{context}

Question: {question} [/INST]
Helpful Answer:"""

from langchain_community.llms import HuggingFacePipeline

# Replace your existing load_llm() function with this:
def load_llm():
    """Properly wrapped LLM for LangChain compatibility"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1
    )
    
    return HuggingFacePipeline(pipeline=pipe)  # Critical wrapper

def verify_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected! Required for processing")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

def get_embeddings(model_name: str):
    """Optimized embeddings for compliance docs"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "batch_size": 256,
            "normalize_embeddings": True
        }
    )

def process_model_queries(model_name: str, queries_path: str, output_dir: str, top_k: int):
    """Process queries with audit-ready traceability"""
    results = []
    model_dir = f"faiss_index_{model_name.replace('/', '_')}"
    
    if not os.path.exists(model_dir):
        print(f"Vector DB not found for {model_name}")
        return

    try:
        # Load components
        embeddings = get_embeddings(model_name)
        db = FAISS.load_local(model_dir, embeddings, allow_dangerous_deserialization=True)
        llm = load_llm()
        
        # Compliance-focused QA chain
        qa_prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": qa_prompt,
                "verbose": False  # Disable for production
            }
        )

        # Process compliance queries
        with open(queries_path) as f:
            queries = json.load(f)

        for query in queries:
            try:
                response = qa_chain.invoke({"query": query["query"]})
                
                # Format for audit compliance
                results.append({
                    "query_id": query["id"],
                    "regulation": query["category"],
                    "question": query["query"],
                    "answer": response["result"].split("Helpful Answer:")[-1].strip(),
                    "source_documents": [{
                        "content": doc.page_content,
                        "source": os.path.basename(doc.metadata.get("source", "unknown")),
                        "page": doc.metadata.get("page", "N/A")
                    } for doc in response["source_documents"]],
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Error processing query {query['id']}: {str(e)}")

        # Save with compliance metadata
        output_file = os.path.join(output_dir, f"compliance_answers_{model_name.replace('/', '_')}.json")
        with open(output_file, 'w') as f:
            json.dump({
                "model": model_name,
                "gpu_config": {
                    "name": torch.cuda.get_device_name(0),
                    "vram_used_gb": f"{torch.cuda.max_memory_allocated()/1e9:.2f}",
                    "quantization": "4-bit NF4"
                },
                "results": results
            }, f, indent=2)

    except Exception as e:
        print(f"Critical error with {model_name}: {str(e)}")
    finally:
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Compliance RAG Processor')
    parser.add_argument('--k', type=int, default=3, 
                       help='Number of context chunks per query (default: 3)')
    args = parser.parse_args()

    verify_gpu()
    output_dir = "compliance_results"
    os.makedirs(output_dir, exist_ok=True)

    for model in MODELS_TO_TEST:
        print(f"\n{'='*40}")
        print(f"Processing {model} (HIPAA/SOC2/FedRAMP)")
        print(f"{'='*40}")
        process_model_queries(model, "queries.json", output_dir, args.k)

if __name__ == "__main__":
    # Install: pip install bitsandbytes>=0.43.0 transformers>=4.40.0 langchain faiss-gpu
    main()
