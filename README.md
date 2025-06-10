# 1.) RAG System for Regulatory Compliance 

Goal: Equip LLMs with relevant context and solid answering for questions in relevant domains.

We explore RAG for applying regulatory compliance to 3 common LLMs. 

    "thenlper/gte-large",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2"

We leverage FAISS for the vector db

1. **Document Collection**  
   Gather regulatory PDFs (e.g., HIPAA, SOC 2, FedRAMP).

2. **Chunking**  
   Split documents into overlapping, context-preserving chunks for optimal retrieval and LLM performance.

3. **Embedding**  
   Convert each chunk to a dense vector using one of the selected embedding models.

4. **Vector Database Creation**  
   Store all chunk embeddings in a FAISS vector database for fast similarity search.

5. **Retrieval**  
   At query time, embed the question and retrieve the most relevant chunks from the vector DB.

6. **LLM Generation**  
   Feed retrieved chunks and the query to an LLM (e.g., Mistral-7B, TinyLLM, or other Hugging Face models) to generate a grounded, auditable answer.

7. **Evaluation**  
   Assess retrieval and answer quality using precision, recall, and compliance-specific metrics.


