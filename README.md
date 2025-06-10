# 1.) RAG System for Regulatory Compliance 

**Implementation Example:** To evaluate RAG’s impact on compliance accuracy, we processed 10 critical regulatory PDFs (HIPAA, SOC2, FedRAMP) and utilized mistralai/Mistral-7B-Instruct-v0.2 for answer generation.

Prompt: What are the 8 required elements of a valid HIPPA autorization form?

LLM w/ RAG:
- Description of the information to be used or disclosed: The authorization must specify the PHI that is to be used or disclosed.
- Name or title of the person, or the organization, entity, or business associates authorized to make the requested use or disclosure.
- Purpose of the requested use or disclosure: The authorization must state the purpose of the requested use or disclosure.
....


LLM w/out RAG:
- Name of the person who is authorizing the use or disclosure of the protected health information (PHI). This is the individual who is granting permission for their PHI to be shared.
- A description of the PHI to be used or disclosed. This should be specific enough to clearly identify the information being shared.
- The name of the person(s) or entity who may make the requested use or disclosure of the PHI. This is the person or organization that will be receiving and using the PHI.
....

**Analysis**: Incorporating RAG results in more concise and complete responses, reducing unnecessary elaboration compared to using the LLM alone. This enhances both clarity and coherence. Notably, the use of a limited set of 10 documents introduces only a minimal increase in computational overhead.

We evaluate Retrieval-Augmented Generation (RAG) techniques for regulatory compliance by benchmarking three widely used embedding models:

- thenlper/gte-large

- intfloat/e5-large-v2

- sentence-transformers/all-mpnet-base-v2

**Overview**

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


