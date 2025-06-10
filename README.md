# 1.) RAG System for Regulatory Compliance 

Example: What are the 8 required elements of a valid HIPPA autorization form?

LLM w/ RAG:  "The HIPAA Privacy Rule sets out specific requirements for valid authorizations for the use and disclosure of protected health information (PHI) under \u00a7164.508(c). The following eight elements are required in a valid HIPAA authorization form:
- Description of the information to be used or disclosed: The authorization must specify the PHI that is to be used or disclosed.
- Name or title of the person, or the organization, entity, or business associates authorized to make the requested use or disclosure.
- Purpose of the requested use or disclosure: The authorization must state the purpose of the requested use or disclosure.
- Expiration date: An expiration date, if any, must be specified in the authorization.
- Signature of the individual: The individual must sign the authorization.
- Date: The date the individual signed the authorization must be included.
- Statement of the individual's right to revoke the authorization: The authorization must include a clear and conspicuous statement of the individual's right to revoke the authorization in writing.
- Contact person: The individual must be informed of the contact person to whom the individual may direct any inquiries concerning the use or disclosure of their PHI.


LLM w/out RAG: "Under HIPAA (Health Insurance Portability and Accountability Act) regulations, a valid authorization form for the disclosure of protected health information (PHI) must contain the following eight elements as specified in \u00a7164.508(c):\n\n1.
- Name of the person who is authorizing the use or disclosure of the protected health information (PHI). This is the individual who is granting permission for their PHI to be shared.
- A description of the PHI to be used or disclosed. This should be specific enough to clearly identify the information being shared.
- The name of the person(s) or entity who may make the requested use or disclosure of the PHI. This is the person or organization that will be receiving and using the PHI.
- A description of the permitted or required purpose of the use or disclosure of the PHI. This explains why the information is being shared and what will be done with it.
- An expiration date, if any. This is the date after which the authorization will no longer be valid. If no expiration date is specified, the authorization will remain in effect until it is revoked by the individual.
- A signature of the individual who is authorizing the use or disclosure of the PHI. This is the signature of the person granting permission for their PHI to be shared.
- A date on which the individual signed the authorization. This is the date when the individual granted permission for their PHI to be shared.
- A statement of the individual's right to revoke the authorization. This statement should inform the individual that they have the right to revoke their authorization in writing at any time, and that this revocation will be honored by the person or organization that received the authorization.

**Analysis**: Incorporating RAG results in more structured and complete responses, reducing unnecessary elaboration compared to using the LLM alone. This enhances both clarity and coherence. Notably, the use of a limited set of 10 documents introduces only a minimal increase in computational overhead.
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


