# Retrieval Augmented Generation (RAG)

## 1. Definition

Retrieval Augmented Generation (RAG) is a technique used to improve the output of Large Language Models (LLMs) by providing them with external knowledge that is not part of their original training data.

LLMs are trained on large datasets and use billions of parameters to generate outputs such as:
- Question answering
- Language translation
- Text generation

RAG allows LLMs to reference an external knowledge base while generating responses.  
This external knowledge can include:
- Real-time data
- Domain-specific information
- Organization’s internal knowledge base

RAG is a cost-effective way to improve LLM performance without retraining.  

**Augmentation means giving additional context to the LLM at query time.**

---

## 2. How LLMs Work

### Content Generation Flow

User → Query + Prompt → LLM → Response

---

## 3. Limitations of LLMs

LLMs have several limitations:

- They are trained on a fixed dataset
- They do not have access to real-time information
- They may hallucinate when the required knowledge is missing
- Training or retraining LLMs is:
  - Time-consuming
  - Computationally expensive
  - Costly due to billions of parameters

---

## 4. Why RAG is Needed (Use Case)

Consider a startup building a customer support chatbot.

The chatbot should:
- Provide real-time information
- Answer domain-specific queries
- Access internal documents such as:
  - Company policies
  - HR-related information
  - Frequently updated internal knowledge

Training or fine-tuning an LLM every time this data changes is not practical.

RAG solves this problem by allowing the LLM to retrieve relevant information from an external knowledge base without retraining.

---

## 5. RAG Architecture Overview

- The LLM is pre-trained on large public datasets
- An external database (Vector Database) stores custom knowledge
- During query time, relevant information is retrieved from the vector database and passed to the LLM

---

## 6. Data Ingestion Pipeline

The data ingestion pipeline is responsible for preparing and storing knowledge in the vector database.

### Steps Involved

1. Data Source  
   Data can be:
   - Structured (databases, CSV files)
   - Unstructured (PDFs, documents, web pages)

2. Parsing  
   Structured and unstructured data is read and converted into smaller, manageable chunks.

3. Chunking  
   Large documents are split into smaller parts to improve retrieval accuracy.

4. Embedding Generation  
   Each chunk is converted into a vector.
   Vectors are numerical representations of text that enable similarity search using techniques like cosine similarity.

5. Embedding Models  
   Common embedding models include:
   - OpenAI Embeddings
   - Google Gemini
   - Hugging Face embedding models  
   The choice depends on cost, performance, and availability.

6. Vector Database  
   The vector database:
   - Stores vector embeddings and metadata
   - Acts as the external knowledge base  

   Examples include FAISS, Pinecone, Chroma, and Weaviate.

The knowledge stored in the vector database does not exist inside the LLM.

---

## 7. Retrieval Pipeline (Traditional RAG)

### Query Flow

User Query  
→ Query converted into embedding  
→ Similarity search in Vector Database  
→ Relevant context retrieved  
→ Context and prompt sent to LLM  
→ LLM generates response

---

## 8. Limitations of Traditional RAG

Traditional RAG has some limitations:

- It does not completely eliminate hallucinations
- If relevant information is not present in the vector database, the LLM may still hallucinate
- Performance depends on:
  - Quality of data
  - Chunking strategy
  - Retrieval method
  - Prompt design

Example:  
Products like Perplexity AI are based on RAG-based approaches.

---

## 9. Key Takeaways

- RAG combines LLMs with external knowledge
- It improves accuracy, relevance, and freshness of responses
- It eliminates the need for expensive retraining
- It is best suited for:
  - Chatbots
  - Enterprise search
  - Internal knowledge management systems
