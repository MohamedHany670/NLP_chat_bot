# RAG-Based E-commerce Customer Support Chatbot

This project implements an end-to-end customer support chatbot using traditional NLP, deep learning, intent classification, Retrieval-Augmented Generation (RAG), and a FastAPI deployment layer.

The chatbot analyzes each customer message through four main NLP stages:

1. Language Detection
2. Sentiment Classification
3. Intent Classification
4. RAG-based Response Generation

The final system is exposed locally through a FastAPI API.

---

## System Architecture

```text
Customer Message
      |
      v
Language Detection
      |
      v
Translation to English if needed
      |
      v
Sentiment Classification
      |
      v
Intent Classification
      |
      v
Routing
      |
      +-- Conversation --------> Direct Response
      |
      +-- Out of Scope --------> Human Escalation
      |
      +-- Complaint -----------> Priority RAG
      |
      +-- Negative Support ----> Empathetic RAG
      |
      +-- Normal Support ------> RAG
                                  |
                                  v
                            Chroma Retrieval
                                  |
                                  v
                            Groq LLM Response