# RAG PySpark Pipeline

## Project Overview
This repository contains a distributed Retrieval-Augmented Generation (RAG) and embedding pipeline running on Apache Spark (Databricks). It demonstrates how to perform large-scale document chunking, generate deep learning embeddings using Arrow-optimized Pandas UDFs, store vectors in Delta Lake, and retrieve relevant context for LLM prompt augmentation.

## Tech Stack
- **Python**: Core scripting language.
- **Apache Spark / PySpark**: Distributed data processing framework.
- **Delta Lake**: Storage layer for managing the RAG vector store.
- **Apache Arrow**: In-memory columnar data format for high-speed PySpark-to-Python serialization.
- **Sentence-Transformers (PyTorch)**: Model used for generating 384-dimensional dense vector embeddings.
- **NumPy**: Linear algebra and similarity calculation.

## Key Implementation Details
- **Arrow-Optimized Pandas UDFs**: Implements vectorized inference to bypass JVM-to-Python socket bottlenecks. It batches text inputs and utilizes CPU/GPU matrix vectorization during embedding generation.
- **Cosine Similarity via Dot Product**: Since embeddings from the `all-MiniLM-L6-v2` model are pre-normalized, Cosine Similarity simplifies to a fast, low-overhead dot product calculated across distributed executors.
- **Delta Lake Integration**: Vectors and metadata are stored in a managed Delta Lake table (`default.rag_vector_store`) for ACID compliance and temporal versioning.
- **End-to-End Testing & Validation**:
  - `validate_pipeline.py`: Tests the integrity of the data pipelines and schema properties.
  - `test_rag_pipeline.py`: Unit and E2E test suite covering chunking, embedding generation, and context retrieval.
