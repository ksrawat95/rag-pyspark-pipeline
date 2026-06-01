import sys

# 1. Test Chunking Logic
def test_chunking():
    print("Running Text Chunking Verification...")
    
    # Importing chunking function
    from rag_pipeline_notebook import chunk_unstructured_text
    
    sample_text = (
        "This is a long technical text designed to test the chunking capability of our PySpark RAG pipeline. "
        "It should be long enough to exceed five hundred characters and result in at least two overlapping chunks. "
        "We are verifying that the chunk size boundaries and the overlap parameters work exactly as expected. "
        "The overlap parameter is set to fifty characters, which ensures that context is not lost between "
        "consecutive windows. Let's make this text even longer to guarantee it wraps across multiple chunks. "
        "Here is some more filler text talking about Spark, Delta Lake, and how vector embeddings are generated "
        "in parallel across worker nodes in our cluster."
    )
    
    chunk_size = 300
    overlap = 50
    
    chunks = chunk_unstructured_text(sample_text, chunk_size=chunk_size, overlap=overlap)
    
    print(f"Total Chunks Generated: {len(chunks)}")
    for idx, chunk in enumerate(chunks):
        print(f"\nChunk {chunk['chunk_index']} (Length: {chunk['char_length']}):")
        print(f"Text: '{chunk['chunk_text']}'")
        
        # Verify char length
        assert chunk["char_length"] <= chunk_size, f"Chunk {idx} exceeds target chunk size!"
        assert chunk["chunk_index"] == idx, "Chunk index mismatch!"
        
    # Check overlap sanity (first chunk end matches second chunk start)
    if len(chunks) > 1:
        chunk1 = chunks[0]["chunk_text"]
        chunk2 = chunks[1]["chunk_text"]
        
        # The last 'overlap' characters of chunk1 should match the start of chunk2
        expected_overlap = chunk1[-overlap:]
        actual_overlap = chunk2[:overlap]
        print(f"\nExpected Overlap (Last {overlap} chars of Chunk 0): '{expected_overlap}'")
        print(f"Actual Overlap (First {overlap} chars of Chunk 1): '{actual_overlap}'")
        
        assert expected_overlap == actual_overlap, "Chunk overlap logic is mathematically incorrect!"
        
    print("✅ Text Chunking Verification Passed!")

# 2. Test SentenceTransformer Embedding Generation (Lightweight)
def test_embeddings():
    print("\nRunning SentenceTransformer Embedding Generation Verification...")
    try:
        from sentence_transformers import SentenceTransformer
        print("sentence-transformers is installed!")
    except ImportError:
        print("⚠️ sentence-transformers package not installed locally. Skipping model loading test.")
        print("Note: This is expected if you haven't run pip install locally yet. It will run in Databricks via Cell 1.")
        return
        
    print("Loading all-MiniLM-L6-v2 model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    test_sentences = [
        "Spark processes data in parallel.",
        "Delta Lake provides ACID compliance."
    ]
    
    embeddings = model.encode(test_sentences)
    
    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embedding 0 Dimensions: {embeddings[0].shape}")
    
    assert len(embeddings) == 2, "Should have generated exactly 2 embeddings."
    assert embeddings[0].shape == (384,), "MiniLM embedding dimensions must be 384."
    
    print("✅ Embedding Generation Verification Passed!")

if __name__ == "__main__":
    try:
        test_chunking()
        test_embeddings()
        print("\n🎉 ALL LOCAL COMPONENT VERIFICATIONS PASSED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)
