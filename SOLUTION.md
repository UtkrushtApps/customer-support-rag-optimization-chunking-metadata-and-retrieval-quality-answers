# Solution Steps

1. Load the support documents, ensuring each has an id, text, category, priority, and date metadata.

2. Implement a chunking function that splits each document into overlapping chunks: 200 tokens per chunk, with a 50-token overlap.

3. For each chunk, attach metadata: doc_id, category, priority, date, and chunk sequence number.

4. Load the 'all-MiniLM-L6-v2' SentenceTransformer model for embedding.

5. Embed each chunk's text using the model, normalizing embeddings for cosine similarity.

6. Set up a Chroma vector database collection with cosine similarity enabled (use hnsw:space: cosine).

7. Delete all existing contents in the collection to ensure a clean database for reruns.

8. Ingest all chunks into Chroma, storing the chunk text, embedding, and metadata.

9. Implement the retrieval function to embed any query and fetch the top-5 closest chunks from Chroma by cosine similarity, returning also chunk metadata.

10. Evaluate retrieval: run manual spot checks to compare retrieved chunk categories to query intent.

11. Calculate mock recall@5 by comparing if any relevant chunk's doc_id occurs within the top 5 retrieved chunk results.

