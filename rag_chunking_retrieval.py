# rag_chunking_retrieval.py
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import json
from tqdm import tqdm

# --------------------------
# MOCKED DATA LOADING (replace with your real data loader)
# --------------------------
def load_documents():
    """
    Mock function to load support documents.
    Each document is a dict with keys: 'id', 'text', 'category', 'priority', 'date'
    """
    # In a real implementation, replace this with real loading logic
    docs = []
    for i in range(1, 8001):
        docs.append({
            'id': f'doc_{i}',
            'text': f"This is the content of document {i}. " * 100,  # 8k*100 tokens is plenty
            'category': ('billing' if i % 3 == 0 else 'technical' if i % 3 == 1 else 'account'),
            'priority': ('low' if i % 3 == 0 else 'medium' if i % 3 == 1 else 'high'),
            'date': f"2024-05-{(i%28)+1:02d}"
        })
    return docs

# --------------------------
# TOKENIZATION LOGIC
# --------------------------
def simple_tokenizer(text:str) -> List[str]:
    """
    Very simple whitespace tokenization. Replace with a tokenizer from transformers if needed.
    """
    return text.split()

def detokenizer(tokens:List[str]) -> str:
    return ' '.join(tokens)

# --------------------------
# CHUNKING LOGIC
# --------------------------
def chunk_document(doc:Dict[str,Any], chunk_size:int=200, overlap:int=50) -> List[Dict]:
    """
    Chunks the document text into fixed token windows with overlap, attaching metadata to each chunk.
    Returns a list of {'chunk_id', 'chunk_text', 'metadata'} dicts.
    """
    tokens = simple_tokenizer(doc['text'])
    chunks = []
    start = 0
    doc_id = doc['id']
    seq = 0
    while start < len(tokens):
        end = min(start+chunk_size, len(tokens))
        chunk_toks = tokens[start:end]
        chunk_text = detokenizer(chunk_toks)
        chunk_metadata = {
            'doc_id': doc_id,
            'category': doc['category'],
            'priority': doc['priority'],
            'date': doc['date'],
            'chunk_seq': seq
        }
        chunk_id = f"{doc_id}_chunk_{seq}"
        chunks.append({'chunk_id': chunk_id, 'chunk_text': chunk_text, 'metadata': chunk_metadata})
        seq += 1
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

# --------------------------
# EMBEDDING LOADING
# --------------------------
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts:List[str], model) -> List[np.ndarray]:
    """
    Embed a batch of texts using given SentenceTransformer model.
    """
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# --------------------------
# CHROMA SETUP
# --------------------------
def get_chroma_collection(client, collection_name="support_chunks"):
    """
    Returns a persistent Chroma collection, using cosine similarity for ANN search.
    """
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,  # we'll manually supply embeddings
        metadata={
            "hnsw:space": "cosine"  # explicit cosine similarity
        }
    )

# --------------------------
# PIPELINE: INGESTION
# --------------------------
def ingest_documents(docs:List[Dict[str,Any]], batch_size:int=128):
    """
    Chunks, embeds, and stores docs & metadata into Chroma vector DB.
    """
    model = get_embedding_model()
    client = chromadb.Client(Settings(persist_directory="./chroma_db"))
    coll = get_chroma_collection(client)

    # Clear existing data
    try:
        coll.delete(where={})
    except Exception:
        pass
    
    chunk_entries = []
    for doc in tqdm(docs, desc="Chunking documents"):
        for chunk in chunk_document(doc):
            chunk_entries.append(chunk)
    print(f"Prepared {len(chunk_entries)} chunks.")

    # Now embed and add to Chroma in batches
    all_texts = [c['chunk_text'] for c in chunk_entries]
    all_ids = [c['chunk_id'] for c in chunk_entries]
    all_metas = [c['metadata'] for c in chunk_entries]

    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding and adding to Chroma"):
        texts = all_texts[i:i+batch_size]
        ids = all_ids[i:i+batch_size]
        metas = all_metas[i:i+batch_size]
        embeddings = embed_texts(texts, model)
        coll.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metas, documents=texts)
    print(f"Ingestion complete. Chunks in collection: {coll.count()}")

# --------------------------
# RETRIEVAL LOGIC (TO COMPLETE)
# --------------------------
def retrieve_top_k_chunks(
    query:str,
    k:int=5,
    collection_name:str="support_chunks",
    persist_path:str="./chroma_db"
):
    """
    Given a query string, return top-k most relevant support chunks and their metadata (using cosine similarity).
    Returns: List of dict (with 'chunk_text', 'metadata', 'score')
    """
    # Implementation:
    model = get_embedding_model()
    client = chromadb.Client(Settings(persist_directory=persist_path))
    coll = get_chroma_collection(client, collection_name)
    
    # Embed the query
    q_emb = embed_texts([query], model)[0]
    res = coll.query(
        query_embeddings=[q_emb.tolist()],
        n_results=k,
        include=["metadatas", "documents", "distances"]
    )
    hits = []
    for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
        # For cosine similarity, Chroma returns distance = 1-similarity, so we can recover similarity as 1-distance
        similarity = 1.0 - dist
        hits.append({
            'chunk_text': doc,
            'metadata': meta,
            'score': similarity
        })
    return hits

# --------------------------
# EVALUATION UTILITIES
# --------------------------
def recall_at_k(true_chunk_ids:List[str], candidate_chunk_ids:List[str], k:int) -> float:
    found = any(tid in candidate_chunk_ids[:k] for tid in true_chunk_ids)
    return float(found)

def spot_check_examples():
    print("--- Spot check retrieval quality ---")
    typical_queries = [
        ("How do I reset my account password?", "account"),
        ("I was double charged on my invoice", "billing"),
        ("My device won't connect to WiFi", "technical"),
        ("Can I get a refund for last month?", "billing"),
        ("Change my account email", "account"),
        ("Fix priority outage for premium clients", "technical"),
    ]
    for query, expected_cat in typical_queries:
        print(f"\nQuery: {query}")
        hits = retrieve_top_k_chunks(query, k=5)
        for i, hit in enumerate(hits):
            print(f"  Rank {i+1}, Score={hit['score']:.3f}, Category={hit['metadata']['category']}, Date={hit['metadata']['date']}")
            snippet = hit['chunk_text'][:90].replace('\n',' ')
            print(f"     Text: {snippet}")
        cats = [hit['metadata']['category'] for hit in hits]
        print(f"  Categories retrieved: {cats}  (expected: {expected_cat})")

# Example recall@5 evaluation (mocked ground-truth, in practice collect this for real queries)
def run_mock_recall_evaluation():
    queries_and_gt = [
        {"query": "I need to restore my lost password", "gt_doc_ids": [f"doc_{5}"]},
        {"query": "Change credit card for my account", "gt_doc_ids": [f"doc_{9}"]},
        {"query": "Immediate fix for network issue", "gt_doc_ids": [f"doc_{7}"]},
    ]
    recalls = []
    for q in queries_and_gt:
        hits = retrieve_top_k_chunks(q['query'], k=5)
        candidate_chunk_ids = [h['metadata']['doc_id'] for h in hits]
        recall = recall_at_k(q['gt_doc_ids'], candidate_chunk_ids, 5)
        recalls.append(recall)
        print(f"Query='{q['query']}' - Recall@5: {recall}")
    if recalls:
        print(f"Mean recall@5: {np.mean(recalls):.3f}")


# --------------------------
# MAIN SCRIPT ENTRY POINT
# --------------------------
if __name__ == "__main__":
    # Ingestion
    print("Loading documents...")
    docs = load_documents()
    print("Ingesting documents...")
    ingest_documents(docs)

    # Basic spot checks
    spot_check_examples()
    
    run_mock_recall_evaluation()
    print("Done!")
