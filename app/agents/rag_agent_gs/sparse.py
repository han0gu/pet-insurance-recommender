# concise sparse uploader (HTTP mode) - written for local Qdrant at http://localhost:6333
from typing import List, Dict, Any
import math
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

# import pre-split chunks (assumes tc_chunk.py creates `chunks` list)
from tc_chunk import chunks

# connect to Qdrant HTTP API (Docker / remote) on port 6333
client = QdrantClient(url="http://localhost:6333")  # connect via HTTP

COLLECTION = "meritz_sparse"  # collection name to create / overwrite

# build TF-IDF sparse vectors (very small and readable implementation)
def build_tfidf(chunks: List[Dict[str, Any]]):
    vocab: Dict[str, int] = {}
    df = defaultdict(int)
    docs_tokens: List[List[str]] = []

    # collect tokens and document frequencies
    for c in chunks:
        tokens = c["text"].lower().split()
        docs_tokens.append(tokens)
        for t in set(tokens):
            df[t] += 1
            if t not in vocab:
                vocab[t] = len(vocab)

    N = len(chunks)
    idf = {t: math.log(N / (1 + df[t])) for t in df}

    sparse_vectors: Dict[str, List[tuple]] = {}
    for i, tokens in enumerate(docs_tokens):
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        vec = [(vocab[t], tf[t] * idf[t]) for t in tf]
        sparse_vectors[chunks[i]["id"]] = vec

    return sparse_vectors, vocab

# create collection configured for sparse vectors (no dense vectors required)
def create_collection(name: str):
    # delete if exists
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass
    # create minimal collection with sparse vector config
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

# upload points: store sparse indices/values in payload and use dummy dense vector
def upload(name: str, chunks: List[Dict[str, Any]], sparse_vectors: Dict[str, List[tuple]]):
    points = []
    for idx, chunk in enumerate(chunks, start=1):
        vec = sparse_vectors.get(chunk["id"], [])
        indices = [int(i) for i, _ in vec]
        values = [float(v) for _, v in vec]
        points.append(
            {
                "id": idx,
                "vector": [0.0],  # dummy dense vector so collection accepts point
                "payload": {
                    "chunk_id": chunk["id"],
                    "page": chunk.get("page"),
                    "text": chunk.get("text"),
                    "sparse_indices": indices,
                    "sparse_values": values,
                },
            }
        )
    client.upsert(collection_name=name, points=points)

# simple interactive inspector to view one chunk's sparse vector and vocab size
def inspect_interactive(sparse_vectors: Dict[str, List[tuple]], vocab: Dict[str, int], chunks: List[Dict[str, Any]]):
    print(f"vocab size (sparse dimension): {len(vocab)}")
    print("Type 'list' to show first 20 chunks, 'q' to quit.")
    while True:
        s = input("select chunk (pX_cY or number): ").strip()
        if not s:
            continue
        if s.lower() in ("q", "quit"):
            break
        if s.lower() == "list":
            for i, c in enumerate(chunks[:20], start=1):
                print(i, c["id"], f"(page {c.get('page')})")
            continue
        # number -> index
        if s.isdigit():
            n = int(s)
            if 1 <= n <= len(chunks):
                cid = chunks[n - 1]["id"]
            else:
                print("number out of range")
                continue
        else:
            cid = s
        vec = sparse_vectors.get(cid)
        if vec is None:
            print("chunk id not found")
            continue
        print(f"chunk={cid} non-zero={len(vec)}")
        # print top 30 by value
        for i, v in sorted(vec, key=lambda x: -x[1])[:30]:
            print(i, f": {v:.6f}")
        # show snippet
        txt = next((c["text"] for c in chunks if c["id"] == cid), "")
        print("text:", (txt[:300] + "...") if len(txt) > 300 else txt)


if __name__ == "__main__":
    # 1) build tf-idf sparse vectors
    sparse_vectors, vocab = build_tfidf(chunks)
    # 2) create collection (HTTP mode)
    create_collection(COLLECTION)
    # 3) upload points
    upload(COLLECTION, chunks, sparse_vectors)
    # 4) print quick instructions to open dashboard
    print("uploaded -> open in browser: http://localhost:6333/dashboard#")
    print("api check: curl http://localhost:6333/collections")
    # 5) interactive inspect locally
    inspect_interactive(sparse_vectors, vocab, chunks)
