import numpy as np
import faiss

def embed_chunks(chunks, model):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def retrieve_chunks(query, embed_model, index, chunks, k=3):
    query_vec = embed_model.encode([query])
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]