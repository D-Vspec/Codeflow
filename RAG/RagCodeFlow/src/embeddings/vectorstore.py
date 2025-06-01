# src/embeddings/vectorstore.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def create_embeddings(chunks, model_name):
    """
    Create embeddings for document chunks
    
    Args:
        chunks (list): List of dictionaries containing path and content chunks
        model_name (str): Name of the sentence transformer model to use
        
    Returns:
        list: List of dictionaries with path, content, and embeddings
    """
    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    vector_data = []
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"Processing chunk {i}/{len(chunks)}")
        embedding = embedding_model.encode(chunk["content"])
        vector_data.append({
            "path": chunk["path"],
            "content": chunk["content"],
            "embedding": embedding
        })
    return vector_data

def build_faiss_index(vector_data):
    """
    Build a FAISS index for fast similarity search
    
    Args:
        vector_data (list): List of dictionaries with path, content, and embeddings
        
    Returns:
        tuple: (faiss.Index, numpy.array) The FAISS index and the vector matrix
    """
    print("Building FAISS index...")
    dimension = len(vector_data[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)
    vector_matrix = np.array([item["embedding"] for item in vector_data]).astype("float32")
    index.add(vector_matrix)
    return index, vector_matrix

def search_similar_chunks(index, query_vector, vector_data, k=15):
    """
    Search for chunks similar to the query
    
    Args:
        index (faiss.Index): The FAISS index
        query_vector (numpy.array): The embedding of the query
        vector_data (list): List of dictionaries with path, content, and embeddings
        k (int): Number of similar chunks to retrieve
        
    Returns:
        list: List of strings containing relevant chunks
    """
    distances, indices = index.search(np.array([query_vector]).astype("float32"), k)
    retrieved_chunks = []
    
    # Prioritize actual source code files over build artifacts
    source_paths = []
    other_paths = []
    
    for i in indices[0]:
        path = vector_data[i]['path']
        content = vector_data[i]['content']
        chunk = f"File: {path}\n{content}"
        
        # Prioritize src directory files
        if "/src/" in path and not any(build_dir in path for build_dir in [".cxx", "build"]):
            source_paths.append(chunk)
        else:
            other_paths.append(chunk)
    
    # Return source code files first, then other files if needed
    retrieved_chunks = source_paths + other_paths
    return retrieved_chunks[:k]  # Still limit to k chunks total
