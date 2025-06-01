from sentence_transformers import SentenceTransformer

def process_query(query, model_name, index, vector_data):
    """
    Process a query against the codebase
    
    Args:
        query (str): The user's query about the codebase
        model_name (str): Name of the sentence transformer model
        index (faiss.Index): The FAISS index
        vector_data (list): List of dictionaries with path, content, and embeddings
        
    Returns:
        tuple: (numpy.array, list) The query vector and retrieved chunks
    """
    # Create embedding for the query
    embedding_model = SentenceTransformer(model_name)
    query_vector = embedding_model.encode(query)
    
    # Get relevant context
    from src.embeddings.vectorstore import search_similar_chunks
    retrieved_chunks = search_similar_chunks(index, query_vector, vector_data)
    
    return query_vector, retrieved_chunks
