from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_chunks(query, index, df_chunks, embedder, top_k=5):
    try:
        query_embedding = embedder.encode([query], show_progress_bar=False)
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
        retrieved_chunks = df_chunks.iloc[indices[0]][['complaint_id', 'product', 'chunk_idx', 'chunk_text']].to_dict('records')
        return retrieved_chunks, distances[0]
    except Exception as e:
        logging.error(f"Error retrieving chunks for query '{query}': {e}")
        return [], []

def rag_pipeline(query, index, df_chunks, embedder, llm, top_k=5):
    try:
        # Retrieve chunks
        chunks, distances = retrieve_chunks(query, index, df_chunks, embedder, top_k)
        if not chunks:
            return "No relevant complaints found.", [], []
        
        # Build prompt in Zephyr format
        system_prompt = "You are a financial complaint analysis assistant."
        context = "\n".join([f"Complaint (Product: {chunk['product']}): {chunk['chunk_text']}" for chunk in chunks])
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{query}\n<|retrieved|>\n{context}"
        
        # Generate response
        response = llm(full_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        return response[0]["generated_text"], chunks, distances
    except Exception as e:
        logging.error(f"Error in RAG pipeline for query '{query}': {e}")
        return "Error processing query.", [], []
    
    
    
    
    # Test RAG pipeline
sample_queries = [
    "Why are people unhappy with BNPL?",
    "What are common issues with Credit Card fraud?",
    "Why do Savings Account complaints happen?",
    "What issues do people face with Money Transfers?",
    "Why are Personal Loan complaints common?"
]
results = []
for query in sample_queries:
    response, chunks, distances = rag_pipeline(query, index, df_chunks, embedder, llm)
    print(f"\nQuery: {query}")
    print("Response:", response)
    print("Retrieved Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (Product: {chunk['product']}, Distance: {distances[i]:.4f}):")
        print(chunk['chunk_text'])
    results.append({
        'query': query,
        'response': response,
        'retrieved_chunks': chunks,
        'distances': distances.tolist()
    })