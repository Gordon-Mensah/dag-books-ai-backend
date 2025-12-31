from sentence_transformers import SentenceTransformer
import chromadb

def retrieve(query: str, n_results: int = 5):
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to ChromaDB
    chroma = chromadb.PersistentClient(path="vectorstore")
    collection = chroma.get_collection("books")

    # Embed the question
    q_embed = model.encode(query).tolist()

    # Search the vector DB
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=n_results
    )

    return results

if __name__ == "__main__":
    question = "What does the book say about humility"
    results = retrieve(question)

    print("\nTop results:\n")
    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:\n{doc[:500]}")
        print("-" * 50)
