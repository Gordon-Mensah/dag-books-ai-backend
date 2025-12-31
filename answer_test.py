import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma = chromadb.PersistentClient(path="vectorstore")
collection = chroma.get_collection("books")

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question: str, n_results: int = 5):
    # Embed the question
    q_embed = model.encode(question).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=n_results
    )

    # Combine chunks into context
    context = "\n\n".join(results["documents"][0])

    # Build the prompt
    prompt = f"""
You are an assistant that answers questions ONLY using the provided book excerpts,you say the name of the book,the chapter and the point.
If the answer is not in the excerpts, say you don't know.

User question: {question}

Relevant excerpts:
{context}
"""

    # Call Groq (OpenAI-style API)
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    question = "What does the book say about humility?"
    print("\nAnswer:\n")
    print(answer_question(question))
