import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq


# ---------------------------------------------------------
# Load embedding model once
# ---------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# Connect to ChromaDB
# ---------------------------------------------------------
chroma = chromadb.PersistentClient(path="vectorstore")
collection = chroma.get_collection("books")


# ---------------------------------------------------------
# Initialize Groq client
# ---------------------------------------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------
def answer_question(question: str, selected_book: str, n_results: int = 5):
    """
    Retrieve ONLY from the selected book.
    Answer ONLY from retrieved text.
    """

    # 1. Embed the question
    q_embed = model.encode(question).tolist()

    # 2. Query ONLY the selected book
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=n_results,
        where={"book": selected_book},   # STRICT FILTER
        include=["documents", "metadatas"]
    )

    # If no results found
    if not results["documents"] or not results["documents"][0]:
        return "I don't know. The answer is not in the selected book."

    # 3. Combine retrieved chunks
    context = "\n\n".join(results["documents"][0])

    # 4. Build strict prompt
    prompt = f"""
You must answer ONLY using the provided book excerpts.
If the answer is not in the excerpts, say: "I don't know."

BOOK: {selected_book}

USER QUESTION:
{question}

RELEVANT EXCERPTS:
{context}
"""

    # 5. Call Groq
    completion = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You must answer ONLY using the provided excerpts. Never invent book titles or content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return completion.choices[0].message.content


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
if __name__ == "__main__":
    question = "What does the book say about humility?"
    selected_book = "The Art of Leadership - Dag Heward-Mills"

    print("\nAnswer:\n")
    print(answer_question(question, selected_book))
