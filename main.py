from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
print("DEBUG KEY:", os.getenv("GROQ_API_KEY"))



# ---------------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Connect to ChromaDB (your existing vectorstore)
# ---------------------------------------------------------
chroma = chromadb.PersistentClient(path="vectorstore")
collection = chroma.get_collection("books")

# ---------------------------------------------------------
# Groq client
# ---------------------------------------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    book: str
    chapter: str | None = None
    n_results: int = 5

class BookFinderRequest(BaseModel):
    query: str
    n_results: int = 10

# ---------------------------------------------------------
# Helper: embed using Chroma’s stored embedding function
# ---------------------------------------------------------
def embed(text: str):
    return collection._embedding_function(text)

# ---------------------------------------------------------
# Endpoint: list all books
# ---------------------------------------------------------
@app.get("/books")
def list_books():
    data = collection.get(include=["metadatas"])
    books = sorted({m["book"] for m in data["metadatas"]})
    return {"books": books}

# ---------------------------------------------------------
# Endpoint: list chapters for a book
# ---------------------------------------------------------
@app.get("/chapters/{book}")
def list_chapters(book: str):
    data = collection.get(where={"book": book}, include=["metadatas"])
    chapters = sorted({m["chapter"] for m in data["metadatas"]})
    return {"chapters": chapters}

# ---------------------------------------------------------
# Book Finder Mode — search across ALL books
# ---------------------------------------------------------
@app.post("/find-book")
def find_book(req: BookFinderRequest):

    q_embed = embed(req.query)

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=req.n_results,
        include=["documents", "metadatas"]
    )

    matches = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        matches.append({
            "book": meta["book"],
            "chapter": meta.get("chapter", None),
            "excerpt": doc[:300] + "..."
        })

    return {"matches": matches}

# ---------------------------------------------------------
# Streaming generator for Groq
# ---------------------------------------------------------
def stream_groq_response(prompt: str):
    def generate():
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You must answer ONLY using the provided excerpts. Never invent book titles or content."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    return StreamingResponse(generate(), media_type="text/plain")

# ---------------------------------------------------------
# Endpoint: ask a question (STREAMING)
# ---------------------------------------------------------
@app.post("/ask")
def ask_question(req: AskRequest):

    q_embed = embed(req.question)

    where_filter = {"book": req.book}
    if req.chapter:
        where_filter["chapter"] = req.chapter

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=req.n_results,
        where=where_filter,
        include=["documents"]
    )

    if not results["documents"] or not results["documents"][0]:
        return StreamingResponse(
            iter(["I don't know. The answer is not in the selected book or chapter."]),
            media_type="text/plain"
        )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You must answer ONLY using the provided book excerpts.
If the answer is not in the excerpts, say: "I don't know."

BOOK: {req.book}
CHAPTER: {req.chapter}

USER QUESTION:
{req.question}

RELEVANT EXCERPTS:
{context}
"""

    return stream_groq_response(prompt)
