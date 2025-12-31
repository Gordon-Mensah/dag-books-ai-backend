import json
import os
from pathlib import Path

import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")

client = Groq(api_key=GROQ_API_KEY)

BOOKS_DIR = Path("books")  # put your .txt extracted book content here
OUTPUT_PATH = Path("data") / "book_chunks_groq.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model="nomic-embed-text",  # or Groq embedding model you use
        input=text,
    )
    return response.data[0].embedding


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + " " + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def main():
    all_chunks = []

    for file in BOOKS_DIR.glob("*.txt"):
        print(f"Processing {file.name}...")
        text = file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            all_chunks.append(
                {
                    "book_file": file.name,
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": emb,
                }
            )

    print(f"Total chunks: {len(all_chunks)}")
    OUTPUT_PATH.write_text(json.dumps(all_chunks), encoding="utf-8")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
