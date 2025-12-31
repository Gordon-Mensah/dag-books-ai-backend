import os
import chromadb
from chromadb.utils import embedding_functions

BOOKS_DIR = "books"

client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_or_create_collection(
    name="books",
    metadata={"hnsw:space": "cosine"}
)

embedder = embedding_functions.DefaultEmbeddingFunction()

for book in os.listdir(BOOKS_DIR):
    book_path = os.path.join(BOOKS_DIR, book)

    if not os.path.isdir(book_path):
        continue

    for chapter_file in os.listdir(book_path):
        if not chapter_file.endswith(".txt"):
            continue

        chapter_path = os.path.join(book_path, chapter_file)

        with open(chapter_path, "r", encoding="utf-8") as f:
            text = f.read()

        chapter_name = chapter_file.replace(".txt", "")

        collection.add(
            documents=[text],
            metadatas=[{
                "book": book,
                "chapter": chapter_name
            }],
            ids=[f"{book}-{chapter_name}"]
        )

print("Ingestion complete!")
