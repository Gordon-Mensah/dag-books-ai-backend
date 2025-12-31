from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int = 800, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return chunks

if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up ChromaDB...")
    chroma = chromadb.PersistentClient(path="vectorstore")
    collection = chroma.get_or_create_collection("books")

    books_folder = Path("books")
    txt_files = list(books_folder.glob("*.txt"))

    if not txt_files:
        print("No TXT files found.")
    else:
        for txt_file in txt_files:
            print(f"\nProcessing: {txt_file.name}")
            content = load_txt(str(txt_file))
            chunks = chunk_text(content)

            print(f"Embedding {len(chunks)} chunks...")

            embeddings = model.encode(chunks).tolist()

            ids = [f"{txt_file.stem}-{i}" for i in range(len(chunks))]
            metadata = [{"book": txt_file.stem}] * len(chunks)

            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )

            print(f"Stored {len(chunks)} chunks in vector DB.")
