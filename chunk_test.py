from pathlib import Path

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks.
    size = length of each chunk
    overlap = repeated text between chunks (helps context)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap

    return chunks

if __name__ == "__main__":
    books_folder = Path("books")
    txt_files = list(books_folder.glob("*.txt"))

    if not txt_files:
        print("No TXT files found.")
    else:
        for txt_file in txt_files:
            print(f"\nProcessing: {txt_file.name}")
            content = load_txt(str(txt_file))
            chunks = chunk_text(content)

            print("Total characters:", len(content))
            print("Number of chunks:", len(chunks))
            print("\nPreview of first chunk:\n")
            print(chunks[0][:500])
            print("-" * 50)
