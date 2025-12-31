import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------
# Load TXT
# ---------------------------------------------------------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------
# Split into chapters
# ---------------------------------------------------------
def split_into_chapters(book_text: str):
    chapter_titles = re.findall(r"(CHAPTER\s+\d+.*)", book_text)
    parts = re.split(r"CHAPTER\s+\d+.*", book_text)

    chapters = []
    for i, content in enumerate(parts[1:]):
        title = chapter_titles[i].strip()
        chapters.append({
            "title": title,
            "content": content.strip()
        })

    return chapters


# ---------------------------------------------------------
# Split into points
# ---------------------------------------------------------
def split_into_points(chapter_text: str):
    point_numbers = re.findall(r"\n\s*(\d+)\.\s+", chapter_text)
    parts = re.split(r"\n\s*\d+\.\s+", chapter_text)

    points = []
    for i, content in enumerate(parts[1:]):
        number = point_numbers[i]
        points.append({
            "point_number": number,
            "content": content.strip()
        })

    return points


# ---------------------------------------------------------
# MAIN INGESTION PIPELINE
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n📚 Starting ingestion...")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to ChromaDB
    chroma = chromadb.PersistentClient(path="vectorstore")
    try:
        collection = chroma.get_collection("books")
    except:
        collection = chroma.create_collection("books")

    books_folder = Path("books")
    txt_files = list(books_folder.glob("*.txt"))

    if not txt_files:
        print("No TXT files found in books/")
        exit()

    for txt_file in txt_files:
        print(f"\n📘 Processing book: {txt_file.name}")

        book_title = txt_file.stem
        book_text = load_txt(str(txt_file))

        chapters = split_into_chapters(book_text)
        print(f"  → Chapters found: {len(chapters)}")

        for chapter in chapters:
            chapter_title = chapter["title"]
            points = split_into_points(chapter["content"])

            print(f"    → {chapter_title}: {len(points)} points")

            for point in points:
                point_text = point["content"]
                point_number = point["point_number"]

                # Embed the point
                embedding = model.encode(point_text).tolist()

                # Create a unique ID
                uid = f"{book_title}_{chapter_title}_{point_number}"

                # Store in ChromaDB
                collection.add(
                    ids=[uid],
                    documents=[point_text],
                    embeddings=[embedding],
                    metadatas=[{
                        "book": book_title,
                        "chapter": chapter_title,
                        "point": point_number
                    }]
                )

    print("\n✅ Ingestion complete!")
