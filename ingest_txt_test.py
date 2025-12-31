import re
from pathlib import Path


# ---------------------------------------------------------
# STEP 1 — Load a TXT file
# ---------------------------------------------------------
def load_txt(path: str) -> str:
    """Read all text from a TXT file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------
# STEP 2 — Split book into chapters
# ---------------------------------------------------------
def split_into_chapters(book_text: str):
    """
    Splits a book into chapters based on lines starting with 'CHAPTER'.
    Returns a list of {title, content}.
    """

    # Find chapter titles like: CHAPTER 1, CHAPTER 2, etc.
    chapter_titles = re.findall(r"(CHAPTER\s+\d+.*)", book_text)

    # Split the text at each chapter title
    parts = re.split(r"CHAPTER\s+\d+.*", book_text)

    chapters = []

    # parts[0] is everything before CHAPTER 1 → ignore it
    for i, content in enumerate(parts[1:]):
        title = chapter_titles[i].strip()
        chapters.append({
            "title": title,
            "content": content.strip()
        })

    return chapters


# ---------------------------------------------------------
# STEP 3 — Split chapter into numbered points
# ---------------------------------------------------------
def split_into_points(chapter_text: str):
    """
    Splits a chapter into points based on numbered lines like '1.' '2.' etc.
    Returns a list of {point_number, content}.
    """

    # Find point numbers
    point_numbers = re.findall(r"\n\s*(\d+)\.\s+", chapter_text)

    # Split content at each numbered point
    parts = re.split(r"\n\s*\d+\.\s+", chapter_text)

    points = []

    # parts[0] is text before point 1 → ignore it
    for i, content in enumerate(parts[1:]):
        number = point_numbers[i]
        points.append({
            "point_number": number,
            "content": content.strip()
        })

    return points


# ---------------------------------------------------------
# STEP 4 — Process a full book
# ---------------------------------------------------------
def process_book(book_title: str, book_text: str):
    """
    Splits a book into chapters and points.
    Returns a structured list.
    """

    chapters = split_into_chapters(book_text)

    processed = []

    for chapter in chapters:
        points = split_into_points(chapter["content"])

        processed.append({
            "book": book_title,
            "chapter_title": chapter["title"],
            "points": points
        })

    return processed


# ---------------------------------------------------------
# STEP 5 — MAIN TEST SCRIPT
# ---------------------------------------------------------
if __name__ == "__main__":
    books_folder = Path("books")
    txt_files = list(books_folder.glob("*.txt"))

    if not txt_files:
        print("No TXT files found in the books/ folder.")
        exit()

    for txt_file in txt_files:
        print("\n" + "=" * 80)
        print(f"📘 BOOK: {txt_file.name}")
        print("=" * 80)

        # Load book
        content = load_txt(str(txt_file))

        # Process book
        structured = process_book(txt_file.stem, content)

        print(f"Total chapters found: {len(structured)}")

        # Preview first chapter + first 3 points
        if structured:
            first_chapter = structured[0]
            print("\n--- FIRST CHAPTER ---")
            print(first_chapter["chapter_title"])

            print("\n--- FIRST 3 POINTS ---")
            for p in first_chapter["points"][:3]:
                print(f"\nPoint {p['point_number']}:")
                print(p["content"][:300], "...")
        else:
            print("No chapters detected.")

        print("\n" + "-" * 80)
