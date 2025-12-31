import chromadb

chroma = chromadb.PersistentClient(path="vectorstore")
collection = chroma.get_collection("books")

data = collection.get(include=["metadatas"])

books = set()

for m in data["metadatas"]:
    if "book" in m:
        books.add(m["book"])

print("\n=== UNIQUE BOOKS IN VECTORSTORE ===")
for b in sorted(books):
    print("-", b)
