import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
from config import (
    ENDEE_AUTH_TOKEN, ENDEE_INDEX_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    CHUNK_SIZE, CHUNK_OVERLAP
)

embed_model = SentenceTransformer(EMBEDDING_MODEL)

def get_client():
    return Endee(ENDEE_AUTH_TOKEN if ENDEE_AUTH_TOKEN else "")

def create_index(client):
    try:
        client.create_index(
            name=ENDEE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type="cosine",
            precision=Precision.INT8
        )
        print(f"[Endee] Index '{ENDEE_INDEX_NAME}' created.")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print(f"[Endee] Index '{ENDEE_INDEX_NAME}' already exists.")
        else:
            raise e

def extract_text(file_path: str) -> str:
    """Extract text from .txt, .md, .pdf, or .docx files."""
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    elif ext == ".pdf":
        import fitz   # pymupdf
        doc  = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    elif ext == ".docx":
        from docx import Document
        doc  = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]

def load_docs(folder: str) -> list[dict]:
    docs = []
    for p in Path(folder).glob("**/*"):
        if p.suffix.lower() in (".txt", ".md", ".pdf", ".docx"):
            try:
                text = extract_text(str(p))
                docs.append({"text": text, "source": str(p)})
            except Exception as e:
                print(f"[Ingest] Skipping {p}: {e}")
    return docs

def upsert(client, chunks: list[str], source: str):
    index      = client.get_index(name=ENDEE_INDEX_NAME)
    embeddings = embed_model.encode(chunks, show_progress_bar=False).tolist()
    vectors    = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id":     str(uuid.uuid4()),
            "vector": emb,
            "meta":   {
                "text":        chunk,
                "source":      source,
                "chunk_index": i
            },
            "filter": {"source": source}
        })
    index.upsert(vectors)
    print(f"[Ingest] {len(vectors)} chunks from '{source}'")

def ingest_folder(folder: str = "./docs"):
    client = get_client()
    create_index(client)
    docs = load_docs(folder)
    if not docs:
        print(f"No supported files found in '{folder}'.")
        return
    for doc in docs:
        chunks = chunk_text(doc["text"])
        upsert(client, chunks, doc["source"])
    print(f"\n[Ingest] Done — {len(docs)} document(s) ingested.")

if __name__ == "__main__":
    ingest_folder("./docs")