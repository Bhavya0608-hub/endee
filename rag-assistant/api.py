import os, uuid, shutil, traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from retriever import retrieve
from llm import generate_answer
from ingest import extract_text, chunk_text, upsert, create_index, get_client

app = FastAPI(title="RAG Assistant — Endee + Groq Qwen", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# track uploaded files in memory
uploaded_files_registry: list[dict] = []

class QueryRequest(BaseModel):
    question:      str
    top_k:         int        = 5
    filter_source: str | None = None

class ChunkResult(BaseModel):
    text:   str
    source: str
    score:  float

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[str]
    chunks:   list[ChunkResult]

@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-assistant"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = (".txt", ".md", ".pdf", ".docx")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"
        )
    try:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract text based on file type
        text   = extract_text(save_path)
        client = get_client()
        create_index(client)
        chunks = chunk_text(text)
        upsert(client, chunks, save_path)

        # Register file
        uploaded_files_registry.append({
            "filename": file.filename,
            "path":     save_path,
            "chunks":   len(chunks),
            "ext":      ext
        })

        return {
            "status":   "ingested",
            "filename": file.filename,
            "chunks":   len(chunks),
            "ext":      ext
        }
    except Exception as e:
        print(f"[UPLOAD ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        # Only search in uploaded docs
        uploaded_paths = [f["path"] for f in uploaded_files_registry]

        if not uploaded_paths:
            return QueryResponse(
                question = req.question,
                answer   = "Please upload a document first before asking questions.",
                sources  = [],
                chunks   = []
            )

        chunks = retrieve(
            req.question,
            top_k=req.top_k,
            filter_source=req.filter_source,
            allowed_sources=uploaded_paths
        )

        print(f"[API] Retrieved {len(chunks)} chunks from uploaded docs")

        if not chunks:
            return QueryResponse(
                question = req.question,
                answer   = "No relevant information found in the uploaded documents.",
                sources  = [],
                chunks   = []
            )

        answer  = generate_answer(req.question, chunks)
        sources = sorted(set(c["source"] for c in chunks))

        return QueryResponse(
            question = req.question,
            answer   = answer,
            sources  = sources,
            chunks   = [ChunkResult(
                text   = c["text"],
                source = c["source"],
                score  = c["score"]
            ) for c in chunks]
        )
    except Exception as e:
        print(f"[API ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/uploaded-files")
def get_uploaded_files():
    return {"files": uploaded_files_registry}

@app.get("/sources")
def list_sources():
    return {"files": uploaded_files_registry}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")