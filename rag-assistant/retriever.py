from sentence_transformers import SentenceTransformer
from endee import Endee
from config import ENDEE_AUTH_TOKEN, ENDEE_INDEX_NAME, EMBEDDING_MODEL, TOP_K

embed_model = SentenceTransformer(EMBEDDING_MODEL)

def get_client():
    return Endee(ENDEE_AUTH_TOKEN if ENDEE_AUTH_TOKEN else "")

def retrieve(
    query:           str,
    top_k:           int        = TOP_K,
    filter_source:   str        = None,
    allowed_sources: list[str]  = None
) -> list[dict]:
    client = get_client()
    index  = client.get_index(name=ENDEE_INDEX_NAME)
    vector = embed_model.encode([query])[0].tolist()

    results = index.query(vector=vector, top_k=top_k * 3)

    output = []
    for hit in results:
        meta = hit.get("meta", {}) if isinstance(hit, dict) else {}
        source = meta.get("source", "")

        # Filter to only uploaded files
        if allowed_sources and source not in allowed_sources:
            continue

        # Filter by specific source if requested
        if filter_source and source != filter_source:
            continue

        output.append({
            "text":        meta.get("text", ""),
            "source":      source,
            "chunk_index": meta.get("chunk_index", 0),
            "score":       round(float(hit.get("similarity", 0)), 4)
        })

        if len(output) >= top_k:
            break

    return output