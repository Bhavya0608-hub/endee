from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a precise document assistant.
Your job is to answer questions based ONLY on the provided document chunks.

Rules:
- Answer directly and concisely. No thinking out loud.
- Use only information from the provided context chunks.
- If the answer is clearly in the context, give it directly.
- If the answer is NOT in the context, say exactly: "This information is not available in the uploaded documents."
- Never guess or use outside knowledge.
- Do not include <think> tags or internal reasoning in your response.
- Keep answers focused and to the point."""

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        filename = c['source'].replace('\\', '/').split('/')[-1]
        parts.append(f"[Chunk {i} | File: {filename}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)

def clean_response(text: str) -> str:
    """Remove <think>...</think> blocks if model includes them."""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def generate_answer(query: str, chunks: list[dict]) -> str:
    context  = build_context(chunks)
    user_msg = f"""Here are the relevant document chunks:

{context}

---

Question: {query}

Answer based only on the above chunks:"""

    response = client.chat.completions.create(
        model    = "qwen/qwen3-32b",
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        temperature = 0.1,
        max_tokens  = 512
    )
    raw = response.choices[0].message.content
    return clean_response(raw)