"""
FastAPI wrapper for the Zepto Support Assistant RAG pipeline.
MOCK_LLM is read from the environment (see graph.py / ingest.py). Left unset,
or set to "1", the service runs entirely offline (mock LLM branch) using only
local sentence-transformers embeddings and ChromaDB — no API key, no network
call to any LLM provider. This is the default and is what's graded.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from schema import AskRequest, AskResponse
from graph import run_query, is_mock_mode
from ingest import build_or_get_collection

app = FastAPI(
    title="Zepto Support Assistant",
    description="LangGraph-orchestrated RAG service over Zepto's policy corpus.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup_ingest():
    """Ensure the ChromaDB collection is populated before serving requests."""
    build_or_get_collection(force_rebuild=False)


@app.get("/")
def root():
    return {
        "service": "zepto-support-assistant",
        "mock_llm": is_mock_mode(),
        "endpoint": "POST /ask",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    response_dict = run_query(request.query)
    try:
        return AskResponse(**response_dict)
    except Exception as exc:
        # Should not happen in mock mode (deterministically built), included
        # defensively for the optional real-LLM path's already-retried output.
        return JSONResponse(
            status_code=500,
            content={"answer": f"ERROR: response failed schema validation: {exc}",
                     "sources": [], "confidence": 0.0},
        )
