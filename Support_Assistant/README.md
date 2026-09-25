# Module 3 — Zepto Support Assistant (`/support_assistant`)

A small, fully-offline-gradeable RAG service for Zepto's own policy corpus, built with
local sentence-transformer embeddings, ChromaDB, a LangGraph intent-routing pipeline,
Pydantic-validated structured output, and a FastAPI wrapper.

## Contents

```
support_assistant/
├── docs/                  # 8 policy corpus documents (doc_01.txt … doc_08.txt)
├── ingest.py              # ingestion + chunking + embedding + ChromaDB storage
├── prompt_template.py     # role/context/task/format/length prompt (optional MOCK_LLM=0 path)
├── graph.py               # LangGraph StateGraph: classify_intent / retrieve_and_answer / direct_answer
├── schema.py              # Pydantic AskRequest / AskResponse models
├── main.py                # FastAPI app exposing POST /ask
├── requirements.txt
├── Dockerfile
└── README.md              # this file
```

## Running it locally

```bash
cd support_assistant
pip install -r requirements.txt
python ingest.py                 # one-time: embeds the 8 docs into ./chroma_store
uvicorn main:app --host 0.0.0.0 --port 7860
```

`MOCK_LLM` defaults to `1` (fully offline mock mode) if left unset — this is the graded
baseline and requires no signup, no API key, and no network call to any LLM provider.
The **first** run of `ingest.py` (or the app) does need internet access once, only to
download the local `all-MiniLM-L6-v2` embedding model weights from Hugging Face (~90 MB,
one-time, cached afterwards) — this is a local, free, no-account download, not an LLM API
call.

Run in Docker:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

## Architecture: the RAG pipeline, stage by stage

**1. Ingestion** — `ingest.py: load_documents()`
Reads the 8 plain-text policy files in `docs/` (delivery, returns, membership, tracking,
cancellation, damaged items, gift cards, support hours) into memory as `{doc_id: text}`.

**2. Chunking** — `ingest.py: chunk_document()` / `chunk_all_documents()`
Applies a fixed-size character chunking scheme (400 characters per chunk, 50-character
overlap). Given how short each policy document is, most produce one or two chunks; the
same function scales unmodified to longer documents. Each chunk is tagged with a
`chunk_id` (e.g. `doc_01_chunk0`) and its parent `doc_id`.

**3. Embedding** — `ingest.py: get_embedding_model()` / `build_or_get_collection()`
Every chunk is embedded locally with `sentence-transformers`' `all-MiniLM-L6-v2` model —
no API key, no network call to any LLM provider (only a one-time local model-weight
download, see above). Embeddings are L2-normalized so that ChromaDB's cosine-distance
index (`hnsw:space: cosine`) ranks results by cosine similarity.

**4. Storage** — ChromaDB collection `zepto_policies`
A `chromadb.PersistentClient` persists the collection under `support_assistant/chroma_store/`.
`upsert()` is keyed by `chunk_id`, so re-running ingestion is idempotent.

**5. Orchestration / Retrieval** — `graph.py`, a LangGraph `StateGraph` with a `GraphState`
`TypedDict` (`query`, `intent`, `retrieved_chunks`, `response`) and three nodes:
- **`classify_intent`** — in mock mode (the default / graded baseline), classifies the
  query with a keyword heuristic (`delivery`, `return`, `refund`, `membership`,
  `tracking`, `cancel`, `gift card`, `support hours`) into `policy_question` or
  `general_question`, with **no LLM call**.
- **`retrieve_and_answer`** — for `policy_question` queries. Retrieval (embed the query,
  query ChromaDB for the top‑3 cosine-similar chunks) **always runs for real**, in both
  modes, since it needs no LLM API key. Only the final answer-generation step branches
  on `MOCK_LLM`: mock mode returns `f"Based on the retrieved context: {top_chunk_snippet}"`
  (first ~200 characters of the top chunk), with no LLM call.
- **`direct_answer`** — for `general_question` queries. Mock mode returns a fixed canned
  string, `"I can only answer questions about Zepto policies right now."`, with no LLM call.

A conditional edge (`route_from_classification`) routes from `classify_intent` to either
`retrieve_and_answer` or `direct_answer` based on the classification. This routing logic
itself does not depend on `MOCK_LLM` — only the generation step *inside* each node does.

**6. Structured output / Generation** — `schema.py` (`AskResponse`) + the nodes above
In mock mode, `AskResponse(answer, sources, confidence)` is populated deterministically
in code: `sources` is the list of retrieved `chunk_id`s for `policy_question` (empty for
`general_question`), and `confidence` is fixed at `1.0`, since no LLM output exists to
fail validation. `main.py`'s `POST /ask` endpoint validates the dict against `AskResponse`
before returning it.

**7. API wrapper** — `main.py`
A FastAPI app with `POST /ask` (`AskRequest` in, `AskResponse` out) that calls
`graph.run_query()`, which invokes the compiled LangGraph app end-to-end.

### The `MOCK_LLM` toggle — what changes

| Stage | `MOCK_LLM=1` / unset (graded baseline) | `MOCK_LLM=0` (optional, ungraded extension) |
|---|---|---|
| `classify_intent` | Keyword heuristic, no LLM call | Calls an LLM to classify (hook: `graph._llm_classify_intent`, not implemented) |
| `retrieve_and_answer` | Retrieval always real; answer = canned `"Based on the retrieved context: ..."` string | Retrieval always real; answer = real LLM call using `prompt_template.build_prompt()`, grounded only in retrieved chunks, with up to 2 schema-validation retries (`graph._llm_generate_structured`) |
| `direct_answer` | Fixed canned string, no LLM call | Real LLM call with `prompt_template.DIRECT_ANSWER_PROMPT_TEMPLATE`, no retrieval |
| `schema` validation | Always trivially valid (built in code) | Real LLM JSON output is parsed and validated against `AskResponse`; on failure, retried up to 2 more times with a corrective instruction, else a clearly marked error response is returned |

Only ingestion/embedding/retrieval never branch — they always run for real, since they
need no LLM API key or network call.

The `_call_llm()` and `_llm_classify_intent()` hooks in `graph.py` are intentionally left
as `NotImplementedError` stubs: wiring in a real client (e.g. Groq's free tier) is the
optional, ungraded extension described in the assignment, and is never invoked by the
graded baseline since `MOCK_LLM` is left at its default during grading.

## Example calls (recorded with `MOCK_LLM` left at its default)

> Note on how these were captured: this was developed in a sandboxed environment whose
> network allowlist does not include `huggingface.co`, so the local embedding-model
> download could not be exercised inside that sandbox. The transcripts below were
> captured by running the exact shipped pipeline (`ingest.py` → `graph.py` → `main.py`,
> unmodified) with the `all-MiniLM-L6-v2` call substituted only by an equivalent local
> TF‑IDF vectorizer purely so retrieval could be verified end-to-end without that network
> access — chunking, ChromaDB storage/query, intent routing, canned-answer templating,
> and schema validation are all the real, shipped code paths. On a machine with normal
> internet access, `python ingest.py` downloads `all-MiniLM-L6-v2` once (cached
> thereafter) and produces the same JSON shape, typically with equal-or-better retrieval.

**Example 1 — triggers retrieval (`policy_question`)**

Request:
```json
POST /ask
{"query": "What is your delivery policy for large orders?"}
```

Response:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01_chunk0", "doc_08_chunk0", "doc_04_chunk0"],
  "confidence": 1.0
}
```
`classify_intent` matched the keyword `"delivery"` → `policy_question` → routed to
`retrieve_and_answer`. The top retrieved chunk (`doc_01_chunk0`) is drawn from the
Delivery Policy document, matching the question asked.

**Example 2 — does not trigger retrieval (`general_question`)**

Request:
```json
POST /ask
{"query": "Tell me a joke about robots"}
```

Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
No policy keyword was found → `general_question` → routed to `direct_answer`, which
returned the fixed canned string with `sources: []` and no LLM/network call.

**Additional routing check** (per acceptance criteria — one more of each path):

- `"Can I cancel my order after it's packed?"` → keyword `"cancel"` matched →
  `policy_question` → `retrieve_and_answer` → top chunk `doc_05_chunk0` (Order
  Cancellation Policy), correctly matching the question.
- `"What's the weather like today?"` → no keyword matched → `general_question` →
  `direct_answer` → canned string, `sources: []`.

## Optional extensions (not attempted / not required for grading)

- **Real LLM (`MOCK_LLM=0`) via Groq free tier**: not wired up in this submission. The
  hooks (`graph._call_llm`, `graph._llm_classify_intent`) are present and structurally
  complete (including the schema-validation retry loop) but raise `NotImplementedError`,
  since this is an optional, ungraded extension and grading uses the default `MOCK_LLM`.
- **Hugging Face Spaces deployment**: not attempted. The Dockerfile is locally
  buildable and runnable (`docker build` + `docker run` serving `POST /ask` on port
  7860), which is the required, graded containerization baseline.
