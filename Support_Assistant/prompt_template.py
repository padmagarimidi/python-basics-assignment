"""
Structured prompt template for the optional MOCK_LLM=0 real-LLM path.

Used by retrieve_and_answer() in graph.py to prompt a real LLM to answer a
policy_question grounded only in the retrieved ChromaDB chunks. This template
is not used at all in the graded, default mock-mode baseline (mock mode
answers with a canned string built directly from the retrieved chunk, with
no LLM call) — it exists in code, as text, to satisfy the requirement that
the prompt design be demonstrated even though it is only exercised on the
optional extension path.

The template follows the role - context - task - format - length skeleton,
includes one explicit negative constraint, and one embedded few-shot example.
"""

FEW_SHOT_EXAMPLE = """Example:
Context:
[doc_05_chunk0] Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app.

Question: Can I cancel my order after it's been packed?

Answer: {"answer": "No. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing. Orders can only be cancelled free of cost before the status changes to 'Packed', typically within the first 2 minutes of placing the order.", "sources": ["doc_05_chunk0"], "confidence": 0.95}
"""


def build_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Build the full role/context/task/format/length prompt for the real-LLM
    answer-generation call. `retrieved_chunks` is a list of dicts with
    keys "chunk_id" and "text", as returned by the retrieval step.
    """
    context_block = "\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in retrieved_chunks
    )

    prompt = f"""# ROLE
You are Zepto's customer support assistant, an expert on Zepto's own delivery,
returns, membership, and support policies.

# CONTEXT
Below are the policy document excerpts retrieved as most relevant to the
customer's question. This is the ONLY information you may use to answer.

Context:
{context_block}

# TASK
Answer the customer's question below using ONLY the information contained in
the context above. If the context does not contain enough information to
answer the question, say so explicitly rather than guessing.

Negative constraint: Do not answer using information not present in the
provided context, and do not rely on any outside or prior knowledge about
Zepto or grocery delivery services in general.

{FEW_SHOT_EXAMPLE}

# FORMAT
Respond with a single valid JSON object and nothing else (no markdown code
fences, no preamble, no commentary), matching exactly this shape:
{{"answer": "<string>", "sources": ["<chunk_id>", ...], "confidence": <float between 0 and 1>}}
"sources" must list only the chunk_id values (shown in square brackets above)
that you actually used to answer.

# LENGTH
Keep "answer" to 1-3 concise sentences.

Question: {question}

Answer:"""
    return prompt


DIRECT_ANSWER_PROMPT_TEMPLATE = """# ROLE
You are Zepto's customer support assistant.

# CONTEXT
The customer has asked a general question that is not about Zepto's delivery,
returns, membership, tracking, cancellation, damaged items, gift card, or
support-hours policies, so no policy document context has been retrieved.

# TASK
Answer briefly and helpfully if the question is answerable in general terms.
Negative constraint: Do not invent or assume any specific Zepto policy,
pricing, or figure that is not something you already reliably know to be
general public information.

Example:
Question: What is the capital of France?
Answer: {{"answer": "The capital of France is Paris.", "sources": [], "confidence": 0.9}}

# FORMAT
Respond with a single valid JSON object and nothing else, matching exactly:
{{"answer": "<string>", "sources": [], "confidence": <float between 0 and 1>}}

# LENGTH
Keep "answer" to 1-2 concise sentences.

Question: {question}

Answer:"""
