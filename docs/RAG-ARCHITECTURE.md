---
name: weaviate-engram-rag
description: "Use when adding Weaviate RAG to Hermes or a LangGraph orchestrator. Engram managed-memory layer (user-scoped, async ingest, hybrid retrieval) plus optional Weaviate Query Agent for direct collection querying. Covers install, API surface, Engram SearchResults normalization, RAG-injection node wiring, and Darwinian evolution of the injection prompt."
version: 1.0.0
author: MATTS
license: MIT
metadata:
  hermes:
    tags: [rag, weaviate, engram, memory, langgraph, vector-search, retrieval]
    related_skills: [langgraph-orchestrator, darwinian-evolver, mcp-provider-wiring]
---

# Weaviate Engram RAG Suite

Two RAG layers, one default:

1. **Engram (default, managed)** — Weaviate-backed managed-memory service. API-key auth (`ENGRAM_API_KEY`). User-scoped long-term memory: async ingestion via `memories.add()`, hybrid/vector/BM25 retrieval via `memories.search()`. Auto-extracts structured memories from conversations (topics: UserKnowledge, UserProfile). No cluster to manage. This is the layer MATTS uses.
2. **Weaviate Query Agent (optional, requires cloud)** — Direct querying of Weaviate Cloud collections via `ask()`/`search()`/`run()`. Needs `WEAVIATE_URL` + `WEAVIATE_API_KEY` (a Weaviate Cloud cluster) and a hosted agents endpoint. Engram covers the memory use case without this; only add it when you need to query your own raw collections.

   Verified integration (cluster 1.38.6, weaviate-client 4.22): `weaviate.connect_to_weaviate_cloud(cluster_url='https://<REST>', auth_credentials=<raw API key string>, headers={...})`. Raw-string auth works — no `ApiKey()` wrapper needed in v4.22 (`from weaviate.auth import ApiKey` doesn't exist; pass the key string directly). Tool wrapper: `src/langgraph_orchestrator/tools/query_agent.py` — `ingest_scraped()`, `query_agent_ask()`, `query_agent_search()`.

## Query Agent Collection Setup (verified)

The ScrapedContent collection stores scraped web content for the Query Agent to query. Key config that actually worked on this cluster:

- **Vectorizer: `text2vec-weaviate`** — Weaviate's own hosted embedding model (Weaviate Embeddings). Needs NO external inference API key (billed through the WCD cluster). This is the right pick when your `OPENAI_API_KEY` is actually an OpenRouter key (Weaviate's text2vec-openai calls OpenAI directly and 401-rejects `sk-or-v1...` keys).
  - **Available models** (pass via `Configure.Vectorizer.text2vec_weaviate(model=...)`):
    - `Snowflake/snowflake-arctic-embed-l-v2.0` — **default**, multilingual, up to 8192 tokens, best for big-scale/complex retrieval. The right pick for multilingual geo-scrapes.
    - `Snowflake/snowflake-arctic-embed-m-v1.5` — English only, 512 tokens, fast/lightweight. Use for English-only, latency-sensitive workloads.
    - `colmodernvbert` (ColModernVBERT) — English, supports images + document pages (query text up to 8092 tok), best for visual documents (PDFs, slides, invoices) without OCR preprocessing.
  - Pin the model explicitly (`model="Snowflake/snowflake-arctic-embed-l-v2.0"`) rather than relying on the default — the default is currently the same L v2.0, but pinning makes the collection reproducible and immune to default changes.
- **Vector index: `hfresh`** — This cluster only allows `hfresh` (memory-efficient, SPFresh-based). `hnsw` returns 422 `CONFIG_NOT_ALLOWED: hnsw is not allowed for vector_index_type. Allowed values: hfresh.` Check the cluster's allowed index types before creating a collection.
- **Property config:** `from weaviate.classes.config import Configure, Property, DataType` — `Property` and `DataType` are direct imports, NOT `Configure.Property`/`Configure.DataType`. `Configure` only holds `Vectorizer`, `VectorIndex`, `Generative`, etc.
- **Ingestion:** `col.batch.dynamic()` with `batch.add_object(properties={...})`. Objects vectorize async in WCD — wait ~6-8s before querying fresh data.
- **Query Agent `ask()`:** `QueryAgent(client=client, collections=[name], timeout=120).ask(question)` returns a response with `.final_answer`. The agent runs agentic retrieval + synthesis over the collection.

## When to Use

- Adding long-term memory / RAG to a Hermes agent or LangGraph orchestrator
- User says "add Engram" / "add Weaviate RAG" / "give the agent persistent memory"
- Wiring a RAG retrieval node that augments LLM answers with stored user context
- Evolving the prompt that injects retrieved memory into the final answer

Don't use for: short-term scratch state (use session state); differentiable optimization (use DSPy); single-page Q&A (just pass content to the LLM).

## Install + Auth

```bash
uv add weaviate-engram   # Engram (managed); uv add weaviate-agents for the Query Agent
```

`ENGRAM_API_KEY` goes in `~/.hermes/.env` (WSL + Windows + VPS — dedup, keep latest). Verify with a real `memories.add()` + `memories.search()` round-trip, never a dry run.

## Engram API Surface (verified, weaviate-engram 1.0.0)

```python
from engram import EngramClient, HybridRetrieval, VectorRetrieval, BM25Retrieval
client = EngramClient(api_key=os.environ["ENGRAM_API_KEY"])

# Ingest (async — returns run_id immediately; processing happens in background)
run = client.memories.add(
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    user_id="matts@hermes",
)
print(run.run_id)
# Block until searchable: client.runs.wait(run.run_id, timeout=60.0)

# Search — SearchResults is a Sequence of Memory objects (NOT a dict)
results = client.memories.search(query="...", user_id="...", retrieval_config=HybridRetrieval())
for m in list(results):           # iterate the Sequence, don't call .get()
    m.content, m.id, m.score, m.topic, m.tags   # attributes, not dict keys

# CRUD
client.memories.get(memory_id, user_id=)
client.memories.delete(memory_id, user_id=)
```

## Embedding Model Picks (self-hosted path only)

Engram is managed — it embeds server-side, so you do NOT pick the embedding model there. These picks apply when self-hosting: the Query Agent against your own Weaviate collections, or a custom Weaviate/Milvus/Qdrant store.

| Use case | Pick | Why |
|---|---|---|
| English-heavy web scrapes (default) | `text-embedding-3-large` truncated to 1024 dims | Matryoshka trick: ~3× storage savings, ~95% retrieval quality. Best accuracy-per-dollar for English. |
| Multilingual scrapes | BGE-M3 self-hosted (MIT) or Cohere Multilingual-v3 | BGE-M3 is MIT-licensed, self-hostable, strong multilingual. Cohere if you want managed. |
| Private-by-default (matches Venice no-logs posture) | Self-host BGE-M3 or mxbai-embed-large on own GPU/CPU | Zero data leaves your infra. Aligns with the no-logs privacy stance. |

**Dims-to-store rule:** pgvector doesn't care about dims. Milvus/Qdrant/Weaviate prefer 768 or 1024 for best index performance. Truncate to 1024 if going self-hosted. (text-embedding-3-large is 3072 native → truncate to 1024 via the Matryoshka `dimensions` param.)

## Critical Pitfalls (all hit during integration)

1. **`SearchResults` is a Sequence, not a dict.** It has `.count/.total/.index` and is iterable. Calling `.get("results")` raises `AttributeError`. Normalize via `for m in list(results): m.content / m.id / m.score`.
2. **`memories.add()` is async.** Returns a `run_id` immediately; memory isn't searchable until background processing finishes. Call `client.runs.wait(run_id, timeout=60.0)` if you need immediate searchability. Wrap in try/except — wait failure is non-fatal.
3. **Engram auto-extracts knowledge facts.** Ingesting "I prefer specialty coffee, Bonanza is my favorite" yields a stored Memory like `User prefers specialty coffee and is not interested in chain coffee shops.` with `topic='UserKnowledge'`. The retrieved content is the distilled fact, not your raw text.
4. **Query Agent ≠ Engram.** The raw Weaviate Query Agent (`weaviate-agents`) needs a Weaviate **Cloud** cluster (`connect_to_weaviate_cloud` + `WEAVIATE_URL/API_KEY`) and routes to a hosted agents endpoint. Embedded/local Weaviate works for the client but the agent's `agents_host` still wants cloud auth. Use Engram unless you genuinely need to query your own collections.
5. **Vectorizer inference keys.** Weaviate Cloud collections with a vectorizer need the inference provider key passed as a header: `'X-INFERENCE-PROVIDER-API-KEY': OPENAI_API_KEY`. Engram handles this server-side — no extra header.
6. **Retrieval type choice.** `HybridRetrieval` (default, vector+BM25) for general queries; `VectorRetrieval` for semantic/paraphrase; `BM25Retrieval` for exact keyword. Pick per query type.
7. **Multi-tenancy → use the `flat` vector index, not HNSW.** Engram's user-scoped memory is multi-tenant (one tenant per user_id). Weaviate's own guidance: `flat` index is recommended when objects-per-index is low, i.e. multi-tenancy. Only switch a collection to `hnsw` if a single tenant shard grows large. Engram manages this server-side; if you build your own collection with per-user tenants, set `vectorIndexType: flat`.
8. **Darwinian evolver crashes if the seed aces everything.** If your evaluator's seed organism scores 1.000, it produces zero `trainable_failure_cases`, and the loop's `sample_parents` raises `RuntimeError: No eligible organisms for parent selection` (it drops untrainable organisms). Make the evaluator STRICTER than the seed can pass — include "must NOT mention X" negative checks and "must explicitly anchor to preference" phrasing checks so the seed lands <0.9 and evolution has signal. This bit the RAG-instruction evolver on its first run.
9. **Cluster may restrict the vector index type.** This WCD cluster only allows `hfresh` — `Configure.VectorIndex.hnsw()` returns 422 `CONFIG_NOT_ALLOWED`. Always check `client.get_meta()['modules']` for enabled modules, and if a collection create fails with `CONFIG_NOT_ALLOWED`, switch to the allowed index (here: `hfresh`).
10. **`OPENAI_API_KEY` that's actually an OpenRouter key breaks `text2vec-openai`.** Weaviate's `text2vec-openai` vectorizer calls OpenAI directly (it does not honor `OPENAI_BASE_URL`). If your `OPENAI_API_KEY` is `sk-or-v1...` (OpenRouter), vectorization 401s: `Incorrect API key provided: sk-or-v1***`. Fix: use `text2vec-weaviate` (Weaviate's own hosted embeddings, no external key) or `text2vec-huggingface` (with `HUGGINGFACE_API_KEY`). `text2vec-weaviate` is the cleanest — billed through the cluster, zero external dependency.
11. **`Property`/`DataType` are top-level imports, not on `Configure`.** In weaviate-client 4.22: `from weaviate.classes.config import Configure, Property, DataType`. `Configure` only holds `Vectorizer`, `VectorIndex`, `Generative`, `multi_tenancy`, etc. `Configure.Property` / `Configure.DataType` raise `AttributeError`.

## RAG Node Wiring (LangGraph) — CONDENSED

The vectorize + memory-retrieval layers are merged into a single `rag` node (was two nodes: vectorize → rag). One pass, two responsibilities, both non-blocking.

```
START → plan → get → run → [rag → summary | retry_bump → plan]
```

Nodes: plan, get, run, rag, summary, retry_bump (7 nodes, was 8).

- `rag_node` (condensed, `nodes/rag.py`): in ONE pass — (1) chunks + embeds + stores scraped markdown via the active `VectorStore` backend, (2) retrieves user-scoped long-term memory via Engram. Both wrapped in try/except — RAG augments, never blocks. Writes `rag_context`, `rag_memories`, `vectorized_chunks`, `vectorize_backend` to state.
- `summary_node`: if `rag_context` non-empty, injects a memory block + the evolved injection instruction. Raw scraped content is the factual ground truth; memory personalizes.

State fields: `rag_context: str`, `rag_memories: list`, `vectorized_chunks: int`, `vectorize_backend: str`.

## Hand-Selected Model Matrix (`nodes/models.py`)

Every graph node reads its model from `resolve("<task>")` — no node inherits a generic default. Edit `nodes/models.py` to swap; re-verify a new model serves with a 1-token completion before committing.

| Task | Model | Provider | Why | Servable? |
|---|---|---|---|---|
| plan (JSON routing) | `blackboxai/deepseek/deepseek-v4-pro` | Blackbox | Strong reasoning + JSON discipline. Best servable reasoner — Anthropic/GPT-5 routes were Vertex-down at selection time. | ✓ verified |
| summary (synthesis) | `blackboxai/google/gemini-3.5-flash` | Blackbox | 1M-context, fast, cheap, strong grounded synthesis over scraped content + memory. | ✓ verified |
| venice (private scrape) | `llama-3.3-70b` | Venice (direct) | Proven Venice scraper with `enable_web_scraping`. Private inference. | ✓ verified |
| embedding | `Snowflake/snowflake-arctic-embed-l-v2.0` | Weaviate (text2vec-weaviate) | Multilingual 8192-tok, no external key, private to WCD. Locked. | ✓ verified |
| fallback | `z-ai/glm-5.2` | Blackbox | Cheap, always-on. Used when a primary is rate-limited. | ✓ verified |

**Servability gotcha (2026-07-22):** Blackbox routes Anthropic (claude-sonnet/opus/fable) and GPT-5.x through Vertex AI, which was returning `Vertex_aiException InternalServerError` for all of them. DeepSeek, Kimi-k2.7-code, Gemini 3.5 Flash, Gemini 3.1 Flash Lite, and glm-5.2 all served fine. Always test a 1-token completion against `/v1/chat/completions` before pinning a model — the `/models` listing shows availability, not live servability.

## Layer 4.5 — Vectorize (Pluggable Embedding/Store)

`src/langgraph_orchestrator/tools/vectorize.py` — a provider interface (`VectorStore`) with three swappable backends, selected via `VECTORIZE_BACKEND` env var. Default = `weaviate_cloud`. (The standalone `vectorize_node` was merged into `rag_node`; the `VectorStore` tool itself remains.)

| Backend | `VECTORIZE_BACKEND` | Embedding model | Private? | Needs | Use when |
|---|---|---|---|---|---|
| Weaviate Cloud (default) | `weaviate_cloud` | Snowflake Arctic L v2.0 (text2vec-weaviate, multilingual 8192-tok) | Private to WCD (no external key) | `WEAVIATE_API_KEY` + `WEAVIATE_URL` | **Default.** Matches the no-egress posture. Zero external inference key. |
| Weaviate + OpenAI | `weaviate_openai` | text-embedding-3-large @ 1024 dims (Matryoshka) | OpenAI sees the text | A REAL OpenAI key (NOT `sk-or-*`) | You want the accuracy ceiling and accept OpenAI data handling. |
| Self-hosted | `self_hosted` | BGE-M3 (BAAI/bge-m3, MIT) | Zero egress | `uv add sentence-transformers` (~2GB model) | Strict no-logs posture (matches Venice). CPU works, GPU faster. |

**Chunking:** `chunk_text()` — sentence-aware fixed-size windows (default 1200 chars, 200 overlap). One Weaviate object per chunk so retrieval is chunk-granular.

**Why the default is weaviate_cloud, not the MTEB-leading OpenAI/Voyage models:** the rankings table (OpenAI text-embedding-3-large, Voyage voyage-3-large) are proprietary cloud APIs that send your text to OpenAI/Voyage. The architecture constraint is "embeddings should stay inside the private boundary." `text2vec-weaviate` (Snowflake Arctic L v2.0) is hosted by Weaviate itself, needs no external key, is multilingual (handles geo-scrapes), and has 8192-token context (full pages without chunk-games). It's the accuracy-vs-privacy tradeoff that matches the stack. Swap to `weaviate_openai` only if you need the ceiling and accept the egress; `self_hosted` for hard no-egress.

**Verified in production:** vectorize node fires after run, stored 10 chunks from a 3356-char Berlin scrape into ScrapedContent (now 215+ objects). Downstream semantic search works across paraphrased queries:
- "Which Berlin cafes roast their own beans?" → "rams cafe and roastery" (0.669)
- "Where can I find Bonanza coffee?" → Bonanza chunks (0.598)
- "Prenzlauer Berg coffee shops" → Prenzlauer Berg chunks (0.649)

**Retrieve uses direct `near_text`, NOT the QueryAgent `.search()`** — the QueryAgent search response shape is unreliable for raw object retrieval. `col.query.near_text(query, limit, return_metadata=MetadataQuery(distance=True))` returns objects directly with distance metadata (convert to similarity: `1 - distance`).

## Darwinian Evolution of the RAG Injection Prompt

The instruction that tells the summarizer how to weave memory into the answer is evolvable. Driver: `scripts/evolve_rag_instruction.py` — evolves the instruction against 3 test cases with strict checks (required substrings present + forbidden substrings absent + explicit preference anchoring). Scorer: weighted fraction of checks passed.

- Seed instruction: 0.783 score
- Evolved winner: 0.883 (improved, plateaued at iter 1)

The evolved instruction adds: explicit "since you..." anchoring, hard-exclusion of disliked items ("treat any disliked or irrelevant items as hard exclusions—never mention them even if they appear in scraped content"), and the positive-pref-emphasis / negative-pref-exclusion split. Anchoring is the easier signal; full negative exclusion (the model still sometimes mentions a disliked option present in scraped content) is the harder one — the plateau at 0.883 reflects this.

Re-run:
```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver
EVOLVER_MODEL=z-ai/glm-5.2 uv run --with openai python \
  /home/peter/projects/langgraph-orchestrator/scripts/evolve_rag_instruction.py \
  --num_iterations 5 --num_parents_per_iteration 2 \
  --mutator_concurrency 2 --evaluator_concurrency 2 \
  --output_dir /tmp/evolve_rag
# Winner: /tmp/evolve_rag/best_instruction.txt
```

## Engram Health Check (correct endpoint — do NOT guess)

`GET https://api.engram.weaviate.io/health` → 200 + `{"status":"healthy","service":"engram-memory-server"}`. Auth via `Authorization: Bearer $ENGRAM_API_KEY` (optional for /health but harmless).

**Do NOT probe `/v1/health`, `/healthz`, `/v1/ready`, or `/v1/` — those return 404 and look like the service is down when it isn't.** The base path is `/health` (no `/v1/` prefix). This bit the 2026-07-22 audit: a wrong-path 404 got reported as a failure when Engram was healthy the whole time.

For the real round-trip verification, use the Python client (`memories.search(...)`), NOT a hand-crafted curl POST to `/v1/memories/search` — the request schema (especially `retrieval_config`) is strict and a malformed body returns 422, which looks like an API failure but is just a bad payload. The client builds the correct schema. A successful `memories.search()` returning results (or an empty-but-200 result set) is the proof Engram is alive and authed.

## Verification

- [ ] `ENGRAM_API_KEY` in all .env files (deduped, latest only)
- [ ] Engram health: `GET https://api.engram.weaviate.io/health` → 200 healthy (NOT /v1/health)
- [ ] `memories.add()` returns a `run_id` (real API call, not dry run)
- [ ] `memories.search()` returns Memory objects with `.content/.id/.score` (Sequence, not dict)
- [ ] `runs.wait(run_id)` succeeds OR non-fatal on timeout
- [ ] Vectorize node wired: run → vectorize → rag → summary on success
- [ ] Vectorize node fires, stores chunks (`vectorized_chunks` > 0 in state)
- [ ] Downstream `store.retrieve(paraphrased_query)` returns semantic matches without re-scraping
- [ ] RAG node wired after vectorize; rag wrapped in try/except
- [ ] Summary injects `rag_context` block only when non-empty
- [ ] End-to-end: ingest "I prefer X" → query related topic → answer mentions X
- [ ] Injection prompt evolved (seed 0.783 → evolved 0.883; anchoring improved, exclusion plateaued)
- [ ] Query Agent: `WEAVIATE_URL` + `WEAVIATE_API_KEY` in all .env files (raw-string auth)
- [ ] `connect_to_weaviate_cloud` returns `is_ready()=True` (real cluster connection)
- [ ] ScrapedContent collection created with `text2vec-weaviate` + `hfresh` index
- [ ] `ingest_scraped()` writes objects; `query_agent_ask()` returns a grounded `.final_answer`
- [ ] VPS sync attempted (queued if SSH timeout)
