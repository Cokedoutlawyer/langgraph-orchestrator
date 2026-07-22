"""Layer 4.5 — Vectorize: scraped markdown -> chunk -> embed -> store -> retrieve.

Pluggable embedding/vector-store provider interface. Three backends, swappable
via VECTORIZE_BACKEND env var. Default = weaviate_cloud (matches the private
posture: no external inference key, billed through WCD, multilingual 8192-tok).

  scraped markdown
        |
   chunk(text)         -- fixed-size with overlap, sentence-aware
        |
   embed(text) -> vec  -- provider-specific
        |
   store(vec, meta)    -- provider-specific vector DB
        |
   retrieve(query)     -- semantic NN search (downstream agents hit this)

Backends:
  weaviate_cloud  (default) — text2vec-weaviate (Snowflake Arctic L v2.0),
                              ScrapedContent collection, hfresh index.
                              Private to WCD. No external key. LIVE + verified.
  weaviate_openai          — text2vec-openai (text-embedding-3-large @ 1024).
                              Needs a real OpenAI key (NOT OpenRouter).
                              Higher accuracy ceiling, OpenAI sees the text.
  self_hosted              — sentence-transformers BGE-M3 (MIT) on local GPU/CPU.
                              Zero data egress. Matches Venice no-logs posture.
                              Requires: uv add sentence-transformers + a model
                              download (~2GB). CPU works, GPU faster.
"""
from __future__ import annotations

import os
import re
from typing import Any

import structlog

log = structlog.get_logger("vectorize")

DEFAULT_BACKEND = os.environ.get("VECTORIZE_BACKEND", "weaviate_cloud")
CHUNK_SIZE = int(os.environ.get("VECTORIZE_CHUNK_SIZE", "1200"))   # chars
CHUNK_OVERLAP = int(os.environ.get("VECTORIZE_CHUNK_OVERLAP", "200"))


# --------------------------------------------------------------------------- #
# Chunking — sentence-aware fixed-size windows.
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking on sentence boundaries."""
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Backtrack to the last sentence boundary within the window
            window = text[start:end]
            m = list(re.finditer(r"[.!?]\s+", window))
            if m:
                end = start + m[-1].end()
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# --------------------------------------------------------------------------- #
# Provider interface.
# --------------------------------------------------------------------------- #
class VectorStore:
    """Abstract: embed + store + retrieve."""

    name: str = "base"

    def ingest(self, query: str, url: str, content: str, source: str) -> int:
        """Chunk, embed, store. Returns number of chunks stored."""
        raise NotImplementedError

    def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic nearest-neighbor search. Returns [{content, url, score}]."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Backend 1 (default): Weaviate Cloud — text2vec-weaviate (Snowflake Arctic L).
# --------------------------------------------------------------------------- #
class WeaviateCloudStore(VectorStore):
    name = "weaviate_cloud"

    def __init__(self):
        from .query_agent import ingest_scraped, _connect, _env
        self._ingest = ingest_scraped
        self._connect = _connect
        self._env = _env
        self._collection = "ScrapedContent"

    def ingest(self, query: str, url: str, content: str, source: str) -> int:
        # The ScrapedContent collection stores whole documents; the Weaviate
        # vectorizer chunks-implicitly per object. We store one object per
        # explicit chunk so retrieval is chunk-granular (matches the RAG design).
        chunks = chunk_text(content)
        if not chunks:
            return 0
        items = [
            {"url": url, "query": query, "content": c, "source": source}
            for c in chunks
        ]
        try:
            return self._ingest(items)
        except Exception as e:
            log.error("weaviate_cloud.ingest_failed", error=str(e)[:200])
            return 0

    def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        """Direct near_text search (semantic NN) — no LLM round-trip."""
        from weaviate.classes.query import MetadataQuery
        client = self._connect()
        try:
            col = client.collections.get(self._collection)
            res = col.query.near_text(
                query=query, limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )
            out = []
            for o in res.objects:
                d = getattr(o.metadata, "distance", None)
                out.append({
                    "content": str(o.properties.get("content", ""))[:300],
                    "url": o.properties.get("url", ""),
                    "score": (1 - d) if d is not None else None,
                })
            return out
        except Exception as e:
            log.error("weaviate_cloud.retrieve_failed", error=str(e)[:200])
            return []
        finally:
            client.close()


# --------------------------------------------------------------------------- #
# Backend 2: Weaviate Cloud — text2vec-openai (text-embedding-3-large @ 1024).
# Higher accuracy ceiling; OpenAI sees the text. Needs a REAL OpenAI key.
# --------------------------------------------------------------------------- #
class WeaviateOpenAIStore(VectorStore):
    name = "weaviate_openai"

    def __init__(self):
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        self._weaviate = weaviate
        self._Configure = Configure
        self._Property = Property
        self._DataType = DataType
        self._key = _env("WEAVIATE_API_KEY")
        self._url = _env("WEAVIATE_URL")
        self._oai = _env("OPENAI_API_KEY")
        if not self._oai or self._oai.startswith("sk-or-"):
            raise RuntimeError(
                "weaviate_openai backend needs a REAL OpenAI key (OPENAI_API_KEY), "
                "not an OpenRouter key (sk-or-*). Set VECTORIZE_BACKEND=weaviate_cloud "
                "or weaviate_self_hosted instead."
            )
        self._collection = "ScrapedContentOpenAI"

    def _client(self):
        return self._weaviate.connect_to_weaviate_cloud(
            cluster_url=self._url,
            auth_credentials=self._key,
            headers={"X-OpenAI-Api-Key": self._oai},
        )

    def _ensure(self, client):
        if not client.collections.exists(self._collection):
            client.collections.create(
                name=self._collection,
                vectorizer_config=self._Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-large", dimensions=1024,
                ),
                vector_index_config=self._Configure.VectorIndex.hfresh(),
                properties=[
                    self._Property(name="url", data_type=self._DataType.TEXT),
                    self._Property(name="query", data_type=self._DataType.TEXT),
                    self._Property(name="content", data_type=self._DataType.TEXT),
                    self._Property(name="source", data_type=self._DataType.TEXT),
                ],
            )

    def ingest(self, query: str, url: str, content: str, source: str) -> int:
        chunks = chunk_text(content)
        if not chunks:
            return 0
        client = self._client()
        try:
            self._ensure(client)
            col = client.collections.get(self._collection)
            with col.batch.dynamic() as batch:
                for c in chunks:
                    batch.add_object(properties={
                        "url": url, "query": query, "content": c, "source": source,
                    })
            return len(chunks)
        except Exception as e:
            log.error("weaviate_openai.ingest_failed", error=str(e)[:200])
            return 0
        finally:
            client.close()

    def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        from weaviate.classes.query import MetadataQuery
        client = self._client()
        try:
            col = client.collections.get(self._collection)
            res = col.query.near_text(
                query=query, limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )
            out = []
            for o in res.objects:
                d = getattr(o.metadata, "distance", None)
                out.append({
                    "content": str(o.properties.get("content", ""))[:300],
                    "url": o.properties.get("url", ""),
                    "score": (1 - d) if d is not None else None,  # cosine distance -> sim
                })
            return out
        except Exception as e:
            log.error("weaviate_openai.retrieve_failed", error=str(e)[:200])
            return []
        finally:
            client.close()


# --------------------------------------------------------------------------- #
# Backend 3: self-hosted BGE-M3 (MIT). Zero egress. Matches Venice posture.
# --------------------------------------------------------------------------- #
class SelfHostedBGEStore(VectorStore):
    name = "self_hosted"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "self_hosted backend needs sentence-transformers: "
                "`uv add sentence-transformers`. Model BAAI/bge-m3 (~2GB download)."
            ) from e
        self._model_name = os.environ.get("VECTORIZE_SELF_HOSTED_MODEL", "BAAI/bge-m3")
        self._model = SentenceTransformer(self._model_name)
        self._store: list[dict] = []  # in-memory list; swap for qdrant/milvus in prod

    def ingest(self, query: str, url: str, content: str, source: str) -> int:
        chunks = chunk_text(content)
        if not chunks:
            return 0
        vecs = self._model.encode(chunks, normalize_embeddings=True)
        for c, v in zip(chunks, vecs):
            self._store.append({
                "content": c, "url": url, "query": query,
                "source": source, "vec": v.tolist(),
            })
        return len(chunks)

    def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        if not self._store:
            return []
        import numpy as np
        qv = self._model.encode([query], normalize_embeddings=True)[0]
        mats = np.array([r["vec"] for r in self._store])
        scores = mats @ qv  # cosine (normalized)
        idx = np.argsort(scores)[::-1][:limit]
        return [
            {
                "content": self._store[i]["content"][:300],
                "url": self._store[i]["url"],
                "score": float(scores[i]),
            }
            for i in idx
        ]


# --------------------------------------------------------------------------- #
# Registry + factory.
# --------------------------------------------------------------------------- #
_BACKENDS = {
    "weaviate_cloud": WeaviateCloudStore,
    "weaviate_openai": WeaviateOpenAIStore,
    "self_hosted": SelfHostedBGEStore,
}


def get_store(backend: str | None = None) -> VectorStore:
    """Return the active vector store (cached on first call)."""
    name = backend or DEFAULT_BACKEND
    if name not in _BACKENDS:
        raise RuntimeError(
            f"Unknown VECTORIZE_BACKEND={name}. "
            f"Choose from: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[name]()


def _env(name: str) -> str:
    for src in (os.environ.get(name),):
        if src:
            return src
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"')
    return ""
