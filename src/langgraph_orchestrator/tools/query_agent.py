"""Weaviate Query Agent — direct collection querying layer.

Complements the Engram managed-memory layer. This stores scraped content in a
Weaviate Cloud collection and lets the Query Agent run agentic ask()/search()
over it. Engram = user-scoped long-term memory (managed); Query Agent = query
your own scraped-content collections (self-hosted-in-cloud).

Verified: cluster 1.38.6, connect_to_weaviate_cloud with raw-string ApiKey auth.
"""
from __future__ import annotations

import os
from typing import Any

import weaviate
from weaviate.agents.query import QueryAgent
from tenacity import retry, stop_after_attempt, wait_exponential


class QueryAgentError(Exception):
    pass


def _env(name: str) -> str:
    for src in (os.environ.get(name),):
        if src:
            return src
    # fall back to .env
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip()
    return ""


def _connect():
    """Connect to the configured Weaviate Cloud cluster."""
    key = _env("WEAVIATE_API_KEY")
    url = _env("WEAVIATE_URL") or "https://hqivu8vwtqyfwxapxizhpg.c0.us-east-1.aws.weaviate.cloud"
    if not key:
        raise QueryAgentError("WEAVIATE_API_KEY not set")
    oai = _env("OPENAI_API_KEY")
    headers = {"X-OpenAI-Api-Key": oai} if oai else {}
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url, auth_credentials=key, headers=headers,
    )


def ensure_scraped_collection(client: weaviate.WeaviateClient, name: str = "ScrapedContent") -> None:
    """Create the ScrapedContent collection if it doesn't exist.

    Uses text2vec-weaviate with Snowflake Arctic Embed L v2.0 — Weaviate's hosted
    embedding model (Weaviate Embeddings). Multilingual, 8192-token context.
    No external inference API key required (billed through the WCD cluster).
    The cluster only allows the hfresh vector index (memory-efficient, SPFresh-based).

    Available Weaviate Embeddings models:
      - Snowflake/snowflake-arctic-embed-l-v2.0  (default; multilingual, 8192 tok, big-scale)
      - Snowflake/snowflake-arctic-embed-m-v1.5  (English, 512 tok, fast/lightweight)
      - colmodernvbert                            (English, image+doc pages, visual docs w/o OCR)
    """
    if not client.collections.exists(name):
        from weaviate.classes.config import Configure, Property, DataType
        client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(
                model="Snowflake/snowflake-arctic-embed-l-v2.0",  # multilingual, pinned explicit
            ),
            vector_index_config=Configure.VectorIndex.hfresh(),
            properties=[
                Property(name="url", data_type=DataType.TEXT),
                Property(name="query", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ],
        )


def ingest_scraped(
    items: list[dict],
    collection: str = "ScrapedContent",
) -> int:
    """Ingest scraped content dicts {url, query, content, source} into the collection."""
    client = _connect()
    try:
        ensure_scraped_collection(client, collection)
        col = client.collections.get(collection)
        with col.batch.dynamic() as batch:
            for it in items:
                batch.add_object(properties={
                    "url": it.get("url", ""),
                    "query": it.get("query", ""),
                    "content": (it.get("content") or "")[:5000],
                    "source": it.get("source", "brightdata"),
                })
        return len(items)
    finally:
        client.close()


@retry(reraise=True, stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def query_agent_ask(
    question: str,
    collection: str = "ScrapedContent",
    timeout: int = 90,
) -> str:
    """Ask the Query Agent a question over the collection. Returns the final answer."""
    client = _connect()
    try:
        qa = QueryAgent(client=client, collections=[collection], timeout=timeout)
        resp = qa.ask(question)
        return resp.final_answer or str(resp)
    except Exception as e:
        raise QueryAgentError(f"query_agent_ask failed: {e}")
    finally:
        client.close()


@retry(reraise=True, stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def query_agent_search(
    query: str,
    collection: str = "ScrapedContent",
    limit: int = 10,
) -> list[dict]:
    """Search mode — returns raw matching objects (no LLM synthesis)."""
    client = _connect()
    try:
        qa = QueryAgent(client=client, collections=[collection])
        resp = qa.search(query=query, limit=limit)
        out = []
        for obj in getattr(resp, "objects", []) or []:
            out.append({
                "content": str(getattr(obj, "properties", {}).get("content", ""))[:300],
                "url": getattr(obj, "properties", {}).get("url", ""),
                "score": getattr(obj, "metadata", None).score if getattr(obj, "metadata", None) else None,
            })
        return out
    except Exception as e:
        raise QueryAgentError(f"query_agent_search failed: {e}")
    finally:
        client.close()
