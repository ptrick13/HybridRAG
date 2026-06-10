"""Qdrant hybrid retrieval tool.

Implements ``search_documents``, which combines dense (OpenAI
text-embedding-3-large) and sparse (BM25 via fastembed) retrieval, fused
with Reciprocal Rank Fusion (RRF).

RRF is used instead of manual score weighting because it is robust to
different score scales across retrieval methods and requires no calibration.
"""

import asyncio
import json
import logging
import threading
import uuid
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from agents.client import get_async_client
from agents.usage import record_embedding_tokens
from config.settings import settings

logger = logging.getLogger(__name__)

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client instance."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


@lru_cache(maxsize=1)
def _get_sparse_model() -> SparseTextEmbedding:
    """Return a cached fastembed BM25 sparse embedding model."""
    return SparseTextEmbedding(model_name="Qdrant/bm25")


# fastembed's SparseTextEmbedding.embed() is not documented as thread-safe.
# Multiple agents retrieve concurrently via run_in_executor (default thread pool),
# so all calls to _embed_sparse share this lock to prevent state corruption.
_sparse_model_lock = threading.Lock()


async def _embed_dense(text: str) -> list[float]:
    """Generate a dense embedding via the OpenAI embedding API.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the dense embedding vector.
    """
    client = get_async_client()
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    if response.usage:
        record_embedding_tokens(settings.embedding_model, response.usage.total_tokens)
    return response.data[0].embedding


def _embed_sparse(text: str) -> SparseVector:
    """Generate a BM25 sparse embedding using fastembed.

    Args:
        text: The text to embed.

    Returns:
        A ``SparseVector`` with indices and values for Qdrant.
    """
    model = _get_sparse_model()
    with _sparse_model_lock:
        embedding = list(model.embed([text]))[0]
    return SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


_COLLECTION_KEY_MAP = {
    "tickets": "qdrant_collection_tickets",
    "arch_docs": "qdrant_collection_arch_docs",
    "postmortems": "qdrant_collection_postmortems",
}


async def search_documents(
    query: str, collection: str = "tickets"
) -> list[dict[str, Any]]:
    """Retrieve the top-k most relevant Software Development Analytics documents.

    Hybrid retrieval pipeline:
    1. Dense: embed query with text-embedding-3-large → cosine vector search
    2. Sparse: embed query with BM25 via fastembed → sparse vector search
    3. Fusion: combine both result sets using Qdrant's built-in RRF

    Returns raw result dicts without interpretation or summarisation.

    Args:
        query: Natural language search query optimised for hybrid retrieval.
        collection: Logical collection name — ``"tickets"``, ``"arch_docs"``, or
                    ``"postmortems"``. Defaults to ``"tickets"``.

    Returns:
        A list of result dicts, each containing ``id``, ``score``, and
        ``payload`` (document metadata and content).
    """
    attr = _COLLECTION_KEY_MAP.get(collection, "qdrant_collection_tickets")
    collection_name: str = getattr(settings, attr)

    client = _get_qdrant_client()

    # Run dense embedding asynchronously; sparse embedding is CPU-bound
    # so we run it in a thread executor to avoid blocking the event loop.
    dense_vector, sparse_vector = await asyncio.gather(
        _embed_dense(query),
        asyncio.get_running_loop().run_in_executor(None, _embed_sparse, query),
    )

    # query_points is synchronous — run it in an executor so it does not block
    # the event loop while multiple agents retrieve in parallel.
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        lambda: client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using=_DENSE_VECTOR_NAME, limit=20),
                Prefetch(query=sparse_vector, using=_SPARSE_VECTOR_NAME, limit=20),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=settings.top_k_results,
            with_payload=True,
        ),
    )

    formatted = [
        {
            "id": str(point.id),
            "score": point.score,
            "payload": point.payload or {},
        }
        for point in results.points
    ]

    logger.debug(
        "Qdrant '%s' returned %d results for query: %s",
        collection_name, len(formatted), query[:80],
    )
    return formatted


def ensure_collection_exists(collection_name: str | None = None) -> None:
    """Create a single Qdrant collection if it does not exist.

    The collection is configured for hybrid retrieval with:
    - A dense vector field (cosine distance, 3072 dims for text-embedding-3-large)
    - A sparse vector field (BM25 indices)

    This function is idempotent — safe to call on every application start.

    Note:
        Without an argument, only the tickets collection is created (the default
        alias). To set up all three domain collections, call this function once
        per collection name: ``settings.qdrant_collection_tickets``,
        ``settings.qdrant_collection_arch_docs``, and
        ``settings.qdrant_collection_postmortems``.  The ingestion pipeline
        (``ingest_qdrant.py``) handles this automatically.

    Args:
        collection_name: Collection to create. Defaults to ``settings.qdrant_collection``
            (an alias for ``qdrant_collection_tickets``).
    """
    name = collection_name or settings.qdrant_collection
    client = _get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info("Created Qdrant collection '%s'.", name)
    else:
        logger.debug("Qdrant collection '%s' already exists.", name)


async def index_document(
    doc_id: str,
    text: str,
    payload: dict[str, Any],
    collection_name: str | None = None,
) -> None:
    """Embed and index a single document into the Qdrant collection.

    The document text is embedded with both dense and sparse models before
    upserting. Used by the data ingestion pipeline.

    The ``doc_id`` string is converted to a deterministic UUID5 so Qdrant
    receives a valid point ID while preserving the human-readable ID in the
    payload for citation tracing.

    Args:
        doc_id: Human-readable identifier (e.g. ``"1001_chunk_0"``). Stored in
                the payload as ``source_id`` and also used to derive the UUID.
        text: The text content to embed and index.
        payload: Metadata attached to the point (title, source, etc.).
        collection_name: Target collection. Defaults to ``settings.qdrant_collection``.
    """
    name = collection_name or settings.qdrant_collection
    # Qdrant requires UUID or unsigned-int point IDs; derive a deterministic
    # UUID5 from the doc_id string so re-ingestion is idempotent.
    point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

    dense_vector, sparse_vector = await asyncio.gather(
        _embed_dense(text),
        asyncio.get_running_loop().run_in_executor(None, _embed_sparse, text),
    )

    client = _get_qdrant_client()
    point = PointStruct(
        id=point_uuid,
        vector={
            _DENSE_VECTOR_NAME: dense_vector,
            _SPARSE_VECTOR_NAME: sparse_vector,
        },
        payload={**payload, "text": text, "source_id": doc_id},
    )
    await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: client.upsert(collection_name=name, points=[point]),
    )
