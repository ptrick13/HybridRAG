"""Ingest Software Development Analytics data into Qdrant.

Populates three collections:
  1. dev_tickets      — ticket title + description, chunked
  2. dev_arch_docs    — ADR / design doc / RFC texts, chunked
  3. dev_postmortems  — postmortem texts, chunked

Token-based sliding window chunking:
- Chunk size: 512 tokens
- Overlap:    64 tokens (≈12.5%)

Uses tiktoken for token counting (cl100k_base) and fastembed BM25 for sparse
vectors. Hybrid dense + sparse vectors enable RRF retrieval.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import tiktoken

from config.settings import settings
from tools.qdrant_client import ensure_collection_exists, index_document

logger = logging.getLogger(__name__)

_CHUNK_SIZE_TOKENS = 512
_CHUNK_OVERLAP_TOKENS = 64
_ENCODING_NAME = "cl100k_base"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    enc = tiktoken.get_encoding(_ENCODING_NAME)
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


async def _ingest_collection(
    items: list[dict[str, Any]],
    collection_name: str,
    text_fn,
    payload_fn,
    id_fn,
) -> None:
    """Generic chunking + indexing loop for one collection."""
    await asyncio.get_running_loop().run_in_executor(
        None, ensure_collection_exists, collection_name
    )
    total_chunks = 0
    failed_chunks = 0

    for item in items:
        text = text_fn(item)
        chunks = await asyncio.get_running_loop().run_in_executor(
            None, _chunk_text, text, _CHUNK_SIZE_TOKENS, _CHUNK_OVERLAP_TOKENS
        )
        for i, chunk in enumerate(chunks):
            try:
                await index_document(
                    doc_id=f"{id_fn(item)}_chunk_{i}",
                    text=chunk,
                    payload={**payload_fn(item), "chunk_index": i, "total_chunks": len(chunks)},
                    collection_name=collection_name,
                )
                total_chunks += 1
            except Exception as exc:
                failed_chunks += 1
                logger.warning(
                    "Failed to index %s chunk %d (skipping): %s",
                    id_fn(item),
                    i,
                    exc,
                )

        if (total_chunks + failed_chunks) % 50 == 0:
            logger.info(
                "  %s: %d chunks indexed, %d failed …", collection_name, total_chunks, failed_chunks
            )

    if failed_chunks:
        logger.warning(
            "Collection '%s' complete — %d chunks indexed, %d failed.",
            collection_name,
            total_chunks,
            failed_chunks,
        )
    else:
        logger.info("Collection '%s' complete — %d chunks.", collection_name, total_chunks)


async def ingest_from_files(sample_dir: Path) -> None:
    """Run the full Qdrant ingestion pipeline from JSON files."""
    tickets = json.loads((sample_dir / "tickets.json").read_text())
    arch_docs = json.loads((sample_dir / "arch_docs.json").read_text())
    postmortems = json.loads((sample_dir / "postmortems.json").read_text())

    # ── Collection 1: tickets ─────────────────────────────────────────────────
    logger.info("Indexing tickets into '%s' …", settings.qdrant_collection_tickets)
    await _ingest_collection(
        items=tickets,
        collection_name=settings.qdrant_collection_tickets,
        text_fn=lambda t: f"{t['title']}\n\n{t.get('description', '')}",
        payload_fn=lambda t: {
            "ticket_id": t["id"],
            "title": t["title"],
            "component_id": t["component_id"],
            "team_id": t["team_id"],
            "type": t["type"],
            "priority": t["priority"],
            "status": t["status"],
        },
        id_fn=lambda t: t["id"],
    )

    # ── Collection 2: arch docs ───────────────────────────────────────────────
    logger.info("Indexing arch docs into '%s' …", settings.qdrant_collection_arch_docs)
    await _ingest_collection(
        items=arch_docs,
        collection_name=settings.qdrant_collection_arch_docs,
        text_fn=lambda d: f"{d['title']}\n\n{d.get('text', '')}",
        payload_fn=lambda d: {
            "doc_id": d["id"],
            "component_id": d["component_id"],
            "doc_type": d["doc_type"],
            "title": d["title"],
            "date": d["date"],
        },
        id_fn=lambda d: d["id"],
    )

    # ── Collection 3: postmortems ─────────────────────────────────────────────
    logger.info("Indexing postmortems into '%s' …", settings.qdrant_collection_postmortems)
    await _ingest_collection(
        items=postmortems,
        collection_name=settings.qdrant_collection_postmortems,
        text_fn=lambda p: p.get("text", ""),
        payload_fn=lambda p: {
            "pm_id": p["id"],
            "incident_id": p["incident_id"],
            "component_id": p["component_id"],
            "severity": p["severity"],
            "title": p.get("title", ""),
            "date": p["date"],
        },
        id_fn=lambda p: p["id"],
    )

    logger.info("Qdrant ingestion complete.")
