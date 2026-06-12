"""FastAPI web UI for the HybridRAG system.

Run from the project root:
    uvicorn scripts.web_ui:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

# Ensure project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agents.usage import get_llm_token_count
from workflows import v1_workflow, v2_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HybridRAG")

_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_HTML.read_text(encoding="utf-8"))


class QueryRequest(BaseModel):
    query: str
    variant: Literal["v1", "v2"]


@app.post("/query")
async def run_query(req: QueryRequest):
    try:
        if req.variant == "v1":
            result = await v1_workflow.run(req.query)
        else:
            result = await v2_workflow.run(req.query)
        return {
            "answer": result.answer,
            "activated_agents": result.activated_agents,
            "iterations": result.iterations,
            "latency_seconds": result.latency_seconds,
            "cost_usd": result.cost_usd,
        }
    except Exception as exc:
        logger.exception("Workflow error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _v1_stream(query: str) -> AsyncGenerator[str, None]:
    try:
        async for event in v1_workflow.run_streaming(query):
            evt = event["event"]
            if evt == "result":
                data = event["data"]
                total_tokens = get_llm_token_count()
                yield _sse("result", {
                    "answer": data["answer"],
                    "activated_agents": data["activated_agents"],
                    "metrics": {
                        "total_latency_seconds": round(data["latency_seconds"], 2),
                        "total_tokens": total_tokens,
                        "iterations": 1,
                        "cost_usd": round(data["cost_usd"], 6),
                    },
                })
            else:
                payload = {k: v for k, v in event.items() if k != "event"}
                yield _sse(evt, payload)
    except Exception as exc:
        logger.exception("V1 stream error")
        yield _sse("error", {"detail": str(exc)})


async def _v2_stream(query: str) -> AsyncGenerator[str, None]:
    try:
        async for event in v2_workflow.run_streaming(query):
            evt = event["event"]
            if evt == "result":
                data = event["data"]
                total_tokens = get_llm_token_count()
                yield _sse("result", {
                    "answer": data["answer"],
                    "activated_agents": data["activated_agents"],
                    "metrics": {
                        "total_latency_seconds": round(data["latency_seconds"], 2),
                        "total_tokens": total_tokens,
                        "iterations": data["iterations"],
                        "cost_usd": round(data["cost_usd"], 6),
                    },
                })
            else:
                payload = {k: v for k, v in event.items() if k != "event"}
                yield _sse(evt, payload)

    except Exception as exc:
        logger.exception("V2 stream error")
        yield _sse("error", {"detail": str(exc)})


@app.get("/api/query")
async def stream_query(
    q: str = Query(...),
    variant: Literal["v1", "v2"] = Query("v1"),
):
    gen = _v1_stream(q) if variant == "v1" else _v2_stream(q)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
