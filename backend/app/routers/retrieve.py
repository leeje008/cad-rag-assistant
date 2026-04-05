from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..models.schemas import RetrieveRequest, RetrieveResponse
from ..services.retriever import retrieve

router = APIRouter()


@router.post("/api/retrieve", response_model=RetrieveResponse)
async def run_retrieve(request: RetrieveRequest, http_request: Request):
    table = http_request.app.state.table
    client = http_request.app.state.http
    if table is None:
        raise HTTPException(status_code=503, detail="LanceDB table not initialised")

    sources = await retrieve(client, table, request.query, request.k)
    return RetrieveResponse(sources=sources)
