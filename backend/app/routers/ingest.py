from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ..models.schemas import IngestJobStatus, IngestRequest
from ..services.ingest import INGEST_JOBS, create_job, run_ingest
from ..services.vectorstore import count_rows

router = APIRouter()


@router.post("/api/ingest", response_model=IngestJobStatus)
async def start_ingest(
    request: IngestRequest,
    background: BackgroundTasks,
    http_request: Request,
):
    client = http_request.app.state.http
    table = http_request.app.state.table
    if table is None:
        raise HTTPException(status_code=503, detail="LanceDB table not initialised")

    job = create_job(request.path)
    background.add_task(
        run_ingest,
        job.job_id,
        request.path,
        client,
        table,
        recursive=request.recursive,
    )
    return IngestJobStatus(
        job_id=job.job_id,
        status=job.status,
        processed=job.processed,
        total=job.total,
        errors=job.errors,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/api/ingest/{job_id}", response_model=IngestJobStatus)
async def get_job(job_id: str):
    job = INGEST_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return IngestJobStatus(
        job_id=job.job_id,
        status=job.status,
        processed=job.processed,
        total=job.total,
        errors=job.errors,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/api/ingest")
async def list_jobs(http_request: Request):
    table = http_request.app.state.table
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "processed": j.processed,
                "total": j.total,
                "path": j.path,
            }
            for j in INGEST_JOBS.values()
        ],
        "table_rows": count_rows(table),
    }
