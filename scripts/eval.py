"""Automated RAG evaluation with Ragas (reference-free metrics) and a
simple retrieval hit-rate scorer against the manual dataset.

Usage:
    python scripts/eval.py
    python scripts/eval.py --dataset eval/dataset.yaml --judge ollama
    python scripts/eval.py --only-retrieval   # skip LLM generation + ragas
    python scripts/eval.py --limit 5          # first N questions only
    python scripts/eval.py --no-ragas         # skip ragas, keep hit-rate and chat gen

Outputs:
    eval/results_<timestamp>.json   per-question detail
    eval/results_<timestamp>.csv    flat CSV
    stdout summary                  aggregated metrics
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402
import yaml  # noqa: E402

from app.services.llm import stream_chat  # noqa: E402
from app.services.query_rewriter import expand_queries  # noqa: E402
from app.services.retriever import format_context, retrieve_hybrid  # noqa: E402
from app.services.settings import settings  # noqa: E402
from app.services.vectorstore import open_or_create_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval")


# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a YAML list, got {type(data).__name__}")
    return data


def retrieval_hit(
    sources: list[Any], expected_doc: str | None, expected_section: str | None
) -> bool:
    if not expected_doc and not expected_section:
        return True  # no expectation → not evaluated
    for s in sources:
        if expected_doc and expected_doc.lower() not in (s.document or "").lower():
            continue
        if expected_section:
            sec = (s.section or "").lower()
            if expected_section.lower() not in sec:
                continue
        return True
    return False


async def _collect_stream(client, query: str, context: str) -> str:
    """Run the full chat stream into a single string (no `s:` prefix here)."""

    out: list[str] = []
    async for line in stream_chat(client, query, context=context):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk = data.get("message", {}).get("content", "")
        if chunk:
            out.append(chunk)
        if data.get("done"):
            break
    return "".join(out)


# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    dataset = load_dataset(dataset_path)
    if args.limit:
        dataset = dataset[: args.limit]
    logger.info("loaded %d items from %s", len(dataset), dataset_path)

    table = open_or_create_table()
    logger.info("LanceDB rows=%d", table.count_rows())

    timeout = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, item in enumerate(dataset, start=1):
            qid = item.get("id") or f"q-{i}"
            question = item["question"]
            expected_doc = item.get("expected_doc")
            expected_section = item.get("expected_section")
            reference = item.get("reference")

            t0 = time.perf_counter()
            variants = await expand_queries(client, question)
            seen: dict[str, Any] = {}
            order: list[str] = []
            for q in variants:
                hits = await retrieve_hybrid(
                    client,
                    table,
                    q,
                    k=settings.final_top_k,
                    candidate_n=settings.rerank_candidate_n,
                )
                for s in hits:
                    key = s.chunk_id or f"{s.document}:{s.page}"
                    if key not in seen:
                        seen[key] = s
                        order.append(key)
            sources = [seen[k] for k in order][: settings.final_top_k]
            retrieval_ms = (time.perf_counter() - t0) * 1000

            hit = retrieval_hit(sources, expected_doc, expected_section)

            answer = ""
            generation_ms = 0.0
            if not args.only_retrieval:
                context = format_context(sources)
                t1 = time.perf_counter()
                try:
                    answer = await _collect_stream(client, question, context)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("generation failed for %s: %s", qid, exc)
                    answer = ""
                generation_ms = (time.perf_counter() - t1) * 1000

            logger.info(
                "[%d/%d] %s hit=%s ret=%dms gen=%dms",
                i,
                len(dataset),
                qid,
                hit,
                int(retrieval_ms),
                int(generation_ms),
            )

            results.append(
                {
                    "id": qid,
                    "question": question,
                    "expected_doc": expected_doc,
                    "expected_section": expected_section,
                    "reference": reference,
                    "retrieved": [
                        {
                            "document": s.document,
                            "page": s.page,
                            "section": s.section,
                            "relevance": s.relevance,
                            "chunk_id": s.chunk_id,
                        }
                        for s in sources
                    ],
                    "retrieved_text": [s.text for s in sources],
                    "answer": answer,
                    "retrieval_hit": hit,
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                }
            )

    # ------------------------------------------------------------------ Ragas
    ragas_scores: dict[str, float] = {}
    if not args.only_retrieval and not args.no_ragas:
        try:
            ragas_scores = await _run_ragas(results, judge=args.judge)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ragas evaluation failed: %s", exc)

    # ------------------------------------------------------------- Persist
    out_dir = REPO_ROOT / "eval"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    json_path = out_dir / f"results_{stamp}.json"
    csv_path = out_dir / f"results_{stamp}.csv"

    summary = {
        "total": len(results),
        "hit_rate": (
            sum(1 for r in results if r["retrieval_hit"]) / max(1, len(results))
        ),
        "avg_retrieval_ms": sum(r["retrieval_ms"] for r in results) / max(1, len(results)),
        "avg_generation_ms": sum(r["generation_ms"] for r in results) / max(1, len(results)),
        "ragas": ragas_scores,
        "embed_model": settings.embed_model,
        "llm_model": settings.llm_model,
        "use_hybrid": settings.use_hybrid,
        "use_reranker": settings.use_reranker,
        "use_multi_query": settings.use_multi_query,
    }

    json_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["id", "question", "hit", "expected_doc", "retrieved_docs", "retrieval_ms", "generation_ms"]
        )
        for r in results:
            retrieved_docs = ";".join(f"{s['document']}#p{s['page']}" for s in r["retrieved"])
            writer.writerow(
                [
                    r["id"],
                    r["question"],
                    r["retrieval_hit"],
                    r["expected_doc"] or "",
                    retrieved_docs,
                    r["retrieval_ms"],
                    r["generation_ms"],
                ]
            )

    print("\n=== EVAL SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nresults → {json_path.relative_to(REPO_ROOT)}")
    print(f"csv     → {csv_path.relative_to(REPO_ROOT)}")
    return 0


async def _run_ragas(
    results: list[dict[str, Any]],
    *,
    judge: str,
) -> dict[str, float]:
    """Ragas faithfulness + answer_relevancy on generated answers.

    Judge LLM options:
      - "ollama": uses settings.llm_model via Ollama's OpenAI-compat API
      - "openai": uses OPENAI_API_KEY + OPENAI_MODEL env vars
    """

    try:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness
    except ImportError as exc:
        logger.warning("ragas unavailable: %s", exc)
        return {}

    samples: list[SingleTurnSample] = []
    for r in results:
        if not r.get("answer") or not r.get("retrieved_text"):
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r["retrieved_text"],
                response=r["answer"],
            )
        )
    if not samples:
        return {}

    if judge == "openai":
        import os

        chat_llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    else:
        chat_llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",
            temperature=0,
        )

    judge_llm = LangchainLLMWrapper(chat_llm)
    dataset = EvaluationDataset(samples=samples)

    try:
        report = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=judge_llm,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ragas evaluate() raised: %s", exc)
        return {}

    scores: dict[str, float] = {}
    for metric_name in ("faithfulness", "answer_relevancy"):
        try:
            scores[metric_name] = float(report[metric_name])  # type: ignore[index]
        except Exception:  # noqa: BLE001
            pass
    return scores


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/dataset.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--judge", choices=["ollama", "openai"], default="ollama", help="Ragas judge LLM"
    )
    ap.add_argument("--only-retrieval", action="store_true", help="skip LLM generation + ragas")
    ap.add_argument("--no-ragas", action="store_true", help="skip ragas but keep LLM generation")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
