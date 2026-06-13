"""Build the ColQwen page-image sidecar index (C6 pilot).

Restricted to explicitly listed PDF files — whole-corpus page embedding is
deliberately unsupported (~1,030 vectors per page; local memory can't carry
the full SPEC folder). Point this at the P&ID binder / key drawing files
only, then set USE_COLQWEN_PAGES=true.

Each run rebuilds the pilot index from scratch (manifest is overwritten).
Page PNGs go to settings.assets_dir so /api/assets and VL routing reuse them.

Usage:
    python scripts/build_colqwen_index.py "SPEC/P&ID Symbol & Legend.pdf" [more.pdf ...]
    python scripts/build_colqwen_index.py --dpi 150 --batch 2 SPEC/drawing.pdf

Requires the optional pilot deps (see requirements.txt):
    pip install colpali-engine torch
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.services.parsers import _doc_id  # noqa: E402
from app.services.settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_colqwen_index")


def _render_pages(pdf_path: Path, dpi: int) -> list[tuple[int, bytes]]:
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = [(page.number + 1, page.get_pixmap(dpi=dpi).tobytes("png")) for page in doc]
    doc.close()
    return pages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="PDF files to index (explicit list only)")
    ap.add_argument("--dpi", type=int, default=150, help="page render DPI")
    ap.add_argument("--batch", type=int, default=2, help="pages per embed batch")
    args = ap.parse_args()

    try:
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        from PIL import Image
    except ImportError as exc:
        logger.error("optional pilot deps missing (%s) — pip install colpali-engine torch", exc)
        return 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("loading %s on %s", settings.colqwen_model, device)
    model = ColQwen2.from_pretrained(
        settings.colqwen_model, torch_dtype=torch.float16, device_map=device
    ).eval()
    processor = ColQwen2Processor.from_pretrained(settings.colqwen_model)

    settings.colqwen_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for f in args.files:
        path = Path(f)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.suffix.lower() != ".pdf" or not path.is_file():
            logger.warning("skipping %s (not a PDF file)", path)
            continue

        doc_id = _doc_id(path)
        (settings.colqwen_dir / doc_id).mkdir(parents=True, exist_ok=True)
        (settings.assets_dir / doc_id).mkdir(parents=True, exist_ok=True)

        pages = _render_pages(path, args.dpi)
        logger.info("%s → %d pages (doc_id=%s)", path.name, len(pages), doc_id)

        for start in range(0, len(pages), args.batch):
            chunk = pages[start : start + args.batch]
            images = [Image.open(io.BytesIO(png)) for _, png in chunk]
            batch_inputs = processor.process_images(images).to(device)
            with torch.no_grad():
                embeddings = model(**batch_inputs)
            for (page_no, png), emb in zip(chunk, embeddings, strict=True):
                png_rel = f"{doc_id}/page_{page_no}.png"
                (settings.assets_dir / png_rel).write_bytes(png)
                tensor_rel = f"{doc_id}/page_{page_no}.pt"
                torch.save(
                    emb.to(torch.float16).cpu(), settings.colqwen_dir / tensor_rel
                )
                manifest.append(
                    {
                        "doc_id": doc_id,
                        "page": page_no,
                        "title": path.stem,
                        "source_path": str(path),
                        "image_path": png_rel,
                        "tensor": tensor_rel,
                    }
                )
            logger.info("  embedded pages %d-%d", chunk[0][0], chunk[-1][0])

    (settings.colqwen_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "done: %d pages indexed → %s (enable with USE_COLQWEN_PAGES=true)",
        len(manifest),
        settings.colqwen_dir / "manifest.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
