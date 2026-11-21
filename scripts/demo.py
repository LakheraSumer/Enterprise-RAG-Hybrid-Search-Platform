"""Index the sample knowledge base and ask a few questions from the command line.

    python scripts/demo.py
    python scripts/demo.py "How long are silver tables retained?"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.models import DocumentIn  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402

QUESTIONS = [
    "How long do we keep bronze data?",
    "What should on-call do if the nightly load fails because of a timeout?",
    "Which checks must a gold table pass before publishing?",
    "Who won the football world cup?",  # not in the knowledge base -> low-confidence fallback
]


def main() -> None:
    pipeline = RAGPipeline(Settings.from_env())
    docs = json.loads((ROOT / "data" / "sample" / "documents.json").read_text(encoding="utf-8"))
    print(pipeline.ingest([DocumentIn(**d) for d in docs]))
    for question in sys.argv[1:] or QUESTIONS:
        r = pipeline.query(question)
        print(f"\nQ: {question}\nA: {r['answer']}\n   mode={r['answer_mode']} confidence={r['confidence']} "
              f"sources={[c['title'] for c in r['citations']]}")


if __name__ == "__main__":
    main()
