"""Benchmark retrieval quality on BEIR SciFact (5,183 scientific abstracts, 300 test queries with relevance labels).

Compares BM25, dense, hybrid and hybrid + cross-encoder reranking, then checks answer quality (groundedness,
citation hit rate) for the best configuration. Writes results/eval_results.json and results/EVAL.md.

    python eval/evaluate.py                       # semantic embeddings + cross-encoder (downloads ~200 MB of models once)
    python eval/evaluate.py --provider hashing    # offline, no model download
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cache import MemoryTTLCache  # noqa: E402
from app.config import Settings  # noqa: E402
from app.metrics import groundedness, mrr_at_k, ndcg_at_k, percentile, precision_at_k, recall_at_k  # noqa: E402
from app.models import DocumentIn  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402
from app.vector_store import InMemoryVectorStore  # noqa: E402

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
HF_PARQUET = "https://huggingface.co/datasets/BeIR/scifact/resolve/main/{cfg}/{cfg}-00000-of-00001.parquet"
HF_QRELS = "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv"


def _download_from_hugging_face(folder: Path) -> None:
    """Hugging Face mirror of the same BEIR files (parquet). Needs `pip install -r requirements-eval.txt`."""
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    for cfg in ("corpus", "queries"):
        local = folder / f"{cfg}.parquet"
        urllib.request.urlretrieve(HF_PARQUET.format(cfg=cfg), local)
        rows = pq.read_table(local).to_pylist()
        (folder / f"{cfg}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(f"  {cfg}: {len(rows)} rows")
    (folder / "qrels").mkdir(exist_ok=True)
    with urllib.request.urlopen(HF_QRELS, timeout=60) as resp:
        (folder / "qrels" / "test.tsv").write_bytes(resp.read())


def load_scifact(data_dir: Path):
    folder = data_dir / "scifact"
    if not (folder / "corpus.jsonl").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / "scifact.zip"
        try:
            print(f"downloading {SCIFACT_URL} ...")
            with urllib.request.urlopen(SCIFACT_URL, timeout=20) as resp:
                archive.write_bytes(resp.read())
            with zipfile.ZipFile(archive) as z:
                z.extractall(data_dir)
        except OSError as exc:
            print(f"primary source failed ({exc}); using the Hugging Face mirror")
            _download_from_hugging_face(folder)
    corpus = [json.loads(line) for line in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line]
    all_queries = {}
    for line in (folder / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            q = json.loads(line)
            all_queries[q["_id"]] = q["text"]
    qrels: dict[str, set[str]] = {}
    for line in (folder / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        qid, doc_id, score = line.split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, set()).add(doc_id)
    queries = {qid: all_queries[qid] for qid in qrels}
    return corpus, queries, qrels


def unique_docs(results) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r.chunk.doc_id not in seen:
            seen.append(r.chunk.doc_id)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="fastembed", choices=["fastembed", "hashing"])
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "beir"))
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache" / "models"))
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--answer-queries", type=int, default=100)
    args = parser.parse_args()

    use_models = args.provider == "fastembed"
    settings = Settings(
        embedding_provider=args.provider,
        reranker="cross-encoder" if use_models else "none",
        model_cache_dir=args.cache_dir,
        chunk_tokens=256,
        chunk_overlap=32,
        candidate_k=args.candidates,
    )
    corpus, queries, qrels = load_scifact(Path(args.data_dir))
    print(f"corpus={len(corpus)} docs, test queries={len(queries)}")

    pipeline = RAGPipeline(settings, store=InMemoryVectorStore(), cache=MemoryTTLCache())
    docs = [DocumentIn(id=d["_id"], title=d.get("title") or d["_id"], text=d["text"]) for d in corpus if d.get("text")]
    started = time.perf_counter()
    ingest = pipeline.ingest(docs)
    index_seconds = time.perf_counter() - started
    print(f"indexed {ingest['chunks_indexed']} chunks in {index_seconds:.1f}s")

    configs = [("BM25 (keyword)", "sparse", False), ("Dense (vector)", "dense", False), ("Hybrid (BM25 + vector)", "hybrid", False)]
    if pipeline.reranker is not None:
        configs.append(("Hybrid + cross-encoder rerank", "hybrid", True))

    rows = []
    for label, mode, rerank in configs:
        scores = {"P@5": [], "Recall@10": [], "MRR@10": [], "nDCG@10": []}
        latencies = []
        for qid, text in queries.items():
            t0 = time.perf_counter()
            results = pipeline.search(text, top_k=args.candidates, mode=mode, rerank=rerank)
            latencies.append((time.perf_counter() - t0) * 1000)
            ranked, relevant = unique_docs(results), qrels[qid]
            scores["P@5"].append(precision_at_k(ranked, relevant, 5))
            scores["Recall@10"].append(recall_at_k(ranked, relevant, 10))
            scores["MRR@10"].append(mrr_at_k(ranked, relevant, 10))
            scores["nDCG@10"].append(ndcg_at_k(ranked, relevant, 10))
        row = {"config": label, **{k: round(statistics.mean(v), 4) for k, v in scores.items()},
               "p50_ms": round(percentile(latencies, 50), 1), "p95_ms": round(percentile(latencies, 95), 1)}
        rows.append(row)
        print(row)

    best_mode, best_rerank = configs[-1][1], configs[-1][2]
    grounded, hits, fallbacks, answer_ms = [], [], 0, []
    for qid, text in list(queries.items())[: args.answer_queries]:
        response = pipeline.query(text, top_k=5, mode=best_mode, rerank=best_rerank)
        answer_ms.append(response["latency_ms"])
        if response["answer_mode"] == "low-confidence-fallback":
            fallbacks += 1
            continue
        grounded.append(groundedness(response["answer"], [s["text"] for s in response["sources"]]))
        hits.append(any(c["doc_id"] in qrels[qid] for c in response["citations"]))
    answers = {
        "queries": min(args.answer_queries, len(queries)),
        "generator": pipeline.generator.name,
        "groundedness": round(statistics.mean(grounded), 4) if grounded else None,
        "citation_hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
        "low_confidence_fallbacks": fallbacks,
        "p50_ms": round(percentile(answer_ms, 50), 1),
        "llm_tokens": 0,
        "llm_cost_usd": 0.0,
    }
    print(answers)

    report = {
        "dataset": "BEIR SciFact (test split)",
        "documents": len(docs),
        "chunks": ingest["chunks_indexed"],
        "queries": len(queries),
        "embedding_model": pipeline.embedder.name,
        "reranker_model": getattr(pipeline.reranker, "name", None),
        "candidates_per_query": args.candidates,
        "index_seconds": round(index_seconds, 1),
        "machine": f"{platform.system()} {platform.machine()}, CPU only, Python {platform.python_version()}",
        "retrieval": rows,
        "answers": answers,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    suffix = "" if use_models else "_hashing"
    (out / f"eval_results{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Evaluation: {report['dataset']}",
        "",
        f"- Corpus: {report['documents']:,} abstracts -> {report['chunks']:,} chunks; {report['queries']} labelled test queries",
        f"- Embeddings: `{report['embedding_model']}`; reranker: `{report['reranker_model']}`; "
        f"{args.candidates} candidates per query",
        f"- Machine: {report['machine']}; indexing took {report['index_seconds']} s",
        "",
        "| Configuration | P@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency (ms) | p95 latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['config']} | {r['P@5']:.3f} | {r['Recall@10']:.3f} | {r['MRR@10']:.3f} | {r['nDCG@10']:.3f} "
                     f"| {r['p50_ms']} | {r['p95_ms']} |")
    lines += [
        "",
        f"Answer quality ({answers['generator']} generator, first {answers['queries']} queries, best configuration):",
        f"groundedness {answers['groundedness']}, citation hit rate {answers['citation_hit_rate']}, "
        f"low-confidence fallbacks {answers['low_confidence_fallbacks']}, p50 end-to-end {answers['p50_ms']} ms, "
        f"LLM tokens {answers['llm_tokens']} (extractive mode, $0).",
        "",
        "Reproduce with `python eval/evaluate.py`.",
    ]
    (out / f"EVAL{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / f'EVAL{suffix}.md'}")


if __name__ == "__main__":
    main()
