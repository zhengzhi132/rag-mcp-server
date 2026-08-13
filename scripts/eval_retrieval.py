"""检索评估：对比 4 种检索方法在自建评估集上的 HR@k 与 MRR。

评估集：eval/questions.json，每条 {question, gold: [sections]}。
gold 是"正确来源"的 section 名，命中 = top-k 里出现任一 gold section。

4 种方法（一次检索 top-10，同时算 HR@1/5/10 与 MRR@10）：
- vector  : 纯 FAISS 余弦（IndexFlatIP）
- keyword : 纯 BM25
- rrf     : RRF 融合（默认，rrf_k=60）
- weighted: min-max 归一化 + 加权融合

输出对比表，并保存结果到 eval/results/（目录已被 .gitignore 排除）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragmcp.config import settings
from ragmcp.ingestion.chunker import Chunk
from ragmcp.ingestion.embedder import embed_query
from ragmcp.retrieval.hybrid import HybridRetriever
from ragmcp.retrieval.keyword import KeywordIndex
from ragmcp.storage.faiss_store import FaissStore

K = 10  # 单次检索深度，HR@1/5/10 都从这 top-10 里算
POOL = 30
RRF_K = 60


def _load() -> tuple[FaissStore, KeywordIndex, HybridRetriever, list[Chunk]]:
    store = FaissStore(dim=384)
    store.load(settings.faiss_dir)
    chunks = store.meta
    kw = KeywordIndex([c.text for c in chunks])
    return store, kw, HybridRetriever(store, kw, chunks), chunks


def _hit_at(hits: list[tuple[float, Chunk]], gold: set[str], k: int) -> bool:
    return any(c.section in gold for _, c in hits[:k])


def _mrr(hits: list[tuple[float, Chunk]], gold: set[str]) -> float:
    for r, (_, c) in enumerate(hits[:K], 1):
        if c.section in gold:
            return 1.0 / r
    return 0.0


def _method_retrievals(store, kw, hybrid, chunks, q_text, q_emb):
    vec = store.search(q_emb, K)  # [(score, Chunk)]
    keyw = [(s, chunks[i]) for s, i in kw.search(q_text, K)]
    rrf = hybrid.search(q_text, q_emb, K, pool=POOL, method="rrf", rrf_k=RRF_K)
    weighted = hybrid.search(q_text, q_emb, K, pool=POOL, method="weighted")
    return {"vector": vec, "keyword": keyw, "rrf": rrf, "weighted": weighted}


def main() -> None:
    qpath = Path(__file__).resolve().parents[1] / "eval" / "questions.json"
    questions = json.loads(qpath.read_text(encoding="utf-8"))
    store, kw, hybrid, chunks = _load()

    # 每种方法累计每个 query 的 HR@k 与 MRR
    methods = ["vector", "keyword", "rrf", "weighted"]
    n = len(questions)
    acc = {m: {"hits": {k: 0 for k in (1, 5, 10)}, "mrr": 0.0} for m in methods}

    for q in questions:
        q_emb = embed_query(q["question"])
        gold = set(q["gold"])
        for m, hits in _method_retrievals(
            store, kw, hybrid, chunks, q["question"], q_emb
        ).items():
            for k in (1, 5, 10):
                acc[m]["hits"][k] += int(_hit_at(hits, gold, k))
            acc[m]["mrr"] += _mrr(hits, gold)

    print(f"eval set: {n} questions, top-{K}, pool={POOL}, rrf_k={RRF_K}")
    print(f"{'method':<10}{'HR@1':>8}{'HR@5':>8}{'HR@10':>8}{'MRR@10':>8}")
    result = {}
    for m in methods:
        hr = {k: acc[m]["hits"][k] / n for k in (1, 5, 10)}
        mrr = acc[m]["mrr"] / n
        result[m] = {"HR@1": hr[1], "HR@5": hr[5], "HR@10": hr[10], "MRR@10": mrr}
        print(
            f"{m:<10}"
            f"{hr[1]:>8.3f}{hr[5]:>8.3f}{hr[10]:>8.3f}{mrr:>8.3f}"
        )

    result["meta"] = {
        "n_questions": n,
        "top_k": K,
        "pool": POOL,
        "rrf_k": RRF_K,
    }
    out = Path(__file__).resolve().parents[1] / "eval" / "results" / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
