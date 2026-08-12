"""双写编排：加载 → 分块 → embedding → 双写 FAISS+Chroma → 一致性校验。

这是"为什么双向量库"的实证：两个库对同一批查询返回的 top-k chunk_id
重合率应 ≥0.9，证明双库 id 对齐可靠、结果一致。
"""

from __future__ import annotations

import time
from pathlib import Path

from ragmcp.ingestion.chunker import chunk_document
from ragmcp.ingestion.embedder import embed_documents, embed_query
from ragmcp.ingestion.loader import load
from ragmcp.storage.chroma_store import ChromaStore
from ragmcp.storage.faiss_store import FaissStore

CONSISTENCY_QUERIES: list[str] = [
    "How to create a Linear layer in PyTorch?",
    "What does CrossEntropyLoss do?",
    "Explain Adam optimizer",
    "How to load data with DataLoader?",
    "What is the difference between view and reshape?",
]


def _check_consistency(
    faiss_store: FaissStore,
    chroma: ChromaStore,
    queries: list[str],
    k: int = 5,
) -> float:
    """双库 top-k 重合率：对每个查询算两库命中并集的 Jaccard，再平均。"""
    ratios: list[float] = []
    for q in queries:
        qe = embed_query(q)
        faiss_ids = {m.chunk_id for _, m in faiss_store.search(qe, k)}
        chroma_ids = set(chroma.query(qe, k=k)["ids"][0])
        union = faiss_ids | chroma_ids
        if not union:
            continue
        ratios.append(len(faiss_ids & chroma_ids) / len(union))
    return sum(ratios) / len(ratios) if ratios else 0.0


def build_index(
    raw_dir: Path,
    faiss_dir: Path,
    chroma_dir: Path,
    chunk_size: int = 600,
    overlap: int = 100,
) -> dict:
    t0 = time.time()

    # 1. 加载 + 分块（html/pdf 存在 raw 的子目录，如 raw/html/，需递归查找）
    docs = []
    for ext in ("html", "pdf"):
        for path in raw_dir.glob(f"**/*.{ext}"):
            docs.append(load(path))
    chunks = [c for doc in docs for c in chunk_document(doc, chunk_size, overlap)]

    # 2. embedding（批量）
    texts = [c.text for c in chunks]
    embeddings = embed_documents(texts)
    dim = embeddings.shape[1]

    # 3. FAISS 主检索（chunks 本身就是 Chunk，直接复用，无需转成 meta）
    faiss_store = FaissStore(dim)
    faiss_store.add(embeddings, chunks)
    faiss_store.save(faiss_dir)

    # 4. Chroma 持久化
    chroma = ChromaStore(chroma_dir)
    chroma.reset()
    chroma.add(
        [c.chunk_id for c in chunks],
        texts,
        embeddings,
        [{"source": c.source, "section": c.section, "index": c.index} for c in chunks],
    )

    # 5. 双库一致性校验
    overlap_ratio = _check_consistency(faiss_store, chroma, CONSISTENCY_QUERIES)

    return {
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "dim": dim,
        "faiss_entries": len(faiss_store),
        "chroma_entries": chroma.count(),
        "dual_store_overlap": round(overlap_ratio, 4),
        "build_seconds": round(time.time() - t0, 2),
    }
