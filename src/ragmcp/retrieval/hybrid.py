"""混合检索：向量（FAISS）+ 关键词（BM25）融合。

为什么不能直接加权求和？
FAISS 余弦分数在 -1~1，BM25 分数是 0~几十，量纲完全不同，直接
w*vec + (1-w)*kw 没有意义。两个方案：

1. RRF（Reciprocal Rank Fusion，推荐）：
   score(chunk) = Σ 1/(rrf_k + rank)
   只看排名不看分数，排名无量纲，天然规避两路分数不可比的问题。
   rrf_k=60 出自 Cormack 2009 论文的经典值。

2. 加权融合（weighted）：
   先把两路分数各自 min-max 归一化到 0~1，再 w*vec + (1-w)*kw。
   保留分数信息，但引入归一化偏差，且对权重敏感。

默认用 RRF：跨打分器鲁棒、无参数敏感性、实现简单。
"""

from __future__ import annotations

from typing import Literal

from ragmcp.ingestion.chunker import Chunk
from ragmcp.retrieval.keyword import KeywordIndex
from ragmcp.storage.faiss_store import FaissStore

Method = Literal["rrf", "weighted"]


def _minmax(value: float, pool: list[float]) -> float:
    """min-max 归一化到 0~1（池内）。"""
    lo, hi = min(pool), max(pool)
    if hi - lo < 1e-9:
        return 0.5
    return (value - lo) / (hi - lo)


class HybridRetriever:
    def __init__(
        self,
        faiss_store: FaissStore,
        keyword_index: KeywordIndex,
        chunks: list[Chunk],
    ) -> None:
        """chunks 与 keyword_index 的 texts 按位置一一对应。"""
        self._faiss = faiss_store
        self._kw = keyword_index
        self._chunks = chunks

    def search(
        self,
        query_text: str,
        query_emb: object,
        k: int = 5,
        pool: int = 30,
        method: Method = "rrf",
        rrf_k: int = 60,
        weight: float = 0.7,
    ) -> list[tuple[float, Chunk]]:
        """双路各取 top-pool（>k），融合后返回 top-k。返回 [(fused_score, Chunk)]。"""
        # 1. 双路候选（池要比 k 大，给融合留选择空间）
        vec_hits = self._faiss.search(query_emb, pool)  # [(score, Chunk)]
        kw_hits = self._kw.search(query_text, pool)  # [(score, doc_index)]

        # 2. 按 chunk_id 组织两路排名/分数
        vec_rank: dict[str, int] = {c.chunk_id: r + 1 for r, (_, c) in enumerate(vec_hits)}
        kw_rank: dict[str, int] = {
            self._chunks[i].chunk_id: r + 1 for r, (_, i) in enumerate(kw_hits)
        }
        vec_score: dict[str, float] = {c.chunk_id: s for s, c in vec_hits}
        kw_score: dict[str, float] = {self._chunks[i].chunk_id: s for s, i in kw_hits}

        # 3. 融合
        fused: dict[str, float] = {}
        if method == "rrf":
            for cid in set(vec_rank) | set(kw_rank):
                fused[cid] = fused.get(cid, 0.0)
                if cid in vec_rank:
                    fused[cid] += 1.0 / (rrf_k + vec_rank[cid])
                if cid in kw_rank:
                    fused[cid] += 1.0 / (rrf_k + kw_rank[cid])
        else:  # weighted
            v_pool, w_pool = list(vec_score.values()), list(kw_score.values())
            for cid in set(vec_score) | set(kw_score):
                v = _minmax(vec_score.get(cid, 0.0), v_pool)
                w = _minmax(kw_score.get(cid, 0.0), w_pool)
                fused[cid] = weight * v + (1 - weight) * w

        # 4. 取 top-k 并映射回 Chunk
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:k]
        by_id = {c.chunk_id: c for c in self._chunks}
        return [(fused[cid], by_id[cid]) for cid in ranked_ids if cid in by_id]
