"""hybrid（混合检索）单测：RRF 融合 + 去重。

用 pytest 跑：.venv/Scripts/python.exe -m pytest tests/test_hybrid.py -v

用手工构造的小向量场景测试融合逻辑，不依赖真实 embedding——
向量用单位向量，query_emb 直接选某个 chunk 的向量，保证向量路确定性。
"""

from __future__ import annotations

import numpy as np

from ragmcp.ingestion.chunker import Chunk
from ragmcp.retrieval.hybrid import HybridRetriever
from ragmcp.retrieval.keyword import KeywordIndex
from ragmcp.storage.faiss_store import FaissStore


def make_fixture() -> tuple[HybridRetriever, list[Chunk]]:
    chunks = [
        Chunk(
            text="CrossEntropyLoss is a loss function for classification.",
            source="a.html",
            section="CrossEntropyLoss",
            index=0,
            chunk_id="c0",
        ),
        Chunk(
            text="DataLoader loads data in batches.",
            source="b.html",
            section="DataLoader",
            index=1,
            chunk_id="c1",
        ),
        Chunk(
            text="Adam is an optimizer for training.",
            source="c.html",
            section="Adam",
            index=2,
            chunk_id="c2",
        ),
    ]
    # 单位向量：query [1,0,0,0] 必命中 c0，[0,1,0,0] 必命中 c1，[0,0,1,0] 必命中 c2
    embeddings = np.array(
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
        dtype="float32",
    )
    fs = FaissStore(4)
    fs.add(embeddings, chunks)
    kw = KeywordIndex([c.text for c in chunks])
    return HybridRetriever(fs, kw, chunks), chunks


def test_rrf_ranks_two_channel_hit_first():
    """双路（向量+BM25）都命中的 chunk，RRF 分数应最高。"""
    hr, chunks = make_fixture()
    qemb = np.array([1.0, 0, 0, 0], dtype="float32")  # 向量路命中 c0
    hits = hr.search("CrossEntropyLoss", qemb, k=3, method="rrf")  # BM25 也命中 c0
    assert hits[0][1].chunk_id == "c0"
    # 两路证据叠加 → c0 分数明显高于单路命中的
    assert hits[0][0] > hits[1][0]


def test_hybrid_no_duplicate_chunk_ids():
    """融合去重后，返回的 chunk_id 不应重复。"""
    hr, chunks = make_fixture()
    qemb = np.array([0, 0, 1.0, 0], dtype="float32")  # 向量命中 c2
    hits = hr.search("optimizer", qemb, k=3)  # BM25 也命中 c2
    ids = [c.chunk_id for _, c in hits]
    assert len(ids) == len(set(ids))


def test_weighted_fusion_also_works():
    """加权融合（weighted）同样能工作，两路命中的 chunk 排第一。"""
    hr, chunks = make_fixture()
    qemb = np.array([1.0, 0, 0, 0], dtype="float32")
    hits = hr.search("CrossEntropyLoss", qemb, k=3, method="weighted", weight=0.7)
    assert hits[0][1].chunk_id == "c0"


def test_pool_greater_than_k():
    """候选池应大于返回数，给融合留空间。"""
    hr, chunks = make_fixture()
    qemb = np.array([1.0, 0, 0, 0], dtype="float32")
    hits = hr.search("CrossEntropyLoss", qemb, k=2, pool=10)
    assert len(hits) <= 2
