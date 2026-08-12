"""FAISS 向量存储：主检索路径。

FAISS 负责"快"：精确余弦扫描（IndexFlatIP），内存省，本身无持久化。
元数据（即 Chunk 本身）放在内存 list，按向量索引位置与 FAISS 对齐；
落盘用 index.faiss + metadata.jsonl。

为什么元数据不进 FAISS？FAISS 索引只存向量和向量间的图结构，
它不知道"这个向量是哪篇文档的哪一段"。

前置约定：写入的 embedding 必须是归一化向量（embedder 已 normalize），
这样 IndexFlatIP 的内积 = 余弦相似度。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from ragmcp.ingestion.chunker import Chunk


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """批量写入：embeddings shape (n, dim)，chunks 与行一一对应。"""
        assert embeddings.shape[0] == len(chunks)
        self.index.add(embeddings.astype("float32"))
        self.meta.extend(chunks)

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[float, Chunk]]:
        """精确余弦 top-k。返回 [(score, Chunk)]，按分数降序。"""
        scores, idxs = self.index.search(query_emb[None, :].astype("float32"), k)
        out: list[tuple[float, Chunk]] = []
        for score, pos in zip(scores[0], idxs[0]):
            if pos < 0:  # FAISS 未找到时位置为 -1
                break
            out.append((float(score), self.meta[int(pos)]))
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.jsonl", "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    def load(self, directory: Path) -> None:
        self.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.jsonl", encoding="utf-8") as f:
            self.meta = [Chunk(**json.loads(line)) for line in f]

    def __len__(self) -> int:
        return self.index.ntotal
    