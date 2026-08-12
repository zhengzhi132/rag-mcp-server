"""Chroma 存储：元数据 / 持久化层。

Chroma 负责"持久化 + 过滤"：数据落盘、支持 where 元数据过滤、
自带 collection 管理。建库时与 FAISS 双写同一批 chunk（相同 chunk_id），
作为一致性校验的对照库。

为什么还需要 Chroma（不是只有 FAISS）？
- FAISS 无持久化、无元数据过滤能力
- Chroma 落盘 + where 过滤 + 集合管理
两者分工：FAISS 主检索，Chroma 持久化/过滤/备份。面试可讲"双库取舍"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, directory: Path, collection: str = "pytorch_docs") -> None:
        self._client = chromadb.PersistentClient(
            path=str(directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._name = collection
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._col.add(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings.astype("float32").tolist(),
            metadatas=metadatas,
        )

    def query(
        self,
        query_emb: np.ndarray,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """向量查询，可选 where 过滤（如 {"source": "torch.nn.Linear.html"}）。"""
        return self._col.query(
            query_embeddings=[query_emb.astype("float32").tolist()],
            n_results=k,
            where=where,
        )

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        """删除并重建 collection（保证构建脚本幂等）。"""
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )
