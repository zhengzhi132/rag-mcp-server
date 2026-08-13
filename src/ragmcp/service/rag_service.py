"""服务编排：一次加载，暴露 search / ask / list_topics。

RagService 把「检索 + 生成」编排起来，是 CLI demo 与 P6 MCP Server 的共用
入口。模型与索引都是重资源，懒加载一次缓存，避免每次调用都重新读盘 / 重
新加载 embedding 模型。
"""

from __future__ import annotations

from ragmcp.config import settings
from ragmcp.generation.generator import Generator
from ragmcp.ingestion.chunker import Chunk
from ragmcp.ingestion.embedder import embed_query
from ragmcp.retrieval.hybrid import HybridRetriever
from ragmcp.retrieval.keyword import KeywordIndex
from ragmcp.storage.faiss_store import FaissStore


class RagService:
    def __init__(self) -> None:
        self._faiss: FaissStore | None = None
        self._chunks: list[Chunk] = []
        self._retriever: HybridRetriever | None = None
        self._generator = Generator()

    def ensure_loaded(self) -> None:
        """懒加载：索引 + BM25 + 混合检索器，只做一次。

        公开给 lifespan / 需要预热的调用方。BGE 模型本身是模块级单例
        （embedder.get_model），这里管的是索引与检索器的内存缓存。
        """
        if self._faiss is not None:
            return
        store = FaissStore(dim=384)
        store.load(settings.faiss_dir)
        self._chunks = store.meta
        # KeywordIndex 与 chunks 按位置一一对应（faiss meta 保持建库时的顺序）
        kw = KeywordIndex([c.text for c in self._chunks])
        self._retriever = HybridRetriever(store, kw, self._chunks)
        self._faiss = store

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """向量 + BM25 混合检索 top-k，返回来源信息与摘要片段。"""
        self.ensure_loaded()
        k = k or settings.top_k_default
        hits = self._retriever.search(query, embed_query(query), k=k, pool=30)
        return [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "section": c.section,
                "score": round(float(s), 4),
                "snippet": c.text[:200],
            }
            for s, c in hits
        ]

    def ask(self, question: str, k: int | None = None) -> dict:
        """端到端问答：检索 → 生成带引用回答。"""
        self.ensure_loaded()
        k = k or settings.top_k_default
        hits = self._retriever.search(question, embed_query(question), k=k, pool=30)
        result = self._generator.generate(question, hits)
        return {
            "question": question,
            "answer": result.answer,
            "refused": result.refused,
            "sources": [
                {
                    "n": s.n,
                    "source": s.source,
                    "section": s.section,
                    "score": round(s.score, 4),
                }
                for s in result.sources
            ],
        }

    def list_topics(self) -> list[str]:
        """知识库覆盖的 API 主题（去重排序），供 MCP list_topics 使用。"""
        self.ensure_loaded()
        return sorted({c.section for c in self._chunks})

    def chunk_text(self, chunk_id: str) -> str:
        """按 chunk_id 取完整文本，供引用验证等需要完整上下文的场景。"""
        self.ensure_loaded()
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c.text
        return ""

    def search_section(self, section: str, k: int = 5) -> list[dict]:
        """按 API 主题（section）精确检索，返回该主题下的 chunks。

        供 Agent 的 search_section 工具使用：已知确切主题（如从 list_topics
        拿到）时，按主题精确定位，不受模糊查询召回干扰。score 置 0（精确命中，
        无排序意义）。
        """
        self.ensure_loaded()
        hits = [c for c in self._chunks if c.section == section]
        return [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "section": c.section,
                "score": 0.0,
                "snippet": c.text[:200],
            }
            for c in hits[:k]
        ]

    def stats(self) -> dict:
        """知识库统计：分块数、文档数、主题数、向量维度。"""
        self.ensure_loaded()
        return {
            "n_chunks": len(self._chunks),
            "n_sources": len({c.source for c in self._chunks}),
            "n_topics": len({c.section for c in self._chunks}),
            "dim": self._faiss.index.d if self._faiss is not None else 0,
        }
