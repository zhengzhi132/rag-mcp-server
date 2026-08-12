"""BGE embedding 封装：文档 / 查询两套编码入口。

关键点：
1. BGE 官方检索姿势 —— 文档编码不加 instruction，查询编码加前缀
   "Represent this sentence for searching relevant passages: "。
   因为训练时查询侧带了 instruction、文档侧没带，推理保持一致才能
   让查询向量和文档向量落在同一语义空间。
2. normalize_embeddings=True —— 归一化后 FAISS 的 IndexFlatIP（内积）
   就等于余弦相似度，可以用最朴素的精确索引做检索。
3. 模块级单例 —— 模型是重资源（约 130MB），只加载一次，
   MCP lifespan（P6）复用同一个实例，避免每次 tool call 重载。
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# bge-small-en-v1.5：384 维，英文效果好，体积小
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """懒加载单例：首次调用时下载模型并缓存。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_documents(
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """文档编码（建库用）：不加 instruction，归一化。返回 (n, 384)。"""
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(query: str) -> np.ndarray:
    """查询编码（检索用）：加检索 instruction，归一化。返回 (384,)。"""
    model = get_model()
    vec = model.encode(
        [QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec[0]


if __name__ == "__main__":
    # 冒烟验证：dim=384，相同句子余弦接近 1，不同句子显著更低
    v1 = embed_query("How to create a neural network in PyTorch?")
    v2 = embed_documents(["Create a neural network using torch.nn"])
    v3 = embed_documents(["This text is about cooking recipes."])
    print(f"query dim: {v1.shape}, doc shape: {v2.shape}")
    sim_same = float(np.dot(v1, v2[0]))
    sim_diff = float(np.dot(v1, v3[0]))
    print(f"similar(query, related_doc) = {sim_same:.4f}")
    print(f"similar(query, unrelated_doc) = {sim_diff:.4f}")
    assert sim_same > 0.5, "相关文档余弦应显著高于无关文档"
