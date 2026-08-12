"""BM25 关键词检索：字面匹配，弥补向量检索的短板。

向量检索擅长"语义相近"，但对专业术语、缩写、编号（如 CrossEntropyLoss、
view(2,3)）召回差——embedding 按语义聚类，字面精确的词不一定排最前。
BM25 是经典关键词匹配，字面命中强，与向量检索互补。

tokenizer 保留下划线：代码标识符（如 CrossEntropyLoss）要当成一个词，
不能按空格拆成 cross entropy loss。
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """英文小写分词，保留下划线与数字。"""
    return re.findall(r"[a-z0-9_]+", text.lower())


class KeywordIndex:
    def __init__(self, texts: list[str]) -> None:
        """texts 与外部 chunks 按位置一一对应（调用方保证）。"""
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 5) -> list[tuple[float, int]]:
        """返回 [(score, doc_index)]，doc_index 是 texts 的位置，分数降序。"""
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(float(scores[i]), i) for i in top]

    def __len__(self) -> int:
        return self._bm25.corpus_size
