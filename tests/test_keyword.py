"""keyword（BM25）单测：分词正确性 + 命中正确性 + 排序正确性。

用 pytest 跑：.venv/Scripts/python.exe -m pytest tests/test_keyword.py -v
"""

from __future__ import annotations

from ragmcp.retrieval.keyword import KeywordIndex, tokenize


def test_tokenize_keeps_identifiers():
    """代码标识符要保留为一个词（下划线/数字不清除）。"""
    assert tokenize("CrossEntropyLoss(view(2, 3))") == [
        "crossentropyloss",
        "view",
        "2",
        "3",
    ]


def test_bm25_hits_matching_doc():
    """查询包含某词的文档应被命中，且 doc_index 正确。"""
    texts = [
        "CrossEntropyLoss is a loss function for classification.",
        "DataLoader loads data in batches.",
        "Adam is an optimizer for training.",
    ]
    ki = KeywordIndex(texts)
    hits = ki.search("CrossEntropyLoss", k=1)
    assert hits[0][1] == 0  # 命中 doc 0


def test_search_returns_sorted_scores():
    """高频命中文档排第一，分数降序。"""
    texts = [
        "apple banana cherry",
        "apple apple apple",
        "nothing relevant here",
    ]
    ki = KeywordIndex(texts)
    hits = ki.search("apple", k=3)
    assert hits[0][1] == 1  # doc 1 命中 apple 最多
    scores = [s for s, _ in hits]
    assert scores == sorted(scores, reverse=True)
