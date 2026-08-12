"""chunker 单测：钉死分块边界行为。

用 pytest 跑：.venv/Scripts/python.exe -m pytest tests/test_chunker.py -v
"""

from __future__ import annotations

from ragmcp.ingestion.chunker import DEFAULT_SEPS, _split_long, _split_no_overlap, chunk_document
from ragmcp.ingestion.loader import ApiEntry, Document


def make_doc(entry_bodies: list[str], source: str = "test.html") -> Document:
    """构造 Document：signature 用 API 名，body 放正文。"""
    entries = [
        ApiEntry(signature=f"torch.fn{i}()", body=body)
        for i, body in enumerate(entry_bodies)
    ]
    return Document(source=source, title="test", entries=entries)


def test_short_entry_stays_one_chunk():
    doc = make_doc(["computes the sum of all elements."])
    chunks = chunk_document(doc, chunk_size=600, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].section == "torch.fn0"
    assert "sum of all elements" in chunks[0].text


def test_long_entry_gets_split_recursively():
    long_body = "paragraph one. " * 300  # ~4200 字符，远超 600
    doc = make_doc([long_body])
    chunks = chunk_document(doc, chunk_size=600, overlap=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 600 + 100


def test_chunk_id_is_stable_across_calls():
    doc = make_doc(["short body."])
    c1 = chunk_document(doc)[0]
    c2 = chunk_document(doc)[0]
    assert c1.chunk_id == c2.chunk_id
    assert len(c1.chunk_id) == 16


def test_overlap_preserves_context():
    text = "word " * 300  # 1800 字符
    chunks = _split_long(text, chunk_size=600, overlap=100, seps=(" ",))
    if len(chunks) > 1:
        # 相邻块共享 overlap：后块开头 == 前块尾部 overlap 字符
        assert chunks[1].startswith(chunks[0][-100:])


def test_split_preserves_all_content():
    # 新分隔符层级（含句号/逗号）递归切分，不应丢失任何字符
    text = "The quick brown fox jumps over the lazy dog. " * 60
    blocks = _split_no_overlap(text, chunk_size=300, seps=DEFAULT_SEPS)
    assert "".join(blocks) == text
