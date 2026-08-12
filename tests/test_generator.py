"""Generator 核心逻辑契约测试：引用解析 + 无答案降级。

骨架先定义接口契约（RED），用户实现 _extract_citations / _low_confidence
到全部通过（GREEN）。测试只钉"行为形状"（明显强/明显弱），阈值本身由用户
对着真实语料校准，不写死在用例里。
"""

from __future__ import annotations

from ragmcp.generation.generator import _extract_citations, _low_confidence
from ragmcp.ingestion.chunker import Chunk


def _chunk() -> Chunk:
    return Chunk(
        text="sample",
        source="torch.nn.Linear.html",
        section="torch.nn.Linear",
        index=0,
        chunk_id="abc123",
    )


class TestExtractCitations:
    def test_simple_brackets(self):
        assert _extract_citations("use [1] and [3].", 5) == [1, 3]

    def test_out_of_range_dropped(self):
        assert _extract_citations("see [4] and [9].", 5) == [4]

    def test_non_numeric_ignored(self):
        assert _extract_citations("see [abc] and [2].", 5) == [2]

    def test_duplicates_deduped(self):
        assert _extract_citations("[1] ... again [1].", 5) == [1]

    def test_no_citations(self):
        assert _extract_citations("no brackets here.", 5) == []


class TestLowConfidence:
    def test_empty_hits_refuses(self):
        assert _low_confidence([]) is True

    def test_zero_score_refuses(self):
        assert _low_confidence([(0.0, _chunk())]) is True

    def test_strong_hit_not_refused(self):
        assert _low_confidence([(0.99, _chunk())]) is False
