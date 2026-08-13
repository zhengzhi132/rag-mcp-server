"""RagAgent 循环逻辑契约测试：工具调用驱动、引用映射、去重、降级。

用 FakeLLM + FakeService 打桩，不碰真实 DeepSeek / 索引，专注测循环行为。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ragmcp.agent.rag_agent import RagAgent
from ragmcp.generation.generator import REFUSAL_MESSAGE, _extract_citations


def _msg(*, content: str = "", tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


class FakeLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._i = 0

    def invoke(self, messages: list) -> AIMessage:
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r


class FakeService:
    def __init__(
        self,
        results_by_query: dict[str, list[dict]],
        sections: dict[str, list[dict]] | None = None,
    ) -> None:
        self._results = results_by_query
        self._sections = sections or {}

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self._results.get(query, [])

    def search_section(self, section: str, k: int = 5) -> list[dict]:
        return self._sections.get(section, [])

    def list_topics(self) -> list[str]:
        return ["topic_a", "topic_b"]

    def chunk_text(self, chunk_id: str) -> str:
        return f"dummy text for {chunk_id}"


class FakeJudge:
    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = list(verdicts)
        self._i = 0
        self.calls = 0

    def invoke(self, messages: list) -> AIMessage:
        v = self._verdicts[min(self._i, len(self._verdicts) - 1)]
        self._i += 1
        self.calls += 1
        return AIMessage(content=v)


def _hit(query: str, cid: str, section: str) -> dict:
    return {
        "chunk_id": cid,
        "source": "torch.nn.Linear.html",
        "section": section,
        "score": 0.9,
        "snippet": "sample text",
    }


def _agent(
    fake_llm: FakeLLM,
    service: FakeService,
    judge: FakeJudge | None = None,
) -> RagAgent:
    a = RagAgent(k=2, max_iters=3)
    a._llm = fake_llm  # type: ignore[attr-defined]
    a._service = service  # type: ignore[attr-defined]
    a._judge = judge or FakeJudge(["[1] supported: 资料支持该论断"])  # type: ignore[attr-defined]
    return a


def test_direct_answer_no_tools():
    a = _agent(FakeLLM([_msg(content="直接回答。")]), FakeService({}))
    r = a.ask("q")
    assert r.refused is False
    assert r.answer == "直接回答。"
    assert r.sources == []
    assert r.trace == []


def test_search_then_answer_maps_citation():
    llm = FakeLLM(
        [
            _msg(tool_calls=[{"name": "search", "args": {"query": "linear"}, "id": "c1"}]),
            _msg(content="用 [1]。"),
        ]
    )
    svc = FakeService({"linear": [_hit("linear", "cid-1", "class torch.nn. Linear")]})
    r = _agent(llm, svc).ask("How to make a Linear layer?")
    assert r.refused is False
    assert r.answer == "用 [1]。"
    assert len(r.sources) == 1
    assert r.sources[0].chunk_id == "cid-1"
    assert r.sources[0].section == "class torch.nn. Linear"
    assert len(r.trace) == 1
    assert r.trace[0]["action"] == "search"


def test_multi_search_accumulates_and_cites_later():
    llm = FakeLLM(
        [
            _msg(tool_calls=[{"name": "search", "args": {"query": "q1"}, "id": "c1"}]),
            _msg(tool_calls=[{"name": "search", "args": {"query": "q2"}, "id": "c2"}]),
            _msg(content="结合 [2]。"),
        ]
    )
    svc = FakeService(
        {"q1": [_hit("q1", "cid-1", "sec1")], "q2": [_hit("q2", "cid-2", "sec2")]}
    )
    r = _agent(llm, svc).ask("multi")
    assert len(r.trace) == 2
    assert [s.chunk_id for s in r.sources] == ["cid-2"]  # [2] → 第二个来源
    assert r.sources[0].n == 2


def test_search_dedup_across_calls():
    llm = FakeLLM(
        [
            _msg(tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "c1"}]),
            _msg(tool_calls=[{"name": "search", "args": {"query": "y"}, "id": "c2"}]),
            _msg(content="[1]"),
        ]
    )
    svc = FakeService({"x": [_hit("x", "cid-1", "sec1")], "y": [_hit("y", "cid-1", "sec1")]})
    r = _agent(llm, svc).ask("dup")
    assert len(r.sources) == 1  # 同一 chunk 去重
    assert r.sources[0].chunk_id == "cid-1"


def test_refuses_when_max_iters_exhausted():
    llm = FakeLLM([_msg(tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "c1"}])])
    svc = FakeService({})  # 检索无结果，模型只会继续调工具
    a = _agent(llm, svc)
    a._max_iters = 2
    r = a.ask("no answer")
    assert r.refused is True
    assert r.answer == REFUSAL_MESSAGE
    assert len(r.trace) == 2  # 两轮工具调用后放弃


def test_extract_citations_reused():
    assert _extract_citations("see [2]", 5) == [2]


def test_verify_pass_returns_first_candidate():
    llm = FakeLLM(
        [
            _msg(tool_calls=[{"name": "search", "args": {"query": "linear"}, "id": "c1"}]),
            _msg(content="用 [1]。"),
        ]
    )
    svc = FakeService({"linear": [_hit("linear", "cid-1", "sec1")]})
    judge = FakeJudge(["[1] supported: 资料支持该论断"])
    r = _agent(llm, svc, judge).ask("q")
    assert r.answer == "用 [1]。"
    assert judge.calls == 1  # 只验证一次，通过即返回


def test_verify_fail_then_agent_corrects():
    llm = FakeLLM(
        [
            _msg(tool_calls=[{"name": "search", "args": {"query": "linear"}, "id": "c1"}]),
            _msg(content="错版答案 [1]。"),
            _msg(content="修正后答案 [1]。"),
        ]
    )
    svc = FakeService({"linear": [_hit("linear", "cid-1", "sec1")]})
    judge = FakeJudge(["[1] unsupported: 资料没提到该说法", "[1] supported: ok"])
    r = _agent(llm, svc, judge).ask("q")
    assert r.answer == "修正后答案 [1]。"
    assert judge.calls == 2  # 第一版被驳回，第二版通过


def test_verify_exhausted_returns_last_candidate():
    llm = FakeLLM(
        [
            _msg(tool_calls=[{"name": "search", "args": {"query": "linear"}, "id": "c1"}]),
            _msg(content="始终不改 [1]。"),
        ]
    )
    svc = FakeService({"linear": [_hit("linear", "cid-1", "sec1")]})
    judge = FakeJudge(["[1] unsupported: 每次都不支持"])
    a = _agent(llm, svc, judge)
    a._max_verify = 1
    r = a.ask("q")
    assert r.answer == "始终不改 [1]。"
    assert r.refused is False  # 验证轮耗尽，返回最后一版而非放弃
    assert judge.calls == 2


def test_search_section_tool_precise_retrieval():
    llm = FakeLLM(
        [
            _msg(tool_calls=[{"name": "search_section", "args": {"section": "torch.nn.Linear"}, "id": "c1"}]),
            _msg(content="用 [1]。"),
        ]
    )
    svc = FakeService(
        {},
        sections={"torch.nn.Linear": [_hit("sec", "cid-linear", "class torch.nn. Linear")]},
    )
    r = _agent(llm, svc).ask("What does Linear do?")
    assert r.refused is False
    assert r.answer == "用 [1]。"
    assert len(r.sources) == 1
    assert r.sources[0].section == "class torch.nn. Linear"
    assert r.trace[0]["action"] == "search_section"
