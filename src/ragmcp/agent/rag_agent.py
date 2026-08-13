"""ReAct 检索 Agent：迭代检索 + 判断，直到能回答。

一次「检索 → 生成」的静态 RAG 有上限：检索词没问对、答案跨多个主题，
一次搜索就找偏。ReAct 让模型自己决定「要不要再搜、搜什么」，工具只有
search / list_topics 两个，天然贴合 MCP 已暴露的能力。

关键设计：
- 工具调用：DeepSeek 走 OpenAI 兼容协议，bind_tools 原生支持函数调用。
- 编号一致性：多次 search 的 chunk 去重后进全局来源表 self._sources，
  每次 search 返回的结果用全局绝对编号 [n]，最终回答引用的 [n] 能精确
  反查到真实 chunk（来源表索引 n-1）。
- 停止条件：模型不调用工具 = 给出最终答案；或超过 max_iters 强制拒绝。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from ragmcp.config import settings
from ragmcp.generation.generator import REFUSAL_MESSAGE, Source, _extract_citations
from ragmcp.service.rag_service import RagService

# ============ 用户核心：Agent 系统提示词 ============
# 决定模型何时搜索、搜几次、何时停、怎么引用。可自行调优。
AGENT_SYSTEM_PROMPT = (
    "你是 RAG 检索 Agent，负责基于 PyTorch 官方文档知识库回答问题。\n"
    "你有两个工具：\n"
    "- search(query)：检索知识库，返回相关资料片段，每个片段带全局 [n] 编号\n"
    "- list_topics()：列出知识库覆盖的 API 主题\n"
    "规则：\n"
    "1. 回答问题前先用 search 检索；不确定就多搜几次（换关键词 / 拆主题），直到证据足够。\n"
    "2. 最终回答必须基于检索到的资料，用 [n] 引用（编号来自 search 返回结果），不要编造。\n"
    "3. 答案跨多个主题时，分多次检索，别只搜一次就答。\n"
    "4. 搜了几次仍无法回答，直接说「根据已有资料无法回答该问题」。\n"
    "5. 证据够了就停止检索，直接给出最终回答，不要多余搜索。\n"
    "6. 使用与问题相同的语言回答。"
)

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "检索 PyTorch 文档知识库，返回带 [n] 编号的资料片段",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索词"}},
            "required": ["query"],
        },
    },
}

_LIST_TOPICS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_topics",
        "description": "列出知识库覆盖的全部 API 主题",
        "parameters": {"type": "object", "properties": {}},
    },
}

# ============ 用户核心：引用验证提示词 ============
# 自纠正循环里的"裁判"：逐条判断 [n] 引用是否真的支持回答中的论断。
VERIFY_PROMPT = (
    "你是引用验证器。判断下面回答中每个 [n] 引用对应的资料，是否真正支持"
    "回答里的相关论断。\n"
    "规则：资料明确提到才算 supported；资料没提到、含糊或与论断矛盾，标 unsupported。\n\n"
    "回答：\n{answer}\n\n资料：\n{blocks}\n\n"
    "对每个被引用的编号输出一行，格式：\n[n] supported|unsupported: <一句话理由>"
)


def _parse_unsupported(verdict: str) -> list[int]:
    """从验证器输出里抓所有 [n] unsupported 的编号。"""
    return [int(n) for n in re.findall(r"\[(\d+)\]\s*unsupported", verdict)]


def _line_is_unsupported(line: str) -> bool:
    return re.search(r"\[\d+\]\s*unsupported", line) is not None


@dataclass(frozen=True)
class AgentResult:
    """Agent 一轮推理的完整输出。refused=True 表示放弃回答。"""

    answer: str
    sources: list[Source]
    refused: bool
    trace: list[dict] = field(default_factory=list)  # 每轮 action / observation


class RagAgent:
    def __init__(self, k: int = 5, max_iters: int = 4, max_verify: int = 2) -> None:
        self._service = RagService()
        self._llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.2,
            timeout=60,
            max_retries=2,
        ).bind_tools([_SEARCH_TOOL, _LIST_TOPICS_TOOL])
        # 引用验证器：独立 LLM，temperature 0，只看"引用是否支持论断"
        self._judge = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.0,
            timeout=60,
            max_retries=2,
        )
        self._k = k
        self._max_iters = max_iters
        self._max_verify = max_verify
        self._sources: list[Source] = []
        self._seen: set[str] = set()

    def ask(self, question: str) -> AgentResult:
        """ReAct 检索 + 自纠正验证循环。

        先跑 ReAct 循环得到候选答案；再用独立验证器逐条检查每个 [n] 引用
        是否真的支持论断；不支持的引用反馈回 Agent 让它重搜/改写，最多
        max_verify 轮。
        """
        self._sources = []
        self._seen = set()
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[dict] = []
        last: AgentResult | None = None

        for _ in range(self._max_verify + 1):
            candidate = self._react_round(messages, trace)
            if candidate.refused:
                return candidate
            last = candidate
            feedback = self._verify(candidate)
            if feedback is None:
                return candidate  # 引用全部通过验证
            messages.append({"role": "user", "content": feedback})

        return last  # 验证轮耗尽：返回最后一版候选，不强制放弃

    def _react_round(self, messages: list[dict], trace: list[dict]) -> AgentResult:
        """ReAct 工具调用循环：直到模型给出最终答案或耗尽 max_iters。"""
        for _ in range(self._max_iters):
            ai = self._llm.invoke(messages)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                # 模型不再调用工具 = 最终答案
                answer = (ai.content or "").strip()
                citations = _extract_citations(answer, len(self._sources))
                sources = [self._sources[n - 1] for n in citations]
                return AgentResult(answer=answer, sources=sources, refused=False, trace=list(trace))
            messages.append(ai)
            for tc in tool_calls:
                args = tc.get("args") or {}
                obs = self._run_tool(tc["name"], args)
                trace.append({"action": tc["name"], "args": args, "observation": obs[:200]})
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": obs}
                )

        # 超过 max_iters 还没给出最终答案 → 拒绝
        return AgentResult(answer=REFUSAL_MESSAGE, sources=[], refused=True, trace=list(trace))

    def _verify(self, result: AgentResult) -> str | None:
        """验证器逐条检查引用支持性；全部通过返回 None，否则返回反馈文本。"""
        if not result.sources:
            return None
        blocks = []
        for s in result.sources:
            text = self._service.chunk_text(s.chunk_id) or "(未取到文本)"
            blocks.append(f"[{s.n}] ({s.source} :: {s.section})\n{text}")
        verdict = self._judge.invoke(
            [
                {
                    "role": "user",
                    "content": VERIFY_PROMPT.format(
                        answer=result.answer, blocks="\n\n".join(blocks)
                    ),
                }
            ]
        ).content.strip()
        unsupported = _parse_unsupported(verdict)
        if not unsupported:
            return None
        reasons = "\n".join(
            line for line in verdict.splitlines() if _line_is_unsupported(line)
        )
        return (
            "【验证反馈】以下引用未通过支持性验证："
            + ", ".join(f"[{n}]" for n in unsupported)
            + "\n"
            + reasons
            + "\n请重新检索或修正回答，确保每个引用都真正支持对应论断。"
        )

    def _run_tool(self, name: str, args: dict) -> str:
        if name == "search":
            return self._search(args.get("query", ""))
        if name == "list_topics":
            return "\n".join(self._service.list_topics())
        return f"未知工具: {name}"

    def _search(self, query: str) -> str:
        """检索并把结果登记进全局来源表，返回带全局 [n] 编号的文本。"""
        lines: list[str] = []
        for r in self._service.search(query, k=self._k):
            cid = r["chunk_id"]
            if cid not in self._seen:
                self._seen.add(cid)
                self._sources.append(
                    Source(
                        n=len(self._sources) + 1,
                        chunk_id=cid,
                        source=r["source"],
                        section=r["section"],
                        score=r["score"],
                    )
                )
            n = next(
                i for i, s in enumerate(self._sources, 1) if s.chunk_id == cid
            )
            lines.append(f"[{n}] ({r['source']} :: {r['section']})\n{r['snippet']}")
        if not lines:
            return "检索无结果，请换关键词或考虑该问题不在知识库范围内。"
        return "\n\n".join(lines)
