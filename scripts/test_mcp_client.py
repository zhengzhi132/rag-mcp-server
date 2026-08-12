"""MCP stdio 客户端验收：拉起 server，验证 4 个工具 + ask 的引用。

运行：
    cd rag-mcp-server
    PYTHONPATH=src .venv/Scripts/python scripts/test_mcp_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _text(res) -> str:
    """从 call_tool 结果里取第一个文本块。"""
    return res.content[0].text


async def _run() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ragmcp.server.mcp_server"],
        env=_env(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. list_tools = 4
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("tools:", names)
            assert names == ["search", "ask", "list_topics", "stats"], names

            # 2. ask：断言回答 + 引用 sources
            res = await session.call_tool(
                "ask", {"question": "How to create a Linear layer in PyTorch?"}
            )
            payload = json.loads(_text(res))
            print("ask keys:", sorted(payload.keys()))
            assert payload["answer"], "回答不能为空"
            assert isinstance(payload["sources"], list), "sources 必须是列表"
            print(f"ask refused={payload['refused']} latency_ms={payload['latency_ms']}")
            print(f"  sources: {[(s['n'], s['source']) for s in payload['sources']]}")
            assert len(payload["sources"]) > 0, "应有至少一个引用来源"

            # 3. 其余工具冒烟
            sres = await session.call_tool("search", {"query": "CrossEntropyLoss", "top_k": 3})
            search_out = json.loads(_text(sres))
            assert search_out["results"], "search 应返回非空"
            print(f"search top1: {search_out['results'][0]['section']} "
                  f"score={search_out['results'][0]['score']} latency={search_out['latency_ms']}ms")

            tres = await session.call_tool("list_topics", {})
            topics = json.loads(_text(tres))["topics"]
            assert len(topics) > 0, "topics 非空"
            print(f"list_topics: {len(topics)} topics")

            st = await session.call_tool("stats", {})
            print("stats:", json.loads(_text(st)))

    print("\nP6 acceptance passed")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
