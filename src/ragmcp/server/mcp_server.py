"""FastMCP 服务：把 RagService 能力暴露为标准 MCP 工具。

四个工具：
- search       混合检索（向量 + BM25），返回来源片段与分数
- ask          端到端问答，带 [n] 引用与 Sources
- list_topics  知识库覆盖的 API 主题
- stats        知识库统计

工具用同步函数：RagService 内部是阻塞 I/O（embedding + DeepSeek 调用），
FastMCP 会把同步工具丢进线程池执行，不卡事件循环。
"""

from __future__ import annotations

import time
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from ragmcp.server.lifespan import lifespan

mcp = FastMCP(
    "ragmcp-server",
    instructions="PyTorch 文档知识库检索与问答：混合检索、带引用的回答、主题与统计。",
    lifespan=lifespan,
)


@mcp.tool()
def search(
    query: str,
    top_k: Annotated[int, Field(ge=1, le=20, description="返回条数")] = 5,
    ctx: Context = None,
) -> dict:
    """混合检索（向量 + BM25 RRF 融合），返回 top-k 来源片段与分数。

    返回单个 JSON 对象而非列表：FastMCP 会把 list 返回值拆成多个
    content 块，客户端取数别扭；包一层 dict 是更稳的 MCP 惯例。
    """
    svc = ctx.request_context.lifespan_context
    t0 = time.time()
    results = svc.search(query, k=top_k)
    return {
        "results": results,
        "latency_ms": round((time.time() - t0) * 1000),
    }


@mcp.tool()
def ask(
    question: str,
    top_k: Annotated[int, Field(ge=1, le=20, description="返回条数")] = 5,
    ctx: Context = None,
) -> dict:
    """端到端问答：检索 + DeepSeek 生成带 [n] 引用的回答（含 Sources）。"""
    svc = ctx.request_context.lifespan_context
    t0 = time.time()
    result = svc.ask(question, k=top_k)
    result["latency_ms"] = round((time.time() - t0) * 1000)
    return result


@mcp.tool()
def list_topics(ctx: Context = None) -> dict:
    """列出知识库覆盖的 API 主题（去重排序）。"""
    svc = ctx.request_context.lifespan_context
    topics = svc.list_topics()
    return {"topics": topics, "count": len(topics)}


@mcp.tool()
def stats(ctx: Context = None) -> dict:
    """知识库统计：分块数、文档数、主题数、向量维度。"""
    svc = ctx.request_context.lifespan_context
    return svc.stats()


if __name__ == "__main__":
    mcp.run(transport="stdio")
