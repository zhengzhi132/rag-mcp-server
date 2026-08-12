"""FastMCP 生命周期：服务启动时初始化一次 RagService，所有工具共享。

为什么需要 lifespan？MCP stdio 每次会话起一个新进程；如果不做共享，
BGE 模型（~130MB）和 FAISS 索引会在每次工具调用时重载。lifespan 保证
整个服务生命周期内只加载一次，工具通过 ctx.request_context.lifespan_context
拿到同一实例。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from ragmcp.service.rag_service import RagService


@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[RagService]:
    """启动时预加载一次服务，yield 给所有工具复用。"""
    service = RagService()
    service.ensure_loaded()  # 预热：加载索引 + BGE，避免第一次工具调用卡顿
    yield service
