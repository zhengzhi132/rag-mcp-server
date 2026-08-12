"""分块：把 Document 切成语义完整的检索单元 Chunk。

为什么先定义 Chunk 再写切分逻辑？
因为 embedding / 双向量库 / 检索全部消费 Chunk，稳定的输出结构是
所有下游模块的契约。

两级策略：
1. 按 API entry 切 —— 每个 entry（签名+描述）是语义完整单元，直接成块。
   不按 tokenizer 硬切，保证代码签名不被拦腰截断。
2. 超长 entry 内部递归切 —— 超过 chunk_size 的部分按分隔符优先级
   (\n\n → \n → 句号/逗号/分号 → 空格 → 硬切) 递归切，
   块间带 overlap 保留切点上下文。硬切是最后兜底，尽量不让单词被切断。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ragmcp.ingestion.loader import Document


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # 文档相对名，如 torch.nn.Linear.html
    section: str  # API 名，如 torch.nn.Linear
    index: int  # 全局递增序号（跨文档唯一）
    chunk_id: str  # md5(source|index)，跨进程可复现

    @property
    def title(self) -> str:
        return f"{self.source} :: {self.section}"


# 分隔符优先级：段落 → 行 → 句号 → 逗号 → 分号 → 空格 → 硬切
# 级别越靠后切得越碎；硬切只作最后兜底，尽量不让单词被拦腰切断。
DEFAULT_SEPS: tuple[str, ...] = ("\n\n", "\n", ".", ",", ";", " ")


def _stable_id(source: str, index: int) -> str:
    # 内置 hash(str) 每进程随机（PYTHONHASHSEED），跨进程不一致；
    # FAISS 与 Chroma 对齐靠 chunk_id，必须可复现，所以用 md5。
    return hashlib.md5(f"{source}|{index}".encode("utf-8")).hexdigest()[:16]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """相邻块共享文本：每块开头带上前一块尾部的 overlap 字符。

    切点若恰好落在句子中间，前后两块都能看到被截断的上文，
    避免"跨块上下文丢失"。
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        result.append(chunks[i - 1][-overlap:] + chunks[i])
    return result


def _split_no_overlap(
    text: str,
    chunk_size: int,
    seps: tuple[str, ...],
) -> list[str]:
    """纯递归切分，不带 overlap：返回每块长度 <= chunk_size。

    递归层不处理 overlap，避免内层块回到外层被二次合并 + 二次 overlap
    导致长度失控。overlap 由 _split_long 在最外层统一加一次。
    """
    if len(text) <= chunk_size:
        return [text] if text else []
    if not seps:
        # 没有任何可用分隔符：按固定步长硬切
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    sep = seps[0]
    parts = text.split(sep)
    if len(parts) <= 1:
        # 当前分隔符不出现，降级用下一个
        return _split_no_overlap(text, chunk_size, seps[1:])
    merged: list[str] = []
    buf = ""
    for i, p in enumerate(parts):
        if len(p) > chunk_size:
            # 超长 part：递归切分得到完整块，直接入结果，不参与外层合并
            if buf:
                merged.append(buf)
                buf = ""
            sub = _split_no_overlap(p, chunk_size, seps[1:])
            for j, b in enumerate(sub):
                prefix = sep if (i > 0 and j == 0) else ""
                merged.append(prefix + b)
        else:
            piece = (sep if i > 0 else "") + p
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf:
                    merged.append(buf)
                buf = piece
    if buf:
        merged.append(buf)
    return merged


def _split_long(
    text: str,
    chunk_size: int,
    overlap: int,
    seps: tuple[str, ...],
) -> list[str]:
    """超长文本切分：先递归切到 <= chunk_size，再统一加一次 overlap。"""
    blocks = _split_no_overlap(text, chunk_size, seps)
    return _apply_overlap(blocks, overlap)


def chunk_document(
    doc: Document,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[Chunk]:
    """把一个 Document 切成 Chunk 列表。

    section 取 entry 签名的 API 名部分（如 torch.nn.Linear），
    供检索结果展示与 Chroma 的 where 过滤使用。
    """
    chunks: list[Chunk] = []
    seq = 0
    for entry in doc.entries:
        section = entry.signature.split("(")[0].strip() or entry.signature[:40]
        text = entry.text
        if len(text) <= chunk_size:
            chunks.append(
                Chunk(
                    text=text,
                    source=doc.source,
                    section=section,
                    index=seq,
                    chunk_id=_stable_id(doc.source, seq),
                )
            )
            seq += 1
        else:
            parts = _split_long(text, chunk_size, overlap, DEFAULT_SEPS)
            for part in parts:
                chunks.append(
                    Chunk(
                        text=part,
                        source=doc.source,
                        section=section,
                        index=seq,
                        chunk_id=_stable_id(doc.source, seq),
                    )
                )
                seq += 1
    return chunks
