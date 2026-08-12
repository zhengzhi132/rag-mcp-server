"""DeepSeek 生成：基于检索上下文生成带 [n] 引用的回答。

RAG 的最后一步：把检索到的 top-k chunks 编号成 [1]..[n]，连同问题一起交给
LLM，要求它只能基于 context 回答、用 [n] 引用。LLM 返回的引用编号不可信
（可能幻觉出越界的 [9]），所以做后校验：只保留范围内的引用，丢弃其余。

无答案正确降级：检索置信度过低或资料覆盖不了问题时，明确拒绝而不是编造。

模块里标了「用户核心」的地方是待补实现（骨架只留接口与注释指引），
其余为机械部分（编号、组装、调 LLM、组装结果）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from ragmcp.config import settings
from ragmcp.ingestion.chunker import Chunk

# ============ 用户核心 1/3：规则提示词 ============
# 要求至少覆盖：
#   1. 只能基于 [context] 中给的内容回答，不得使用外部知识
#   2. 回答引用资料时用 [n] 标注，n 必须是资料里实际存在的编号（1..总数）
#   3. 资料不足以回答时，明确说"根据已有资料无法回答"，不要编造
SYSTEM_PROMPT = (
    "你是 PyTorch 官方文档问答助手。\n"
    "1. 只能基于 [context] 中提供的内容回答，禁止使用外部知识或猜测。\n"
    "2. 引用资料时用 [n] 标注，n 必须对应 [context] 里实际存在的编号（1 到 N）。\n"
    "3. [context] 不足以回答或与问题无关时，直接回答「根据已有资料无法回答该问题」，不要编造。\n"
    "4. 必须使用与问题相同的语言回答（问题为英文则用英文，中文则用中文）。"
)

# 无答案降级时返回的固定拒绝文案（措辞可调）
REFUSAL_MESSAGE = "根据知识库中的现有资料，无法回答该问题。"


@dataclass(frozen=True)
class Source:
    """一个被引用的来源，对应回答里的 [n]。"""

    n: int
    chunk_id: str
    source: str
    section: str
    score: float


@dataclass(frozen=True)
class GenerationResult:
    """一次生成的完整输出。refused=True 表示降级拒绝，此时 sources 为空。"""

    answer: str
    sources: list[Source]
    refused: bool


class Generator:
    def __init__(self) -> None:
        # DeepSeek 走 OpenAI 兼容协议；key 从环境变量来，不进 git
        self._llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.3,
            timeout=60,
            max_retries=2,
        )

    def generate(
        self,
        question: str,
        hits: list[tuple[float, Chunk]],
    ) -> GenerationResult:
        """基于检索命中的 top-k chunks 生成回答。hits 按分数降序。

        流程：降级守卫 → 编号组装 context → 调 LLM → 引用校验 → 组装结果。
        """
        # 用户核心 2/3 决定何时该拒绝（检索置信度过低）
        if _low_confidence(hits):
            return GenerationResult(answer=REFUSAL_MESSAGE, sources=[], refused=True)

        # 编号 + 组装 context：每块带来源，方便 LLM 引用
        context = "\n\n".join(
            f"[{i}] ({c.source} :: {c.section})\n{c.text}"
            for i, (_, c) in enumerate(hits, 1)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"[context]\n{context}\n\n[question]\n{question}",
            },
        ]
        answer = self._llm.invoke(messages).content.strip()

        # 用户核心 3/3 决定引用解析规则；这里把合法引用映射回真实 Chunk
        citations = _extract_citations(answer, len(hits))
        sources = [
            Source(
                n=n,
                chunk_id=hits[n - 1][1].chunk_id,
                source=hits[n - 1][1].source,
                section=hits[n - 1][1].section,
                score=hits[n - 1][0],
            )
            for n in citations
        ]
        return GenerationResult(answer=answer, sources=sources, refused=False)


# ============ 用户核心 2/3：无答案降级守卫 ============
# hits: [(score, Chunk)]，score 是 RRF 融合分（池内相对，通常 < 0.05）。
#
# 两种降级时机：
# 1. 检索侧：top-1 融合分低于经验阈值 → 双路都没召回 → 直接拒绝，省一次 LLM 调用。
#    ⚠ RRF 分是池内相对的，阈值要对着真实语料校准，别拍脑袋。
# 2. 生成侧：LLM 自己判断 context 覆盖不了 → 回答出现拒绝话术，可在 generate
#    末尾检测关键词（"无法回答"/"cannot"）置 refused=True。
# 这里先实现 1；2 作为课后思考。
# 检索置信度阈值：top-1 RRF 融合分低于它 → 直接拒绝。
# 校准说明见 _low_confidence。
REFUSAL_SCORE_THRESHOLD = 0.02


def _low_confidence(hits: list[tuple[float, Chunk]]) -> bool:
    """检索置信度过低 → 直接拒绝，省一次 LLM 调用。

    实测校准（2026-08，pool=30, rrf_k=60）：
    - 好查询 top1 ≈ 0.028~0.033（CrossEntropy 0.033 / view vs reshape 0.032 / Linear 0.028）
    - 无关但话题相近的查询 top1 ≈ 0.028（quantum computing），与好查询
      无法靠融合分区分 —— 这类由 SYSTEM_PROMPT 规则 3 兜底（模型判断）
    - 完全无关查询 top1 ≈ 0.016（make a pizza，仅单路 rank1 单边贡献）
    结论：融合分阈值只适合抓"最极端"漏检，不是主要防线；阈值取 0.02
    卡掉单边弱支持，主要拒绝机制是模型自己判断。
    """
    if not hits:
        return True
    return hits[0][0] < REFUSAL_SCORE_THRESHOLD


# ============ 用户核心 3/3：引用解析与校验 ============
# answer 里可能出现 [1]、[1,2]、[abc]、[9]（越界）。
# 规则：只保留 1<=n<=total 的整数引用，按出现顺序去重返回。
# 提示：真实 LLM 常输出 [1,2] 这种复合括号，解析时用正则 \d+ 取数字即可。
def _extract_citations(answer: str, total: int) -> list[int]:
    """从回答解析 [n] 引用，只保留 1<=n<=total，按出现顺序去重。

    为什么只匹配方括号内的数字、不用裸 \\d+？因为正文里可能出现
    "version 2.1" 这类数字，裸 \\d+ 会把 2、1 也当成引用。用
    \\[([\\d,\\s-]+)\\] 只抓方括号内的内容，再按逗号/空格/连字符拆分，
    天然支持 LLM 常输出的 [1]、[1,2]、[2, 3] 复合引用。
    """
    seen: set[int] = set()
    out: list[int] = []
    for inner in re.findall(r"\[([\d,\s\-]+)\]", answer):
        for part in re.split(r"[,，\-\s]+", inner):
            if part.isdigit():
                n = int(part)
                if 1 <= n <= total and n not in seen:
                    seen.add(n)
                    out.append(n)
    return out
