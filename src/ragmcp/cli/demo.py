"""命令行问答入口：python -m ragmcp.cli.demo "问题"

无参数时进入交互式循环，输入 exit / quit 退出。
"""

from __future__ import annotations

import sys

from ragmcp.service.rag_service import RagService


def _print_answer(result: dict) -> None:
    print("\n" + "─" * 60)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    if result["refused"]:
        return
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(
                f"  [{s['n']}] {s['source']} :: {s['section']}"
                f"  (score={s['score']})"
            )


def _make_ask(use_agent: bool):
    """返回 ask 函数：静态 RAG（默认）或 ReAct Agent（--agent）。"""
    if use_agent:
        from ragmcp.agent.rag_agent import RagAgent

        agent = RagAgent()

        def ask(q: str) -> dict:
            r = agent.ask(q)
            return {
                "question": q,
                "answer": r.answer,
                "refused": r.refused,
                "sources": [
                    {"n": s.n, "source": s.source, "section": s.section, "score": s.score}
                    for s in r.sources
                ],
            }

        return ask
    service = RagService()
    return service.ask


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    use_agent = "--agent" in args
    args = [a for a in args if a != "--agent"]
    ask = _make_ask(use_agent)
    if args:
        _print_answer(ask(args[0]))
        return
    print("rag-mcp-server 问答（--agent 走 ReAct；输入 exit 退出）")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        _print_answer(ask(q))


if __name__ == "__main__":
    main()
