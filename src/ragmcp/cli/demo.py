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


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    service = RagService()
    if args:
        _print_answer(service.ask(args[0]))
        return
    print("rag-mcp-server 问答（输入 exit 退出）")
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
        _print_answer(service.ask(q))


if __name__ == "__main__":
    main()
