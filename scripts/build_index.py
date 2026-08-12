"""全量构建双向量库：python scripts/build_index.py

读 data/raw/ 下全部文档，分块 → embedding → 双写 FAISS+Chroma，
打印统计（n_chunks、dim、双库重合率）。幂等：重建前清空双库。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragmcp.config import settings  # noqa: E402
from ragmcp.storage.indexer import build_index  # noqa: E402


def main() -> None:
    stats = build_index(settings.raw_dir, settings.faiss_dir, settings.chroma_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
