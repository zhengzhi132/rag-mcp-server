"""下载 PyTorch 官方 API 文档页面到 data/raw/html/。

只下载 pytorch.org/docs/stable/generated/ 下的单页文档——
每个页面是一个完整 API 的签名+描述，结构规整，适合做 RAG 知识库。

用法：
    python scripts/download_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ragmcp.config import settings

# (文件名, URL) —— 覆盖核心类 / 激活 / 优化器 / 函数 / Tensor方法 / 数据加载
DOC_URLS: list[tuple[str, str]] = [
    ("torch.nn.Module", "https://pytorch.org/docs/2.13/generated/torch.nn.Module.html"),
    ("torch.nn.Linear", "https://pytorch.org/docs/2.13/generated/torch.nn.Linear.html"),
    ("torch.nn.Conv2d", "https://pytorch.org/docs/2.13/generated/torch.nn.Conv2d.html"),
    ("torch.nn.CrossEntropyLoss", "https://pytorch.org/docs/2.13/generated/torch.nn.CrossEntropyLoss.html"),
    ("torch.nn.Embedding", "https://pytorch.org/docs/2.13/generated/torch.nn.Embedding.html"),
    ("torch.nn.ReLU", "https://pytorch.org/docs/2.13/generated/torch.nn.ReLU.html"),
    ("torch.nn.functional.relu", "https://pytorch.org/docs/2.13/generated/torch.nn.functional.relu.html"),
    ("torch.optim.SGD", "https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html"),
    ("torch.optim.Adam", "https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html"),
    ("torch.utils.data.DataLoader", "https://pytorch.org/docs/2.13/data.html"),
    ("torch.Tensor.view", "https://pytorch.org/docs/2.13/generated/torch.Tensor.view.html"),
    ("torch.Tensor.numpy", "https://pytorch.org/docs/2.13/generated/torch.Tensor.numpy.html"),
    ("torch.sum", "https://pytorch.org/docs/2.13/generated/torch.sum.html"),
    ("torch.matmul", "https://pytorch.org/docs/2.13/generated/torch.matmul.html"),
    ("torch.randn", "https://pytorch.org/docs/2.13/generated/torch.randn.html"),
]


def download(url: str, out_path: Path, timeout: int = 30) -> None:
    """下载单个页面，带 User-Agent（部分站点会拦默认 UA）。"""
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def main() -> None:
    html_dir = settings.raw_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for name, url in DOC_URLS:
        out = html_dir / f"{name}.html"
        if out.exists() and out.stat().st_size > 0:
            ok += 1
            print(f"SKIP {name} (已存在)")
            continue
        try:
            download(url, out)
            ok += 1
            print(f"OK   {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, str(exc)))
            print(f"ERR  {name}: {exc}")

    print(f"\n完成：{ok} 成功，{len(failed)} 失败")
    for name, err in failed:
        print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
