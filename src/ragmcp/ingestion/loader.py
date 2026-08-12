"""文档加载：HTML / PDF → 统一 Document 结构。

核心思路：PyTorch 的 generated/ 页面用 <dl class="function|class"> 把每个 API
排成「<dt>签名</dt> + <dd>描述</dd>」的规整结构。按 API entry 提取，得到
语义完整的"签名 + 说明"单元，是后续按章节分块的天然边界。

HTML 是主路径（结构化好、质量高）；PDF 是备选（应对自供文档），
用 pymupdf 逐页抽取纯文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass
class ApiEntry:
    """一个 API 的签名 + 描述。"""

    signature: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.signature}\n{self.body}"


@dataclass
class Document:
    """一个文档源文件的解析结果。"""

    source: str  # 相对文件名，如 torch.nn.Linear.html
    title: str  # API 名，如 torch.nn.Linear
    entries: list[ApiEntry] = field(default_factory=list)
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            self.text = "\n\n".join(e.text for e in self.entries)


def _clean(text: str) -> str:
    """合并多余空白：行内空白压成一个空格，空行归一化。"""
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def parse_html(path: Path) -> Document:
    """解析 PyTorch generated/ 页面：提取每个 dl 的签名与描述。"""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    # 标题：优先 <h1>，兜底从 section id 取
    h1 = soup.find("h1")
    title = _clean(h1.get_text(" ")) if h1 else path.stem

    entries: list[ApiEntry] = []
    for dl in soup.find_all("dl"):
        classes = dl.get("class") or []
        if not any(c in ("function", "class", "method") for c in classes):
            continue
        dt = dl.find("dt", recursive=False)
        dd = dl.find("dd", recursive=False)
        if dt is None:
            continue
        sig = _clean(dt.get_text(" "))
        body = _clean(dd.get_text("\n")) if dd else ""
        if sig:
            entries.append(ApiEntry(signature=sig, body=body))

    return Document(source=path.name, title=title, entries=entries)


def parse_pdf(path: Path) -> Document:
    """PDF 备选路径：pymupdf 逐页抽取纯文本（无结构化 entries）。"""
    import fitz  # pymupdf

    with fitz.open(path) as pdf:
        pages = [page.get_text("text") for page in pdf]
    text = _clean("\n".join(pages))
    return Document(source=path.name, title=path.stem, text=text)


def load(path: Path) -> Document:
    """统一入口：按扩展名分派解析器。"""
    if path.suffix.lower() == ".html":
        return parse_html(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"不支持的文档类型: {path.suffix}（仅支持 .html / .pdf）")
