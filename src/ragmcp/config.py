"""项目配置：从 .env 或环境变量读取。

所有模块通过 `from ragmcp.config import settings` 获取配置。
字段名与 .env 的大写键自动对应（如 deepseek_api_key <-> DEEPSEEK_API_KEY）。
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 固定 HF 镜像：本机直连 huggingface.co 超时，BGE 模型需走镜像下载。
# setdefault 保证用户显式设置的环境变量优先（不会覆盖）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek 生成模型
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # 数据与索引目录
    data_dir: Path = Path("./data")

    # 检索参数（P7 评估后可能调整，作为默认值集中在这里）
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k_default: int = 5
    rrf_k: int = 60
    hybrid_weight: float = 0.7

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
