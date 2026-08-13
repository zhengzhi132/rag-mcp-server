# rag-mcp-server

基于 PyTorch 官方文档的 **RAG 知识检索 MCP Server**：双向量库（FAISS + Chroma）混合检索，DeepSeek 生成带引用的回答，能力通过 **FastMCP** 暴露为标准工具，可被 Claude 等 MCP 客户端直接调用。

## 功能特性

- **双向量库**：FAISS（IndexFlatIP 精确余弦，主检索）+ Chroma（持久化 / where 过滤 / 备份对照），统一 `md5(chunk_id)` 对齐，构建后自动校验双库 top-5 重合率 ≥ 90%
- **章节感知分块**：按 API entry 语义切分，超长块递归切分 + overlap，保证代码签名不被拦腰截断
- **混合检索**：向量（BGE）+ 关键词（BM25）双路，RRF(k=60) 融合，规避两路分数量纲不可比
- **带引用回答**：DeepSeek 基于检索上下文生成，强制 `[n]` 引用 + Sources，越界引用后校验剔除；资料不足正确降级拒绝，不编造
- **MCP Server**：`search` / `ask` / `list_topics` / `stats` 四个标准工具，lifespan 只加载一次模型
- **ReAct 检索 Agent**：模型自行决定检索策略，跨多主题迭代搜索（search / list_topics 工具调用），全局来源表保证多次检索的 `[n]` 引用一致

## 架构

```text
PyTorch 官方文档 (HTML)
        │  loader.py  解析 <dl> 签名+描述
        ▼
      Document ──► chunker.py  章节感知分块 (447 chunks / 96 API topics)
        │  embedder.py  BGE 文档编码 (无 instruction, 384 维)
        ▼
┌────────────────────────── 建库 ──────────────────────────┐
│   FAISS (IndexFlatIP) 主检索       Chroma 持久化/过滤      │
│   同批 chunk，md5 chunk_id 对齐，双库重合率 ≥90% 校验      │
└───────────────────────────────────────────────────────────┘
        ▲
        │  embed_query (查询加检索前缀)
        │
问题 ──► HybridRetriever = 向量 top-k + BM25 top-k ──► RRF 融合 ──► top-k Chunk
        │
        ▼
 Generator.generate: 编号[1]..[n] 组装 context ──► DeepSeek ──► 回答 + [n] 引用
        │                        ▲
        │               引用校验 _extract_citations (越界/非数字剔除)
        ▼
 FastMCP tools: search / ask / list_topics / stats   (lifespan 单次加载)
        │
        ▼
 ReAct Agent (demo --agent): 工具调用迭代检索 → 全局来源表 → 带引用回答
```

## 技术栈

Python 3.13 · FAISS · Chroma · sentence-transformers (BGE) · rank_bm25 · LangChain · DeepSeek · FastMCP

## 快速开始

### 1. 环境

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

BGE 模型需从 HuggingFace 下载，国内可设镜像（config.py 已默认写入 `HF_ENDPOINT=https://hf-mirror.com`）。

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：

```bash
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
```

> 也可直接用环境变量 `DEEPSEEK_API_KEY`，不需要 `.env` 文件。

### 3. 构建索引

```bash
PYTHONPATH=src python scripts/build_index.py
```

输出示例：`n_chunks=447, dim=384, dual_store_overlap=0.93+`。

> 索引数据在 `data/`（已被 .gitignore 排除，可随时重建，幂等）。

### 4. 命令行问答

```bash
# 单次提问
PYTHONPATH=src python -m ragmcp.cli.demo "How to create a Linear layer in PyTorch?"

# 交互式（输入 exit 退出）
PYTHONPATH=src python -m ragmcp.cli.demo
```

### 4.1 ReAct 检索 Agent

```bash
# 模型自行决定检索策略，跨多主题迭代搜索
PYTHONPATH=src python -m ragmcp.cli.demo --agent "How to train a model with Adam on a DataLoader using CrossEntropyLoss?"
```

### 5. 启动 MCP Server

```bash
PYTHONPATH=src python -m ragmcp.server.mcp_server    # stdio transport
```

注册进 Claude Code / Cursor 等客户端后，即可通过标准工具调用：

| 工具 | 说明 |
|---|---|
| `search` | 混合检索（向量 + BM25），返回 top-k 来源片段与分数 |
| `ask` | 端到端问答，返回带 `[n]` 引用的回答 + Sources |
| `list_topics` | 知识库覆盖的 API 主题列表 |
| `stats` | 知识库统计（分块数 / 文档数 / 主题数 / 维度） |

客户端验收脚本：

```bash
PYTHONPATH=src python scripts/test_mcp_client.py
```

### 6. 测试

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## 目录结构

```text
rag-mcp-server/
├── scripts/
│   ├── download_docs.py    # 下载 PyTorch 文档页
│   ├── build_index.py      # 全量构建双向量库（幂等）
│   └── test_mcp_client.py  # MCP stdio 客户端验收
├── src/ragmcp/
│   ├── config.py           # pydantic-settings 配置
│   ├── ingestion/          # loader(HTML/PDF) chunker(章节感知) embedder(BGE)
│   ├── storage/            # faiss_store chroma_store indexer(双写+对齐校验)
│   ├── retrieval/          # keyword(BM25) hybrid(RRF+加权融合)
│   ├── generation/         # generator(DeepSeek + [n]引用 + 降级)
│   ├── agent/              # rag_agent(ReAct 迭代检索, 工具调用)
│   ├── service/            # rag_service(编排 search/ask/list_topics/stats)
│   ├── server/             # mcp_server(FastMCP 4 工具) lifespan(单次加载)
│   └── cli/                # demo(命令行问答)
├── tests/                  # chunker / keyword / hybrid / generator
├── data/                   # gitignore：raw / chroma / faiss
├── requirements.txt
└── .env.example
```

## 关键设计

1. **双向量库分工**：FAISS 快、精确余弦、无持久化；Chroma 落盘、where 过滤、备份对照。统一 `md5(source|index)` 的 chunk_id 对齐，构建后双库 top-5 重合率校验，证明双库结果一致。
2. **BGE 检索姿势**：文档编码不加 instruction、查询编码加前缀 `"Represent this sentence for searching relevant passages: "`，配合 `normalize_embeddings=True` 使 IndexFlatIP 内积 = 余弦。
3. **RRF 融合**：向量分（-1~1）与 BM25 分（0~几十）量纲不可比，直接加权无意义；RRF 只看排名（k=60，Cormack 2009），跨打分器鲁棒。
4. **引用后校验**：LLM 会幻觉出 context 里不存在的编号，`_extract_citations` 只保留 `1<=n<=total` 的合法引用，Sources 才可信。
5. **无答案降级**：`_low_confidence` 阈值（0.02，实测校准）+ SYSTEM_PROMPT 规则 3 双保险，资料不足明确拒绝，不编造。
6. **lifespan 单次加载**：BGE 模型（~130MB）+ FAISS 索引在服务启动时加载一次，所有工具调用复用，stdio 会话不重载。
7. **ReAct 全局来源表**：Agent 多次检索的 chunk 去重后进全局来源表，每次 search 返回的片段用全局绝对编号 [n]，最终回答引用的 [n] 精确反查——跨主题多轮检索的引用依然可信。
