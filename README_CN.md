<div align="center">
  <p align="right">
    <a href="README.md"> 🌎English </a> | <span> 🇨🇳中文 </span>
  </p>
</div>

# IR Papers Retrieval System

信息检索论文检索系统，支持混合搜索（BM25 + 密集嵌入）和重排序。使用 Python 构建，利用 Sentence Transformers、BM25 和 API 集成。

### 功能
- **混合搜索**：结合稀疏（BM25）和密集（嵌入）检索，提高准确性。
- **重排序**：使用 CrossEncoder 或基于 API 的重排序器优化结果。
- **API 支持**：与外部 API 集成，用于嵌入和重排序（例如 DMXAPI）。
- **可配置**：基于 YAML 的配置，便于设置。
- **交互式 CLI**：命令行界面用于查询。

### 安装
1. 克隆仓库：
   ```bash
   git clone https://github.com/blowingwind05/agentic-rag-retrieval-part.git
   cd agentic-rag-retrieval-part
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   （如果没有 requirements.txt，手动安装：pandas, numpy, torch, sentence-transformers, bm25s, pyyaml, requests, openai）

3. 确保安装 PyYAML：
   ```bash
   pip install PyYAML
   ```

### 配置
编辑 `config.yaml` 设置参数：
- `data_path`：数据目录路径（例如 "./"）。
- `json_file`：包含论文数据的 JSON 文件（例如 "ir_papers.json"）。
- 嵌入/重排序的 API 密钥和 URL。
- `top_k_retrieval`、`top_k_final`、`batch_size` 等。

### 使用
1. 准备数据：将 `ir_papers.json` 放在数据路径中。
2. 运行系统：
   ```bash
   python ir_papers_retrieval.py
   ```
3. 交互式输入查询。输入 'quit' 退出。

### 依赖
- Python 3.8+
- pandas, numpy, torch
- sentence-transformers, bm25s
- PyYAML, requests, openai

### 许可证
MIT 许可证。