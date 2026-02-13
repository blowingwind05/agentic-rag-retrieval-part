# IR Papers Retrieval System
# Consolidated from Jupyter Notebook

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import torch
import gc
import bm25s
from pathlib import Path
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
import requests
import urllib3
import yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load configuration from YAML
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Override device based on availability
CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"

class APIEmbeddingModel:
    def __init__(self, model_name, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def encode(self, sentences, batch_size=32, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), disable=not show_progress_bar, desc="API Embeddings"):
            batch = sentences[i : i + batch_size]
            # 注意：OpenAI Embedding 接口有 token 限制，如果 batch 过大或文本过长可能会报错
            max_retries = CONFIG["max_retries"]
            for attempt in range(max_retries):
                try:
                    # 添加超时设置，避免默认超时太短
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name,
                        timeout=60 # 设置60秒超时
                    )
                    embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(embeddings)
                    break  # 成功则跳出重试循环
                except Exception as e:
                    print(f"Embedding API Error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        # 最后一次重试失败，填充零向量作为 fallback
                        print("Max retries reached, using zero vectors as fallback")
                        all_embeddings.extend([[0.0] * 1536 for _ in batch])  # 使用列表而不是numpy数组
                    else:
                        # 等待一秒后重试
                        import time
                        time.sleep(1)

        all_embeddings = np.array(all_embeddings)

        if normalize_embeddings:
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            # 避免除以 0
            norms[norms == 0] = 1e-10
            all_embeddings = all_embeddings / norms

        if convert_to_tensor:
            return torch.tensor(all_embeddings)
        return all_embeddings


class APIReranker:
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.api_key = api_key
        # 构造 rerank url: .../v1/ -> .../v1/rerank
        self.url = base_url.rstrip("/") + "/rerank" if "rerank" not in base_url else base_url
        # 修正可能出现的双 slash 问题
        self.url = self.url.replace("//rerank", "/rerank")

    def predict(self, sentences, batch_size=8, show_progress_bar=False):
        # sentences 是 [ [query, doc], [query, doc], ... ]
        all_scores = []

        # 按 batch_size 切分
        for i in tqdm(range(0, len(sentences), batch_size), disable=not show_progress_bar, desc="API Rerank"):
            batch = sentences[i : i + batch_size]

            if not batch:
                continue

            # 提取 query 和 documents
            current_query = batch[0][0]
            current_documents = [pair[1] for pair in batch]

            # 构造 Payload
            payload = {
                "model": self.model_name,
                "query": current_query,
                "documents": current_documents,
                "top_n": len(batch),
                "return_documents": False
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}" if not self.api_key.startswith("Bearer") else self.api_key,
                "Content-Type": "application/json"
            }

            max_retries = CONFIG["max_retries"]
            for attempt in range(max_retries):
                try:
                    # ！！！添加 verify=False 以解决 SSLEOFError！！！
                    # SSLEOFError 通常是因为本地网络环境与服务器的 SSL 握手异常导致的
                    response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=60, verify=False)
                    response.raise_for_status()
                    result = response.json()

                    # 解析 scores
                    # 期望格式: "results": [ { "index": 0, "relevance_score": 0.97 }, ... ]
                    if "results" in result:
                        batch_scores = [0.0] * len(batch)
                        for item in result["results"]:
                            idx = item["index"]
                            if idx < len(batch_scores):
                                batch_scores[idx] = item["relevance_score"]
                        all_scores.extend(batch_scores)
                    else:
                        print(f"Warning: 'results' key not found. Raw: {str(result)[:100]}")
                        all_scores.extend([0.0] * len(batch))
                    break  # 成功则跳出重试循环
                except Exception as e:
                    print(f"API Rerank Request Failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        # 最后一次重试失败，Fallback: 全0分
                        print("Max retries reached, using zero scores as fallback")
                        all_scores.extend([0.0] * len(batch))
                    else:
                        # 等待一秒后重试
                        import time
                        time.sleep(1)

        return np.array(all_scores)

# Load IR papers data
json_file_path = Path(CONFIG["data_path"]) / CONFIG["json_file"]
with open(json_file_path, "r", encoding="utf-8") as f:
    papers_data = json.load(f)

df_papers = pd.DataFrame(papers_data)
df_papers.fillna("", inplace=True)

# Create a combined text field for search
df_papers["combined_text"] = df_papers.apply(
    lambda x: f"{x['title']} {x['abstract']}", axis=1
)

print(f"Loaded {len(df_papers)} IR papers")
print(f"Sample paper: {df_papers.iloc[0]['title'][:50]}...")

# 1. BM25 Indexing for papers
def tokenize_corpus(texts):
    return bm25s.tokenize([str(t).lower()[:2000] for t in texts], stopwords="en")
print("Indexing IR Papers (BM25)...")
papers_tokens = tokenize_corpus(df_papers["combined_text"].tolist())
retriever_papers = bm25s.BM25()
retriever_papers.index(papers_tokens)
del papers_tokens
gc.collect()

# 2. Dense Indexing for papers
print("Building Dense Index for IR Papers...")
if CONFIG.get("use_api_emb"):
    print(f"Using API for embeddings ({CONFIG['model_emb']})")
    model_emb_papers = APIEmbeddingModel(
        model_name=CONFIG["model_emb"],
        api_key=CONFIG["api_key_emb"],
        base_url=CONFIG["base_url_emb"]
    )
else:
    print(f"Using Local Model for embeddings ({CONFIG['model_emb']})")
    model_emb_papers = SentenceTransformer(
        CONFIG["model_emb"],
        device=CONFIG["device"],
        model_kwargs={"torch_dtype": torch.float16 if "cuda" in str(CONFIG["device"]) else torch.float32},
    )

# Limit to 2000 characters for API compatibility
truncated_papers = [str(t)[:2000] for t in df_papers["combined_text"].tolist()]

papers_embeddings = model_emb_papers.encode(
    truncated_papers,
    batch_size=CONFIG["batch_size"] if not CONFIG.get("use_api_emb") else 32,
    convert_to_tensor=False,
    normalize_embeddings=True,
    show_progress_bar=True,
)
print("Papers Index Built.")

class IRPapersRetriever:
    def __init__(self):
        if CONFIG.get("use_api_rerank"):
            print(f"Using API Reranker ({CONFIG['model_rerank']})...")
            self.reranker = APIReranker(
                model_name=CONFIG["model_rerank"],
                api_key=CONFIG["api_key_rerank"],
                base_url=CONFIG["base_url_rerank"]
            )
        else:
            print(f"Loading local Reranker ({CONFIG['model_rerank']})...")
            self.reranker = CrossEncoder(
                CONFIG["model_rerank"],
                device=CONFIG["device"],
                max_length=512,
                automodel_args={"dtype": torch.float16 if "cuda" in str(CONFIG["device"]) else torch.float32},
            )

    def hybrid_search_papers(self, query, top_k=CONFIG["top_k_retrieval"]):
        # Dense Search
        query_emb = model_emb_papers.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )

        # CPU Search
        hits = util.semantic_search(query_emb, papers_embeddings, top_k=top_k)[0]
        dense_res = {df_papers.iloc[h["corpus_id"]]["arxiv_id"]: h["score"] for h in hits}

        # Sparse Search (BM25)
        query_tokens = bm25s.tokenize([query.lower()], stopwords="en")
        docs, scores = retriever_papers.retrieve(query_tokens, k=top_k)
        sparse_res = {
            df_papers.iloc[docs[0][i]]["arxiv_id"]: scores[0][i]
            for i in range(len(docs[0]))
        }

        # Fusion
        fused = {}
        for arxiv_id in set(dense_res) | set(sparse_res):
            r_dense = list(dense_res).index(arxiv_id) if arxiv_id in dense_res else 2 * top_k
            r_sparse = list(sparse_res).index(arxiv_id) if arxiv_id in sparse_res else 2 * top_k
            fused[arxiv_id] = (1 / (60 + r_dense)) + (1 / (60 + r_sparse))

        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def rerank(self, query, candidates, top_k=CONFIG["top_k_final"]):
        pairs = []
        valid_ids = []
        paper_map = dict(zip(df_papers["arxiv_id"], df_papers["combined_text"]))

        for arxiv_id, _ in candidates:
            text = paper_map.get(arxiv_id, "")
            if text:
                pairs.append([query, str(text)[:2000]])
                valid_ids.append(arxiv_id)

        if not pairs:
            return []

        scores = self.reranker.predict(
            pairs, batch_size=CONFIG["batch_size"], show_progress_bar=False
        )
        scored = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        return [c[0] for c in scored[:top_k]]

# Initialize the new retriever
papers_retriever = IRPapersRetriever()

def predict_papers(query):
    print(f"\nQuery: {query}")
    print("Searching...")

    # Hybrid Search
    candidates = papers_retriever.hybrid_search_papers(query, top_k=CONFIG["top_k_retrieval"])
    print(f"Found {len(candidates)} candidates")

    # Rerank
    if CONFIG.get("use_api_rerank"):
        print("Reranking with API...")
    else:
        print("Reranking locally...")

    top_papers = papers_retriever.rerank(query, candidates, top_k=CONFIG["top_k_final"])

    # Display Results
    print(f"\nTop {len(top_papers)} Results:")
    print("-" * 50)

    for i, arxiv_id in enumerate(top_papers, 1):
        paper = df_papers[df_papers["arxiv_id"] == arxiv_id].iloc[0]
        print(f"{i}. {paper['title']}")
        print(f"   ArXiv ID: {arxiv_id}")
        print(f"   Authors: {paper['authors']}")
        print(f"   Abstract: {paper['abstract'][:200]}...")
        print(f"   URL: https://arxiv.org/abs/{arxiv_id}")
        print()

# Interactive Query Loop
if __name__ == "__main__":
    print("IR Papers Retrieval System")
    print("Type 'quit' to exit")

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not query:
            continue

        try:
            predict_papers(query)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()