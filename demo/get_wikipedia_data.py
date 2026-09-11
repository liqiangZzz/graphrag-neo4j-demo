"""
加载中文维基百科中「阿里巴巴」相关的文档，最多 3 篇。

为什么要这样写
--------------
直连 zh.wikipedia.org 时，底层 `wikipedia` 包拿到的往往不是 JSON 而是空响应或
HTML 错误页，于是抛出：
    requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
这是网络问题，不是代码问题。因此本脚本先做一次轻量连通性探测：
  - 可达  -> 正常走 WikipediaLoader；
  - 不可达 -> 回退读取本地缓存
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

QUERY = "阿里巴巴"
MAX_DOCS = 3
ROOT = Path(__file__).parent
CACHE_PATH = ROOT / "data" / "alibaba_docs.json"
PROBE_TIMEOUT = 8


def wiki_reachable(timeout: int = PROBE_TIMEOUT) -> bool:
    """探测中文维基 API 是否可用：能返回可解析的 JSON 才算通。"""
    import requests

    try:
        resp = requests.get(
            "https://zh.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": QUERY,
                "format": "json",
                "srlimit": 1,
            },
            headers={"User-Agent": "graphrag-neo4j-demo/1.0 (learning)"},
            timeout=timeout,
        )
        resp.raise_for_status()
        resp.json()  # 非 JSON 内容（HTML 拦截页）会在这里抛错
        return True
    except Exception as exc:  # 超时、连接失败、非 JSON 一律视为不可达
        print(f"[提示] 中文维基 API 不可达，改用本地缓存。原因：{type(exc).__name__}")
        return False


def load_online(query: str, max_docs: int) -> List[Document]:
    from langchain_community.document_loaders import WikipediaLoader

    return WikipediaLoader(query=query, lang="zh", load_max_docs=max_docs).load()


def load_from_cache(path: Path, max_docs: int) -> List[Document]:
    if not path.exists():
        print(
            f"\n[缺失] 本地缓存不存在：{path}\n"
            "请在可访问 zh.wikipedia.org 的网络环境下重新运行本脚本。\n",
            file=sys.stderr,
        )
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Document] = []
    for item in raw[:max_docs]:
        docs.append(
            Document(
                page_content=item["text"],
                metadata={
                    "title": item["title"],
                    "summary": item["summary"],
                    "source": item["url"],
                },
            )
        )
    return docs


def load_documents(query: str = QUERY, max_docs: int = MAX_DOCS) -> List[Document]:
    if wiki_reachable():
        return load_online(query, max_docs)
    return load_from_cache(CACHE_PATH, max_docs)


if __name__ == "__main__":
    docs: List[Document] = load_documents()
    print(f"加载的文档数量: {len(docs)}")
    for d in docs:
        print(f"  - {d.metadata.get('title')} ({len(d.page_content)} 字)")
