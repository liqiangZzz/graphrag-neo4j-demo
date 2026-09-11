"""
加载本地JSON文件中加载数据
"""

from langchain_community.document_loaders import JSONLoader


def metadata_func(record: dict, metadata: dict) -> dict:
    meta = record.get("metadata", {})
    metadata["title"] = meta.get("title")
    metadata["summary"] = meta.get("summary")
    metadata["source"] = meta.get("source")
    return metadata


# 从JSON加载Document
def load_docs_from_json(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        content_key="page_content",
        metadata_func=metadata_func,
        # text_content=False
    )
    return loader.load()


# 从JSON加载数据并保存为JSON
loaded_docs = load_docs_from_json("../data/alibaba_docs.json")
