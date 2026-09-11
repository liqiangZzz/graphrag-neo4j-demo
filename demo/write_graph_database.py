"""
写入图数据库
"""
import asyncio
import hashlib
import re
from typing import List

import unicodedata
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph, LLMGraphTransformer
from langchain_neo4j.graphs.graph_document import GraphDocument

from demo.load_wikipedia_datas import loaded_docs
from models.init_chat_model_llm import deepseek_llm_flash

# ============================================================
# 第一步：连接 Neo4j（默认数据库 neo4j）
# ============================================================
graph_db = Neo4jGraph(
    url='bolt://127.0.0.1:7687',
    username='neo4j',
    password='1qaz2wsx',
    database='neo4j',
    enhanced_schema=True,  # 是否启用增强模式
)

# 数据库的图结构
# schema = graph_db.schema
# print(schema)

# ============================================================
# 第二步：创建数据库唯一性约束（幂等操作）
# 在写入任何数据之前执行，确保数据一致性和去重
# ============================================================

# ---------- 1) 为实体节点创建唯一约束 ----------
# 作用：保证所有带 `__Entity__` 标签的节点，其 `id` 属性在全局是唯一的。
# 适用场景：后续通过 `MERGE` 操作写入实体时，如果 `id` 相同，则不会重复创建节点，而是匹配已有节点。
# 注意：`__Entity__` 是 LangChain-Neo4j 的保留标签，当 `baseEntityLabel=True` 时自动添加，
#       用于统一管理所有实体（人员、组织、概念等），便于全局索引和查询优化。
graph_db.query("""
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:`__Entity__`) REQUIRE e.id IS UNIQUE
""")


# ---------- 2) 为文档节点创建唯一约束 ----------
# 作用：保证所有带 `Document` 标签的节点，其 `doc_id` 属性在全局是唯一的。
# 适用场景：每个 `Document` 节点代表一个**完整的源文档**（如一篇维基百科文章、一个 PDF 文件等），
#          `doc_id` 用于标识该文档的唯一性（通常取 URL 的 MD5 或自定义 ID）。
# 为什么不直接用 `id`？因为 `id` 已被 `__Entity__` 占用，且 `Document` 是独立的节点类型，
#       用 `doc_id` 明确区分，避免与实体节点混淆。
# graph_db.query("""
# CREATE CONSTRAINT doc_id IF NOT EXISTS
# FOR (d:Document) REQUIRE d.doc_id IS UNIQUE
# """)


# ============================================================
# 第三步：正规化与稳定 ID
# ============================================================
def stable_doc_id(mata: dict) -> str:
    """
    选用 WikipediaLoader 的 metadata['source']（URL）作为稳定标识；
    若没有，则回退到 (title + summary) 的 hash。
    """
    base = mata.get('source') or (mata.get('title', '') + "|" + mata.get('summary', ''))
    return hashlib.md5(base.encode('utf-8')).hexdigest()


def attach_doc_ids(docs: List[Document]) -> List[Document]:
    """
    为文档列表中的每个文档添加稳定 ID，并统一下元数据格式。
    """
    for doc in docs:
        # copy, 避免 in-place 副作用
        doc.metadata = dict(doc.metadata)
        doc.metadata['id'] = stable_doc_id(doc.metadata)
        # 统一下元数据，便于后续查询
        doc.metadata.setdefault("source_type", "wikipedia")
        doc.metadata.setdefault("title", doc.metadata.get("title", ""))
        doc.metadata.setdefault("summary", doc.metadata.get("summary", ""))
    return docs


def normalize_id(s: str) -> str:
    """
    统一节点 ID：全角半角、空白、大小写等做轻度归一；
    中文不做 lower() 强制，但保留 NFKC 规整和空白规整。
    """

    if not isinstance(s, str):
        return s

    # 兼容性组合形式（Normalization Form KC），分解字符并替换兼容字符到其规范形式
    # "Ｈｅｌｌｏ　Ｗｏｒｌｄ！"  ---> "Hello World!"
    s = unicodedata.normalize('NFKC', s)
    # 将所有连续的空白字符替换为单个空格 例如 "Hello   \nWorld" → "Hello World"
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# 第四步：LLM 图谱抽取实体和关系
# 若通过 strict_mode=False 禁用严格模式，模型可能保留部分未定义的节点类型，但这种情况会导致输出不可预测，且可能引入无关实体
# 风险：违反业务逻辑的节点（如将“天气”错误归类为 Person）可能混入图谱，需额外后处理清洗
# ============================================================

# 允许的节点类型
allowed_nodes = [
    "Person", "Organization", "Company", "Product", "Event",
    "Country", "City", "University", "Award", "Position"
]

# 关系名建议用大写蛇形，便于统一（示例覆盖“马云/阿里巴巴”常见关系）
allowed_relationships = [
    ("Person", "FOUNDER_OF", "Organization"),
    ("Person", "FOUNDER_OF", "Company"),
    ("Person", "CEO_OF", "Company"),
    ("Person", "CHAIRMAN_OF", "Company"),
    ("Person", "BORN_IN", "City"),
    ("Person", "CITIZEN_OF", "Country"),
    ("Person", "EDUCATED_AT", "University"),
    ("Organization", "HEADQUARTERED_IN", "City"),
    ("Company", "HEADQUARTERED_IN", "City"),
    ("Organization", "PART_OF", "Organization"),
    ("Company", "OWNS", "Company"),
    ("Company", "ACQUIRED", "Company"),
    ("Person", "AWARDED", "Award"),
    ("Person", "FOUNDED", "Company"),  # 兼容部分模型输出
    ("Organization", "FOUNDED_BY", "Person"),
    ("Company", "FOUNDED_BY", "Person"),
    ("Organization", "PARTNERED_WITH", "Organization"),
]

# 也可以限定要抽取的一些关键属性（能抽到更稳定的主键/查询特征）
node_properties = ["alias", "dob", "founded_year", "citizenship", "ticker", "hq_city", "hq_country"]

llm_transformer = LLMGraphTransformer(
    llm=deepseek_llm_flash,
    allowed_nodes=allowed_nodes,  # 允许的节点类型
    allowed_relationships=allowed_relationships,  # 允许的关系类型
    node_properties=node_properties,  # 允许的节点属性
)


async def extract_graph_documents(docs: List[Document]) -> List[GraphDocument]:
    """
     将文档列表转换为图结构，并返回包含图结构信息的文档列表。
     Args:
        docs: 文档列表
    Returns:
        List[GraphDocument]: 包含图结构信息的文档列表
    """

    # 调用 LLM 抽取图结构
    # graph_docs 列表，每个元素是一个 GraphDocument，包含图结构信息
    graph_docs = await llm_transformer.aconvert_to_graph_documents(docs)

    # 统一节点 ID（避免“阿里 巴巴” vs “阿里巴巴集团”等微小差异导致拆点）
    for graph_doc in graph_docs:
        # 统一节点 ID
        # 规整所有实体节点的名称
        for node in graph_doc.nodes:
            node.id = normalize_id(node.id)

        # 统一关系 ID
        # 规整关系中源实体和目标实体的名称
        for relationship in graph_doc.relationships:
            relationship.source.id = normalize_id(relationship.source.id)
            relationship.target.id = normalize_id(relationship.target.id)

    return graph_docs


# ============================================================
# 第五步：幂等增量写入（含来源 Document 溯源）
# ============================================================
def upsert_to_neo4j(graph_documents):
    """
    include_source=True: 会把来源 Document 导入为 (:Document {id})，
    并用 (:Document)-[:MENTIONS]->(实体) 连接；
    baseEntityLabel=True: 给实体加二级标签 __Entity__，结合唯一约束，提升合并与查询性能。
    """

    graph_db.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,   # 添加 __Entity__ 次级标签，用于索引优化
        include_source=True     # 包含来源 Document 节点，并连接关系
    )


# ============================================================
# 第六步：入口：从你已有的 Wikipedia 文档持续更新
# ============================================================
async def ingest_wikipedia_docs(docs: List[Document]):
    docs = attach_doc_ids(docs)
    gdocs = await extract_graph_documents(docs)
    print(gdocs)
    upsert_to_neo4j(gdocs)

# ==== 使用举例 ====
# from langchain_community.document_loaders import WikipediaLoader
# docs = WikipediaLoader(query="马云", lang="zh", load_max_docs=5).load()
asyncio.run(ingest_wikipedia_docs(loaded_docs))