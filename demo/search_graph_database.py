import json

from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from models.init_chat_model_llm import deepseek_llm_flash, deepseek_llm_pro

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

# ============================================================
# 第二步：定义提示词模板
# ============================================================

# 定义Cypher提示词模板
cypher_prompt = PromptTemplate(
    input_variables=['schema', 'question'],
    template=(
        "根据以下 Neo4j 图 schema，生成 Cypher 查询语句，"
        "节点必须优先使用 id 属性进行匹配，不得使用 alias。 如果需要查询多个实体，别名都要不一样。 \n"
        "{schema}\n"
        "问题：{question}\n"
        "示例：MATCH (p:Person {{id: '马云'}})-[r]-(x) RETURN p, r, x LIMIT 5 请基于此示例生成 Cypher 查询，用于返回节点、关系和连接节点。不要只返回 p。"
        "请只输出 Cypher 语句。"
    )
)

#  定义QA提示词模板，用于生成最终的查询结果
qa_prompt = PromptTemplate(
    input_variables=['question', 'result'],
    template=(
        "下面是针对用户问题 `{question}`，通过 Neo4j 查询得到的结果来回答问题，不要编造。"
    )
)

# ============================================================
# 第三步：创建一个搜索对象
# ============================================================

# 方式一：单模型标准问答模式（最常用）
# ============================================================
# 核心流程：LLM 调用两次——① 生成 Cypher 查询；② 根据查询结果生成自然语言答案。
# 适用场景：常规问答对话，对答案质量要求一般，追求简洁和成本控制。
# 输出格式：直接返回最终答案（字符串）。
# runnable = GraphCypherQAChain.from_llm(
#     llm=deepseek_llm_flash,
#     graph=graph_db,
#     verbose=True, # 是否显示详细信息【打印详细执行日志(如生成的Cypher语句和查询结果)】
#     cypher_prompt=cypher_prompt, # 提示词模板
#     qa_prompt=qa_prompt,  # QA提示词模板
#     validate_cypher=True,  # 是否验证Cypher查询语句
#     allow_dangerous_requests=True, # 是否允许危险请求
#     return_intermediate_steps=True,  # 是否返回中间步骤
# )

# 方式二：单模型工具调用模式（适合 Agent 工作流）
# ============================================================
# 核心流程：LLM 调用两次——① 生成 Cypher 查询；② 将查询结果封装为 ToolMessage（工具消息），
#          而不是直接生成自然语言答案。最终返回包含结构化数据的 ToolMessage。
# 适用场景：作为 LangGraph / Agent 链条的一部分，查询结果需传递至下游工具或子链处理。
# 输出格式：返回 ToolMessage 对象（含结构化数据），而非纯文本答案。
# 参数说明：use_function_response=True 启用该模式（需确认当前版本是否支持，若报错可移除）。

# runnable = GraphCypherQAChain.from_llm(
#     llm=deepseek_llm_flash,
#     graph=graph_db,
#     verbose=True, # 是否显示详细信息【打印详细执行日志(如生成的Cypher语句和查询结果)】
#     cypher_prompt=cypher_prompt, # 提示词模板
#     validate_cypher=True,  # 是否验证Cypher查询语句
#     allow_dangerous_requests=True, # 是否允许危险请求
#     return_intermediate_steps=True,  # 是否返回中间步骤
#     top_k=30,
#     use_function_response=True # 是否使用函数响应，这个参数在较新版本中可能已移除，建议移除或查阅文档确认
# )


# 方式三：双模型分离 + 工具调用模式（最强配置）生产级推荐
# ============================================================
# 将 "Cypher 生成" 与 "结果理解" 解耦，为不同任务分配最适配的模型。同时将查询结果封装为结构化 ToolMessage，使该链可作为 Agent 的工具节点。
# 执行流程：
#   ① cypher_llm（低成本模型）生成 Cypher 查询语句
#   ② 执行查询，从 Neo4j 获取原始结构化数据
#   ③ 链将原始数据与用户问题组合成上下文，然后调用 qa_llm（高能力模型），
#      让 qa_llm 理解这些数据并生成一个结构化的 ToolMessage（而非自然语言文本）。
#   ④ 返回 ToolMessage 对象（含 content / tool_call_id / name），供下游 Agent 或系统解析。

# 返回值：resp["answer"] 为 ToolMessage 对象，而非纯文本字符串。
# 参数说明：
#   - cypher_llm / qa_llm：分离模型，按任务特性选择（如 cypher 用快模型，qa 用强模型）
#   - use_function_response=True：启用工具消息输出模式（qa_llm 输出 ToolMessage）
#   - top_k：控制查询返回的最大记录数
#
# 适用场景：对查询质量要求高，且希望精细控制两个步骤的模型能力/成本，适用于复杂业务逻辑或 Agent 编排。
# 参数说明：cypher_llm 和 qa_llm 分开指定；use_function_response 用于工具模式。

runnable = GraphCypherQAChain.from_llm(
    cypher_llm=deepseek_llm_flash,
    qa_llm=deepseek_llm_pro,
    graph=graph_db,
    verbose=True, # 是否显示详细信息【打印详细执行日志(如生成的Cypher语句和查询结果)】
    cypher_prompt=cypher_prompt, # 提示词模板
    qa_prompt=qa_prompt,  # QA提示词模板
    validate_cypher=True,  # 是否验证Cypher查询语句
    allow_dangerous_requests=True, # 是否允许危险请求
    return_intermediate_steps=True,  # 是否返回中间步骤
    top_k=30,
    use_function_response=True # 是否使用函数响应，这个参数在较新版本中可能已移除，建议移除或查阅文档确认
)

# ============================================================
# 第四步：执行查询
# ============================================================
resp = runnable.invoke({"query": "马化腾是谁?"})
#resp = runnable.invoke({"query": "马云和淘宝网的关系?"})
# resp = runnable.invoke({"query": "马云和马化腾的关系?"})
print(resp)


# ============================================================
# 第二、三步骤 获取 ToolMessage 对象
# ============================================================
# 提取原始上下文（根据你的打印结构）
intermediate = resp.get("intermediate_steps", [])

# intermediate 可能是列表，也可能是字典，取决于版本
# 如果输出显示 intermediate_steps 是列表，第二个元素是 {'context': [...]}
if isinstance(intermediate, list) and len(intermediate) >= 2:
    context_data = intermediate[1].get("context", [])
else:
    context_data = []  # 降级方案

# 构造 ToolMessage
tool_msg = ToolMessage(
    content=json.dumps(context_data, ensure_ascii=False),
    tool_call_id="cypher_query_001",
    name="neo4j"
)

# 使用 tool_msg
print("ToolMessage 内容:", tool_msg.content)
print("工具名称:", tool_msg.name)