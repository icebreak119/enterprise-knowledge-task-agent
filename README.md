# Enterprise Agent

企业智能知识与任务执行 Agent

面向企业内部知识、客户/订单/业务数据的任务执行型 Agent：不仅回答问题，还能自主决定何时检索知识、查询业务数据、调用工具、继续执行或请求人工确认。

> 目标技术栈：Python 3.12 / FastAPI / LangGraph / PostgreSQL + pgvector / Redis / MCP
> 当前进度：**Day 2 — LLM Provider + Structured Output**（已接入阿里云百炼 Qwen）

## Day 2：LLM 接入

统一走 **OpenAI 兼容协议**，自研 Provider 抽象（`app/llm/base.py` 的 `LLMProvider` Protocol），
不引入 LangChain。换厂商只需改 `.env` 里的 `LLM_BASE_URL` / `LLM_MODEL`：
OpenAI 官方、智谱 GLM、DeepSeek、通义、本地 vLLM 全部同一套代码。

```
app/llm/
├── base.py            # ChatMessage / ToolCall / LLMResponse / LLMProvider Protocol
├── openai_provider.py # OpenAI 兼容实现：重试 + 结构化输出降级链
├── fake_provider.py   # 离线兜底：关键词规则，无 Key 也能跑
├── errors.py          # LLMError / LLMConnectionError / LLMRateLimitError / LLMParseError
└── factory.py         # 按配置选择实现（有 Key 用真的，无 Key 用 Fake）
```

**结构化输出三级降级**：`response_format=json_schema`（`completions.parse`）→
`json_object` + JSON Schema 提示 + 手动 `model_validate_json` → 抛 `LLMParseError`。
阿里云百炼支持原生 `json_schema`，走最优路径；智谱 GLM 不接受，会自动落到第二级，
日志有一行 warning，属正常现象。

**重试策略**：tenacity `AsyncRetrying`，只对连接/超时/429/5xx 重试，指数退避 2–30s；
关闭 SDK 自带重试（`max_retries=0`）避免重试次数翻倍。
注意：429 属可恢复错误，不会把 `json_schema` 永久标记为不支持。

### 推理模型注意事项（踩过的坑）

推理模型（GLM-4.x / Qwen3）默认先输出思维链，可能把 `max_tokens` 吃光导致 `content` 为空，
表现为 `LengthFinishReasonError` 或「模型输出无法解析」；即使不空也很浪费——
实测 Qwen 回答两个字烧掉 194 个思维 token。解决办法：

```ini
LLM_MAX_TOKENS=2048                                  # 留足余量
LLM_EXTRA_BODY={"enable_thinking": false}            # 阿里云 Qwen3
# LLM_EXTRA_BODY={"thinking": {"type": "disabled"}}  # 智谱 GLM
```

`LLM_EXTRA_BODY` 是厂商私有参数的通用逃生舱（JSON 对象，原样并入请求体）——
**各家关闭思维链的字段并不统一**，写死一种会绑死厂商，所以留成配置而非代码分支。

### 厂商切换记录

| 厂商 | base_url | 模型 | 结构化输出 | 限流 |
| --- | --- | --- | --- | --- |
| 阿里云百炼（当前） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.6-flash` | 原生 `json_schema` | 未遇到 |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4/` | `glm-4.7-flash` | 降级到 `json_object` | 免费额度卡很死，429 频繁 |

GLM 免费额度会连续返回 `code 1305`（模型访问量过大）与 `1302`（账户速率限制），
`/api/chat` 一次要打两次模型（意图 + 作答），更容易触顶，故换为阿里云。
模型名注意是 `qwen3.6-flash`（**带连字符**），写成 `qwen3.6flash` 会 404；
可用模型列表：`GET https://dashscope.aliyuncs.com/compatible-mode/v1/models`。

### 安全：Key 不入库

真实 Key **只写进 `.env`**（已被 `.gitignore` 忽略），代码与 `.env.example` 中一律为空值。
提交前自检：

```bash
git grep -n "你的key前8位" -- .   # 应无输出
```

`.env` 中 `LLM_API_KEY` 留空时自动降级为 `FakeProvider`，服务照常启动、测试照常跑。

## 目录结构

```
enterprise-agent/
├── app/
│   ├── api/            # HTTP 路由（routes_chat.py 等）
│   ├── core/           # 配置（config.py）
│   ├── db/             # 连接、建表、种子数据
│   ├── llm/            # LLM 抽象与实现（Day 2）
│   ├── models/         # SQLAlchemy 模型（8 张表）
│   ├── prompts/        # 系统/意图/作答提示词
│   └── schemas/        # Pydantic 请求/响应模型
├── tests/              # pytest + pytest-asyncio（无需 Key）
├── docker-compose.yml  # PostgreSQL(pgvector) + Redis
├── .env.example
└── pyproject.toml
```

## 表设计

`users` / `customers` / `orders` / `conversations` / `messages` / `memories` / `documents` / `document_chunks`（含 pgvector `embedding` 字段）

## 快速开始

### 1. 启动依赖（PostgreSQL + Redis）

```bash
docker compose up -d
```

### 2. 初始化虚拟环境并安装依赖

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 3. 配置环境变量

```bash
copy .env.example .env
```

### 4. 写入演示数据（可选）

```bash
python -m app.db.seed
```

### 5. 启动服务

```bash
uvicorn app.main:app --reload
```

- 健康检查：http://localhost:8000/health
- 接口文档：http://localhost:8000/docs
- 聊天接口：`POST /api/chat`，体：`{"user_id": 1, "message": "差旅报销标准是什么？"}`

响应示例：

```json
{
  "conversation_id": null,
  "reply": "结论：当前资料中没有找到依据。...",
  "intent": "knowledge_qa",
  "need_human": false,
  "model": "qwen3.6-flash"
}
```

`POST /api/chat` 目前是两步式：先用 `temperature=0.0` 做意图分类（5 类：
`knowledge_qa` / `business_query` / `task_execution` / `human_handoff` / `chitchat`），
再生成回答；`task_execution` 与 `human_handoff` 会置 `need_human=true`。
Day 4 起这两步会由 LangGraph 编排，并接上真实检索与工具。

### 6. 跑测试（无需 API Key）

```bash
pytest -q
```

## 不使用 Docker 的本地开发

若 Docker 不可用，可用本机已有的 PostgreSQL（需 16+）：

```bash
# 1) 创建角色与库（trust 认证时无需密码）
psql -U postgres -h localhost -c "CREATE ROLE agent LOGIN PASSWORD 'agent' SUPERUSER;"
psql -U postgres -h localhost -c "CREATE DATABASE enterprise_agent OWNER agent;"

# 2) 按上文步骤 2–5 初始化并启动
```

本机实例已编译安装 **pgvector 0.7.4**，向量检索可用；编译步骤见 `docs/pgvector-windows-build.md`。

`init_db` 仍保留了降级逻辑：若换到没有 pgvector 的实例，扩展创建失败时会输出 warning 并跳过
`document_chunks`，其余 7 张表照常建立，服务依然能启动，只是没有向量能力。

## 里程碑

- [x] Day 1：项目初始化 + FastAPI + PostgreSQL
- [x] Day 2：LLM Provider + Structured Output（阿里云 Qwen 已跑通）
- [ ] Day 3：Tool Calling
- [ ] Day 4：LangGraph State / Node / Edge
- [ ] Day 5：Router + Executor + Validator