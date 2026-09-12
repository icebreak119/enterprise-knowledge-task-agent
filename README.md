# Enterprise Agent

企业智能知识与任务执行 Agent

面向企业内部知识、客户/订单/业务数据的任务执行型 Agent：不仅回答问题，还能自主决定何时检索知识、查询业务数据、调用工具、继续执行或请求人工确认。

> 目标技术栈：Python 3.12 / FastAPI / LangGraph / PostgreSQL + pgvector / Redis / MCP
> 当前进度：**Day 1 — 项目初始化 + FastAPI + PostgreSQL**（最小可运行骨架）

## 目录结构

```
enterprise-agent/
├── app/
│   ├── api/            # HTTP 路由（routes_chat.py 等）
│   ├── core/           # 配置（config.py）
│   ├── db/             # 连接、建表、种子数据
│   ├── models/         # SQLAlchemy 模型（8 张表）
│   └── schemas/        # Pydantic 请求/响应模型
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
- 最小聊天接口：`POST /api/chat`，体：`{"message": "hello"}`

## 不使用 Docker 的本地开发

若 Docker 不可用，可用本机已有的 PostgreSQL（需 16+）：

```bash
# 1) 创建角色与库（trust 认证时无需密码）
psql -U postgres -h localhost -c "CREATE ROLE agent LOGIN PASSWORD 'agent' SUPERUSER;"
psql -U postgres -h localhost -c "CREATE DATABASE enterprise_agent OWNER agent;"

# 2) 按上文步骤 2–5 初始化并启动
```

`init_db` 会尝试 `CREATE EXTENSION IF NOT EXISTS vector`；**若实例未安装 pgvector**（本机默认即如此），
会输出一条 warning 并跳过 `document_chunks` 表，其余 7 张表照常建立，服务可以正常启动。
需要向量检索时请用 `docker compose` 提供的 `pgvector/pgvector:pg16` 镜像，或为本机实例安装 pgvector 插件。

## 里程碑

- [x] Day 1：项目初始化 + FastAPI + PostgreSQL
- [ ] Day 2：LLM Provider + Structured Output
- [ ] Day 3：Tool Calling
- [ ] Day 4：LangGraph State / Node / Edge
- [ ] Day 5：Router + Executor + Validator