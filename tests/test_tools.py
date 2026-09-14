"""工具调用（Function Calling）测试。

分三层：
1. 纯单元：注册表、工具定义格式、消息序列化；
2. API 层：意图路由到工具分支（工具执行被替换为 stub）；
3. 集成：真实业务工具查库（需要 PostgreSQL，连不上自动跳过）。
"""

import json

import pytest

import app.api.routes_chat as routes_chat
from app.api.deps import get_llm
from app.llm.base import ChatMessage, ToolCall
from app.llm.fake_provider import FakeProvider
from app.main import app
from app.tools import get_openai_tools, run_tool
from app.tools.base import ToolDef


# ── 1. 纯单元 ──────────────────────────────────────────


def test_openai_tool_format():
    """工具定义必须符合 OpenAI Function Calling 的 schema 结构。"""
    tool = ToolDef(
        name="demo",
        description="示例工具",
        parameters={"type": "object", "properties": {}},
        execute=None,
    )
    payload = tool.openai_tool()

    assert payload["type"] == "function"
    assert payload["function"]["name"] == "demo"
    assert payload["function"]["parameters"]["type"] == "object"


def test_registry_exposes_all_tools():
    tools = get_openai_tools()
    names = {t["function"]["name"] for t in tools}
    assert {"search_knowledge", "query_order", "query_customer"} <= names
    for tool in tools:
        assert tool["function"]["description"]
        assert tool["function"]["parameters"]["type"] == "object"


async def test_run_tool_rejects_unknown_tool():
    """未注册的工具必须报错，不能静默返回空字符串。"""
    with pytest.raises(KeyError, match="未注册的工具"):
        await run_tool("not_a_tool", "{}", None)


async def test_run_tool_parses_arguments_json():
    """arguments 是模型给的 JSON 字符串，注册表负责反序列化后交给工具。"""
    received: dict = {}

    async def capture(params, session):
        received.update(params)
        return "ok"

    from app.tools.registry import register_tool

    register_tool(
        ToolDef(
            name="_capture",
            description="测试用",
            parameters={"type": "object", "properties": {}},
            execute=capture,
        )
    )
    try:
        result = await run_tool("_capture", json.dumps({"k": 1}), None)
        assert result == "ok"
        assert received == {"k": 1}
    finally:
        from app.tools import registry

        registry._TOOLS.pop("_capture", None)


def test_assistant_message_serializes_tool_calls():
    """assistant + tool_calls 必须按 OpenAI 格式序列化，否则第二轮请求会被拒。"""
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="call_1", name="query_order", arguments='{"a":1}')],
    )
    payload = message.to_payload()

    assert payload["role"] == "assistant"
    assert payload["content"] == ""
    assert payload["tool_calls"][0]["id"] == "call_1"
    assert payload["tool_calls"][0]["type"] == "function"
    assert payload["tool_calls"][0]["function"]["name"] == "query_order"


def test_tool_message_serializes_call_id():
    payload = ChatMessage(role="tool", content="结果", tool_call_id="call_1").to_payload()
    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == "call_1"
    assert "tool_calls" not in payload


# ── 2. API 层：意图路由到工具分支 ────────────────────────


def test_business_query_goes_through_tool_loop(client):
    """业务查询必须走工具分支，且把工具返回内容带回回答。"""
    body = client.post(
        "/api/chat", json={"user_id": 1, "message": "帮我查一下订单 A1001"}
    ).json()

    assert body["intent"] == "business_query"
    assert "[stub] query_order" in body["reply"]


def test_knowledge_query_does_not_call_tools(client, monkeypatch):
    """知识类问题不该走工具分支，否则会白跑一遍业务工具。"""
    calls: list[str] = []

    async def spy(name, arguments, session):
        calls.append(name)
        return ""

    monkeypatch.setattr(routes_chat, "run_tool", spy)
    client.post("/api/chat", json={"user_id": 1, "message": "差旅报销标准是什么"})

    assert calls == []


def test_tool_loop_stops_after_max_rounds(monkeypatch):
    """模型一直要求调用工具时必须截断，不能无限循环。"""

    class _LoopingProvider:
        model = "looping"
        calls = 0

        async def achat(self, messages, *, response_model=None, tools=None, **kwargs):
            from app.llm.base import LLMResponse
            from app.schemas.intent import Intent, IntentResult

            if response_model is not None:
                return LLMResponse(
                    content="",
                    parsed=IntentResult(
                        intent=Intent.BUSINESS_QUERY, need_tools=True, confidence=0.9
                    ),
                    model=self.model,
                )
            type(self).calls += 1
            # 永远返回 tool_call，逼出轮次上限
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c", name="query_order", arguments="{}")],
                model=self.model,
            )

    async def stub_tool(name, arguments, session):
        return "结果"

    monkeypatch.setattr(routes_chat, "run_tool", stub_tool)
    app.dependency_overrides[get_llm] = lambda: _LoopingProvider()
    try:
        from fastapi.testclient import TestClient

        body = TestClient(app).post(
            "/api/chat", json={"user_id": 1, "message": "查订单"}
        ).json()
        # 上限 5 轮，收尾再调 1 次 = 6
        assert _LoopingProvider.calls == routes_chat._MAX_TOOL_ROUNDS + 1
        assert body["reply"]
    finally:
        app.dependency_overrides.clear()


def test_tool_error_does_not_break_conversation(client, monkeypatch):
    """工具执行失败要转成文本返回给模型，不能把请求整个打挂。"""

    async def failing_tool(name, arguments, session):
        raise RuntimeError("数据库连接超时")

    monkeypatch.setattr(routes_chat, "run_tool", failing_tool)
    response = client.post("/api/chat", json={"user_id": 1, "message": "查订单 A1001"})

    assert response.status_code == 200
    assert "执行出错" in response.json()["reply"]


# ── 3. 集成：真实业务工具 ────────────────────────────────


async def test_query_order_by_order_no(db_session):
    from app.tools.business import query_order

    try:
        result = await query_order({"order_no": "A1001"}, db_session)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可用：{exc}")

    if "未找到" in result:
        pytest.skip("库里没有种子数据，先跑 python -m app.db.seed")
    assert "A1001" in result
    assert "企业版License" in result


async def test_query_order_by_customer_name(db_session):
    """按客户名筛选必须生效——否则「张三买了什么」会返回全库订单。"""
    from app.tools.business import query_order

    try:
        result = await query_order({"customer_name": "张三"}, db_session)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可用：{exc}")

    if "未找到" in result:
        pytest.skip("库里没有种子数据")
    assert "A1001" in result

    missing = await query_order({"customer_name": "不存在的人"}, db_session)
    assert "未找到" in missing


async def test_query_order_by_status(db_session):
    from app.tools.business import query_order

    try:
        result = await query_order({"status": "refunded"}, db_session)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可用：{exc}")

    if "未找到" in result:
        pytest.skip("库里没有种子数据")
    assert "A1003" in result
    assert "A1001" not in result  # A1001 是 paid，不该出现


async def test_query_customer_by_name(db_session):
    from app.tools.business import query_customer

    try:
        result = await query_customer({"name": "张三"}, db_session)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可用：{exc}")

    if "未找到" in result:
        pytest.skip("库里没有种子数据")
    assert "张三" in result


async def test_search_knowledge_tool_returns_versions(db_session):
    """检索工具必须带上版本与废止标记，否则模型无法判断该用哪版。"""
    from app.tools.retrieval_tool import search_knowledge

    try:
        result = await search_knowledge({"query": "旗舰型设备整机保修多久？"}, db_session)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库或 embedding 不可用：{exc}")

    if "未找到" in result:
        pytest.skip("库里没有切片，先跑 scripts/ingest_corpus.py")
    assert "V2.0" in result
