import app.api.routes_chat as routes_chat
from app.api.deps import get_llm
from app.api.routes_chat import REFUSAL_MESSAGE
from app.llm.base import LLMResponse
from app.main import app
from app.rag.retriever import RetrievedChunk
from app.schemas.intent import Intent, IntentResult


def test_chat_returns_intent(client):
    response = client.post("/api/chat", json={"message": "差旅报销标准是什么"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "knowledge_qa"
    assert body["reply"]


def test_chat_flags_task_execution(client):
    response = client.post("/api/chat", json={"message": "帮我取消这张订单"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "task_execution"
    assert body["need_human"] is True


def test_chat_rejects_empty_message(client):
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_knowledge_question_refuses_when_nothing_retrieved(client):
    """库里没有足够相近的资料时，后端直接拒答，不把判断交给模型。"""
    response = client.post("/api/chat", json={"message": "差旅报销标准是什么"})
    body = response.json()

    assert body["refused"] is True
    assert body["reply"] == REFUSAL_MESSAGE
    assert body["citations"] == []
    assert body["model"] is None  # 拒答没有调模型


def test_retrieval_skipped_for_non_knowledge_intent(client, monkeypatch):
    """非知识类问题不该白跑一次 embedding 检索。"""
    calls: list[str] = []

    async def spy(session, query, **kwargs):
        calls.append(query)
        return []

    monkeypatch.setattr(routes_chat, "search", spy)
    client.post("/api/chat", json={"message": "帮我取消这张订单"})
    assert calls == []


def test_citations_backfilled_from_answer_markers(client, monkeypatch):
    """回答里标了哪个编号，就只回贴哪个切片——没标的不算引用。"""

    class _CitingProvider:
        model = "stub-citing"

        async def achat(
            self, messages, *, response_model=None, tools=None, temperature=None,
            max_tokens=None,
        ):
            if response_model is not None:
                return LLMResponse(
                    content="",
                    parsed=IntentResult(
                        intent=Intent.KNOWLEDGE_QA, need_retrieval=True, confidence=0.9
                    ),
                    model=self.model,
                )
            return LLMResponse(
                content="旗舰型设备整机保修 36 个月 [12]，详见制度 [99]。",
                model=self.model,
            )

    def _chunk(chunk_id: int) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            content="条款内容",
            source="data/policies/demo.md",
            title="售后制度",
            section_path="二、保修服务政策 > 2.1 保修期限与范围",
            version="V2.0",
            distance=0.2,
        )

    async def fake_search(session, query, **kwargs):
        return [_chunk(12), _chunk(34)]

    monkeypatch.setattr(routes_chat, "search", fake_search)
    app.dependency_overrides[get_llm] = lambda: _CitingProvider()

    body = client.post("/api/chat", json={"message": "旗舰设备保修多久"}).json()

    assert body["refused"] is False
    # 12 被引用，34 未被引用，99 是模型编的编号、库里没有
    assert [c["chunk_id"] for c in body["citations"]] == [12]
    assert body["citations"][0]["version"] == "V2.0"
