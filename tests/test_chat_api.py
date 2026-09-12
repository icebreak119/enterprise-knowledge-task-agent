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
