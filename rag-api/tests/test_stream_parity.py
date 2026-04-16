import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_stream_equals_sync_for_mixed() -> None:
    payload = {
        "question": "How many PTO days do I have and what is the rollover policy?",
        "employee_id": "E001",
        "use_rag": True,
        "skip_cache": True,
    }

    sync_resp = client.post("/api/ask", json=payload)
    assert sync_resp.status_code == 200
    sync_answer = str(sync_resp.json().get("answer", "")).strip()

    stream_resp = client.post("/api/ask/stream", json=payload)
    assert stream_resp.status_code == 200
    sse_body = stream_resp.text

    final_payload: dict = {}
    for raw_line in sse_body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        evt = json.loads(line[6:])
        if evt.get("type") == "done":
            final_payload = dict(evt.get("final") or {})
            break

    stream_answer = str(final_payload.get("answer", "")).strip()
    assert stream_answer
    assert sync_answer in stream_answer or stream_answer in sync_answer
