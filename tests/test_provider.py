import json

import httpx
import pytest
from PIL import Image

from track_game.provider import OpenRouterValidationError, OpenRouterVLMProvider


def response_payload(frame_id=450):
    return {
        "frame_id": frame_id,
        "players": [
            {
                "detection_id": "player_01",
                "team": "A",
                "box": {"left": 0.1, "top": 0.2, "right": 0.2, "bottom": 0.8},
                "foot": {"x": 0.15, "y": 0.8},
                "confidence": 0.9,
                "team_confidence": 0.95,
            }
        ],
        "ball": {
            "center": {"x": 0.16, "y": 0.75},
            "box": None,
            "confidence": 0.8,
        },
        "possession": {"player_detection_id": "player_01", "confidence": 0.7},
        "uncertainty_notes": [],
    }


def test_openrouter_adapter_makes_one_mocked_call_and_records_usage():
    calls = []

    def handler(request):
        calls.append(request)
        sent = json.loads(request.content)
        assert sent["model"] == "google/gemini-3.1-flash-lite"
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["response_format"]["json_schema"]["strict"] is True
        assert sent["reasoning"]["effort"] == "minimal"
        image_url = sent["messages"][0]["content"][1]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        return httpx.Response(
            200,
            json={
                "id": "mock-request",
                "model": "google/gemini-3.1-flash-lite",
                "provider": "mock-provider",
                "choices": [{"message": {"content": json.dumps(response_payload())}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cost": 0.0001,
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterVLMProvider("not-a-real-key", client=client)
    result = provider.request_detection(450, Image.new("RGB", (32, 32), "green"))
    client.close()
    assert len(calls) == 1
    assert result.detection.players[0].possesses_ball
    assert result.usage.cost_usd == pytest.approx(0.0001)
    assert result.usage.reasoning_tokens == 10


def test_openrouter_adapter_never_retries_mocked_failure():
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(503, json={"error": "temporary"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterVLMProvider("not-a-real-key", client=client)
    with pytest.raises(RuntimeError, match="no retry"):
        provider.request_detection(450, Image.new("RGB", (32, 32)))
    client.close()
    assert call_count == 1


def test_schema_failure_retains_usage_and_raw_response():
    invalid = response_payload()
    del invalid["players"][0]["foot"]

    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "bad-schema",
                "model": "google/gemini-3.1-flash-lite",
                "provider": "mock-provider",
                "choices": [{"message": {"content": json.dumps(invalid)}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cost": 0.0001,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterVLMProvider("not-a-real-key", client=client)
    with pytest.raises(OpenRouterValidationError) as caught:
        provider.request_detection(450, Image.new("RGB", (32, 32)))
    client.close()
    assert caught.value.usage.cost_usd == pytest.approx(0.0001)
    assert caught.value.raw_content == json.dumps(invalid)


def test_repository_env_loader_requires_local_env_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Create"):
        OpenRouterVLMProvider.from_repository_env(tmp_path)
